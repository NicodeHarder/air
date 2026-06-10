"""Model + tokenizer loading for AIR.

We always load the **lxt-instrumented** model variant (lxt.models.{llama,mistral,qwen}),
because the merged forward-backward analysis needs the AttnLRP backward and the lxt model's
forward is numerically identical to the plain model. The same loaded model is used for
analysis, then compressed in place, then evaluated.

Memory: bf16 weights + HF device_map (multi-GPU split or CPU offload) so large models fit.
"""

import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerBase

# lxt model variants (forward + AttnLRP backward).  attnlrp = the LRP rule Composite.
from lxt.models.llama import LlamaForCausalLM as LxtLlama, attnlrp as attnlrp_llama
from lxt.models.mistral import MistralForCausalLM as LxtMistral, attnlrp as attnlrp_mistral
from lxt.models.qwen import QwenForCausalLM as LxtQwen, attnlrp as attnlrp_qwen


def model_family(model_name):
    n = model_name.lower()
    if "qwen" in n:
        return "qwen"
    if "mistral" in n:
        return "mistral"
    return "llama"


def get_attnlrp(model_name):
    return {"qwen": attnlrp_qwen, "mistral": attnlrp_mistral, "llama": attnlrp_llama}[model_family(model_name)]


def resolve_model_max_seqlen(config, sanity_cap=1_000_000):
    for c in (getattr(config, "max_position_embeddings", None),
              getattr(config, "n_positions", None),
              getattr(config, "seq_length", None)):
        if isinstance(c, int) and not isinstance(c, bool) and 0 < c <= sanity_cap:
            return c
    return None


def build_device_map(config, device, n_layers, single=False):
    """Decoder-layer -> device map. With `single` everything sits on one GPU (used for
    compression + evaluation); otherwise layers spread across all GPUs (used for the analysis
    backward, which needs the combined VRAM). HF/accelerate then holds layers per this map."""
    n_gpus = 0 if device == "cpu" else torch.cuda.device_count()
    if n_gpus == 0:
        cpu = torch.device("cpu")
        return {"model.embed_tokens": cpu, "model.rotary_emb": cpu, "model.norm": cpu,
                "lm_head": cpu, **{f"model.layers.{i}": cpu for i in range(n_layers)}}
    if single:
        n_gpus = 1                                # all modules on GPU 0
    per = n_layers // n_gpus
    rem = n_layers % n_gpus
    dm = {"model.embed_tokens": 0, "model.rotary_emb": 0}
    li = 0
    for g in range(n_gpus):
        for _ in range(per + (1 if g < rem else 0)):
            dm[f"model.layers.{li}"] = g
            li += 1
    dm["model.norm"] = n_gpus - 1
    dm["lm_head"] = 0 if getattr(config, "tie_word_embeddings", False) else n_gpus - 1
    return dm


_CFG_CACHE = {}   # model_name -> (cfg, fam, cache); the config is needed up to 3x per run


def _config_for(args):
    if args.model_name in _CFG_CACHE:
        return _CFG_CACHE[args.model_name]
    fam = model_family(args.model_name)
    cache = args.cache_dir                       # shared HF cache (weights + datasets); set in config
    os.makedirs(cache, exist_ok=True)
    if fam == "qwen":
        from transformers import Qwen2Config
        cfg = Qwen2Config.from_pretrained(args.model_name, token=args.hf_token, cache_dir=cache)
        cfg.attention_bias = True  # Qwen2 uses bias on q/k/v
    elif fam == "mistral":
        from transformers import MistralConfig
        cfg = MistralConfig.from_pretrained(args.model_name, token=args.hf_token, cache_dir=cache)
    else:
        cfg = LlamaConfig.from_pretrained(args.model_name, token=args.hf_token, cache_dir=cache)
    _CFG_CACHE[args.model_name] = (cfg, fam, cache)
    return _CFG_CACHE[args.model_name]


def num_layers(args):
    """Number of decoder layers without loading weights (to decide whether analysis is cached)."""
    return 2 if args.tiny else _config_for(args)[0].num_hidden_layers


def load_model_and_tokenizer(args, lxt=False, multi_gpu=False):
    """Load `args.model_name` (or a tiny synthetic Llama if --tiny).

    lxt=True       -> the AttnLRP-instrumented model (only needed for the 'relevances' analysis).
    multi_gpu=True -> spread across all GPUs (the analysis backward needs the combined VRAM);
                      otherwise everything on a single GPU (compression + evaluation).
    The two are independent: gradient-based analysis (weightxgrad / fisher) uses a plain model on
    multiple GPUs, so it passes lxt=False, multi_gpu=True.
    """
    if args.tiny:
        return _build_tiny(args, lxt)

    cfg, fam, cache = _config_for(args)
    cfg._attn_implementation = "eager"
    n_layers = cfg.num_hidden_layers
    device_map = build_device_map(cfg, args.device, n_layers, single=not multi_gpu)
    where = "multi-GPU" if multi_gpu else "single GPU"

    if lxt:
        cfg.svd_relevances = False        # plain lxt attention/MLP (LRP-capable)
        cls = {"qwen": LxtQwen, "mistral": LxtMistral, "llama": LxtLlama}[fam]
        model = cls.from_pretrained(
            args.model_name, token=args.hf_token, torch_dtype=torch.bfloat16,
            device_map=device_map, cache_dir=cache, config=cfg, attn_implementation="eager",
        )
        print(f"\nLoaded lxt {fam} model for analysis ({where}): {args.model_name} ({n_layers} layers)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, token=args.hf_token, torch_dtype=torch.bfloat16,
            device_map=device_map, cache_dir=cache, attn_implementation="eager",
        )
        purpose = "analysis" if multi_gpu else "compression/eval"
        print(f"\nLoaded plain {fam} model for {purpose} ({where}): {args.model_name} ({n_layers} layers)")
    model.eval()
    args.device_map = getattr(model, "hf_device_map", device_map)

    tok = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token, cache_dir=cache)
    if not isinstance(tok, PreTrainedTokenizerBase):
        # transformers silently returns False (not a tokenizer) when the slow->fast SentencePiece
        # conversion fails. For SPM-only repos (no tokenizer.json, e.g. jeffwan/llama-7b-hf) under
        # protobuf>=4 the vendored sentencepiece_model_pb2 fails to import; the slow tokenizer reads
        # tokenizer.model via the sentencepiece C++ lib directly, so it loads fine.
        print(f"[tokenizer] fast tokenizer unavailable for {args.model_name}; falling back to slow")
        tok = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token,
                                            cache_dir=cache, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ctx = resolve_model_max_seqlen(cfg)
    if ctx is not None:
        for a in ("relevance_subset_seq_len", "eval_seqlen"):
            if getattr(args, a) > ctx:
                print(f"[seqlen] capping --{a} {getattr(args, a)} -> {ctx}")
                setattr(args, a, ctx)
    return model, tok


def tiny_config():
    """The architecture used by --tiny (shared by model build and the baseline metric counter)."""
    cfg = LlamaConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _build_tiny(args, lxt):
    """Tiny randomly-initialized Llama for instant smoke tests (no download). The lxt and plain
    builds share weights via a saved state_dict so analysis (lxt) and compression (plain) agree."""
    cfg = tiny_config()
    if lxt:
        cfg.svd_relevances = False
    dev = args.device if (torch.cuda.is_available() and args.device != "cpu") else "cpu"
    sd_path = os.path.join(args.analysis_dir, "_tiny_state_dict.pt")
    if lxt:
        cls = LxtLlama
    else:
        from transformers import LlamaForCausalLM as HFLlama
        cls = HFLlama
    # First builder (whether baseline, lxt-analysis, or plain-compress) creates the shared weights;
    # the others load them, so all phases agree on the same random tiny model.
    if os.path.exists(sd_path):
        model = cls(cfg).to(torch.float32).to(dev)
        model.load_state_dict(torch.load(sd_path, map_location=dev), strict=False)
    else:
        torch.manual_seed(args.seed)
        model = cls(cfg).to(torch.float32).to(dev)
        torch.save(model.state_dict(), sd_path)
    model.eval()
    args.tiny_vocab = cfg.vocab_size
    args.device_map = {f"model.layers.{i}": (0 if dev != "cpu" else "cpu") for i in range(cfg.num_hidden_layers)}
    print(f"\nBuilt tiny {'lxt' if lxt else 'plain'} Llama (2 layers, hidden 128)")
    return model, None  # no tokenizer; --tiny uses synthetic random-id data


def get_decoder_layers(model):
    return model.model.layers
