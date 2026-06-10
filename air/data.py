"""Calibration and evaluation data for AIR.

Calibration = random fixed-length windows of the training text (used for the merged
forward-backward analysis). Evaluation = contiguous fixed-length chunks of the test text
for perplexity. `--tiny` uses synthetic random token ids (no tokenizer / no download).
"""

import random
import torch


def _load_text(name, split, cache_dir):
    from datasets import load_dataset
    if "wikitext" in name:
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)["text"]
    if "ptb" in name:
        return load_dataset("ptb_text_only", "penn_treebank", split=split,
                            trust_remote_code=True, cache_dir=cache_dir)["sentence"]
    if "c4" in name:
        f = ("en/c4-train.00000-of-01024.json.gz" if split == "train"
             else "en/c4-validation.00000-of-00008.json.gz")
        return load_dataset("allenai/c4", data_files={"s": f}, split="s", cache_dir=cache_dir)["text"]
    raise ValueError(f"unknown dataset: {name}")


def get_calibration_data(args, tokenizer):
    """List of LongTensor [1, seqlen] — calibration windows for the analysis pass."""
    n = args.n_samples_calibration
    seqlen = args.relevance_subset_seq_len
    if args.tiny:
        seqlen = min(seqlen, 256)
        g = torch.Generator().manual_seed(args.seed)
        return [torch.randint(0, args.tiny_vocab, (1, seqlen), generator=g) for _ in range(n)]

    texts = _load_text(args.calibration_data, "train", args.cache_dir)
    tot = "\n\n".join(t for t in texts if t and not t.isspace())
    random.seed(args.seed)
    out = []
    for _ in range(n):
        i = random.randint(0, len(tot) - seqlen * 10 - 1)
        enc = tokenizer(tot[i:i + seqlen * 10], return_tensors="pt",
                        truncation=True, max_length=seqlen * 10)
        out.append(enc.input_ids[:, :seqlen])
    return out


def get_eval_batches(args, tokenizer):
    """List of LongTensor [batch, seqlen] — contiguous chunks for perplexity."""
    seqlen = args.eval_seqlen
    bs = args.batch_size
    if args.tiny:
        seqlen = min(seqlen, 256)
        g = torch.Generator().manual_seed(args.seed + 1)
        return [torch.randint(0, args.tiny_vocab, (bs, seqlen), generator=g) for _ in range(2)]

    name = args.evaluation_perplexity_data
    split = "validation" if "c4" in name else "test"
    ids = tokenizer("\n\n".join(_load_text(name, split, args.cache_dir)), return_tensors="pt").input_ids[0]
    n_chunks = len(ids) // seqlen
    chunks = torch.stack([ids[i * seqlen:(i + 1) * seqlen] for i in range(n_chunks)])
    return [chunks[i:i + bs] for i in range(0, n_chunks, bs)]
