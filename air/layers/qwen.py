"""Low-rank SVD modules for Qwen2.

Qwen2 matches the Llama attention/MLP structure except q/k/v projections carry a bias
(config.attention_bias=True, set at load time). The config-driven Llama SVD modules already
route the original bias onto the up-projection, so we reuse them directly.
"""

from layers.llama import SVD_LlamaAttention as SVD_QwenAttention
from layers.llama import SVD_LlamaMLP as SVD_QwenMLP

__all__ = ["SVD_QwenAttention", "SVD_QwenMLP"]
