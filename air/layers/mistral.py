"""Low-rank SVD modules for Mistral.

Mistral's attention/MLP have the same structure as Llama (RoPE + GQA + SwiGLU MLP; no q/k/v
bias). The Llama SVD modules are fully config-driven, so we reuse them. The sliding-window
mask is applied by the model via the passed attention_mask, so no change is needed here.
"""

from layers.llama import SVD_LlamaAttention as SVD_MistralAttention
from layers.llama import SVD_LlamaMLP as SVD_MistralMLP

__all__ = ["SVD_MistralAttention", "SVD_MistralMLP"]
