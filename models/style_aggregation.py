"""Style Aggregation module operating in the Fourier domain.

Inputs
------
content_fft : torch.Tensor (complex) shape (B, L, D)
    Output of previous attention layer; we use its amplitude & phase.
style_amp : torch.Tensor (real) shape (K, L, D)
    Amplitude of K reference style sequences extracted by the Frequency Domain
    Style Encoder.

Output
------
updated_fft : torch.Tensor (complex) shape (B, L, D)
    Content feature whose amplitude has been modulated by aggregated styles
    while phase is kept intact.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.kernel_attention import KernelizedLinearAttention

__all__ = ["StyleAggregation"]


# Removed AdaIN and any dependency on image style features (F_r).


class StyleAggregation(nn.Module):
    """Fourier-domain multi-character style fusion module."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        # 使用核化线性注意力来实现交叉注意力（Performer 形式）
        # 这里我们只复用其 ϕ(·) 特征映射与随机特征参数，实现 Q≠K 的交叉注意力：
        # out = (ϕ(Q) (ϕ(K)^T V)) / (ϕ(Q) (ϕ(K)^T 1))
        self.cross_attn = KernelizedLinearAttention(
            d_model=d_model,
            m=256,
            trainable_features=True,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        content_amp: torch.Tensor,  # (B, L, D) real (queries)
        style_amp: torch.Tensor,    # (B, K, L, D) or (K, L, D) real (keys/values)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用核化线性交叉注意力融合风格幅度到内容幅度。
        实现：out = (ϕ(Q) (ϕ(K)^T V)) / (ϕ(Q) (ϕ(K)^T 1))。"""

        if style_amp.dim() == 3:
            # Original behaviour: styles shared across batch
            K, L, D = style_amp.shape
            B = content_amp.size(0)
            # 组装 K×L 的风格 token 序列，按批共享
            style_seq = style_amp.reshape(1, K * L, D).repeat(B, 1, 1)  # (B, KL, D)

            # ϕ(Q), ϕ(K)
            phi_q = self.cross_attn._phi(content_amp)  # (B, L, m)
            phi_k = self.cross_attn._phi(style_seq)    # (B, KL, m)

            # S = ϕ(K)^T V，Z = ϕ(K)^T 1
            # (B, KL, m)^T @ (B, KL, D) → (B, m, D)
            S = torch.einsum("bkm,bkd->bmd", phi_k, style_seq)
            # (B, KL, m)^T @ 1 → (B, m)
            Z = phi_k.sum(dim=1)  # (B, m)

            # num = ϕ(Q) S；den = ϕ(Q) Z
            num = torch.einsum("blm,bmd->bld", phi_q, S)         # (B, L, D)
            den = torch.einsum("blm,bm->bl", phi_q, Z).unsqueeze(-1) + 1e-8  # (B, L, 1)
            f_a = num / den  # (B, L, D)
        elif style_amp.dim() == 4:
            B, K, L, D = style_amp.shape
            if B != content_amp.size(0):
                raise ValueError("Batch size mismatch between content_amp and style_amp")
            style_seq = style_amp.reshape(B, K * L, D)   # (B,KL,D)

            phi_q = self.cross_attn._phi(content_amp)  # (B, L, m)
            phi_k = self.cross_attn._phi(style_seq)    # (B, KL, m)

            S = torch.einsum("bkm,bkd->bmd", phi_k, style_seq)  # (B,m,D)
            Z = phi_k.sum(dim=1)                                 # (B,m)

            num = torch.einsum("blm,bmd->bld", phi_q, S)        # (B,L,D)
            den = torch.einsum("blm,bm->bl", phi_q, Z).unsqueeze(-1) + 1e-8
            f_a = num / den                                      # (B,L,D)
        else:
            raise ValueError("style_amp must have shape (K,L,D) or (B,K,L,D)")

        amp_updated = content_amp + f_a  # (B,L,D)
        return amp_updated, f_a


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D, K = 2, 128, 64, 3
    content_amp = torch.rand(B, L, D)
    style_amp = torch.rand(K, L, D)

    agg = StyleAggregation(d_model=D, num_heads=8)
    out, f_a = agg(content_amp, style_amp)
    print("updated shape", out.shape)
    print("f_a shape", f_a.shape)
