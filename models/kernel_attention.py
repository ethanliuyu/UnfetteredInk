"""Kernelized Linear Self-Attention (Random Fourier Features).

This module implements the linearised self-attention mechanism described in the
user-provided derivation.  It approximates a translation-invariant kernel (e.g.
Gaussian/RBF) via Random Fourier Features (RFF) and reduces the quadratic
complexity of conventional softmax attention to *O(L·m)*, where **m** is the
number of random features.

Notation in code follows the derivation:
    •   Q, K, V ∈ ℝ^{L×d} – here we use the *same* tensor as queries / keys /
        values (i.e. self-attention) so the input has shape (B, L, d).
    •   ϕ(·) ∈ ℝ^{m} is the random features mapping.

Forward complexity:
    1. ϕ(K) is computed once  →  Z ∈ ℝ^{m}  and  S ∈ ℝ^{m×d}
    2. For every token i we compute  ϕ(Q_i)^⊤S  and  ϕ(Q_i)^⊤Z

Both steps require only inner products in **m** dimensions.

Typical usage::

    attn = KernelizedLinearAttention(d_model=128, m=256)
    out  = attn(x)    # x: (B, L, 128)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["KernelizedLinearAttention"]


class KernelizedLinearAttention(nn.Module):
    """Linearised self-attention via Random Fourier Features.

    Parameters
    ----------
    d_model : int
        Dimensionality *d* of the input features.
    m : int, optional (default=64)
        Number of random Fourier features ϕ(x) ∈ ℝ^m.  Larger *m* gives a
        better approximation at the cost of increased compute/memory.
    sigma : float or None, optional (default=None)
        Bandwidth σ of the RBF kernel.  If *None*, uses 1/√d as a heuristic.
    trainable_features : bool, optional (default=False)
        If *True*, ω and b are registered as learnable parameters instead of
        buffers (cf. Performer / Nyströmformer).
    """

    def __init__(
        self,
        d_model: int,
        m: int = 64,
        sigma: Optional[float] = None,
        *,
        trainable_features: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.m = m
        # Heuristic bandwidth if not specified
        self.sigma = 1.0 / math.sqrt(d_model) if sigma is None else sigma

        # ω ~ N(0, σ^{-2} I)
        omega = torch.randn(m, d_model) / self.sigma
        # b ~ Uniform(0, 2π)
        bias = 2 * math.pi * torch.rand(m)

        if trainable_features:
            self.omega = nn.Parameter(omega)  # type: ignore[assignment]
            self.bias = nn.Parameter(bias)    # type: ignore[assignment]
        else:
            self.register_buffer("omega", omega)
            self.register_buffer("bias", bias)

        # Normalisation constant √(2/m)
        self.scale = math.sqrt(2.0 / m)

    # ---------------------------------------------------------------------
    # Random Fourier Features mapping ϕ(·)
    # ---------------------------------------------------------------------

    def _phi(self, x: torch.Tensor) -> torch.Tensor:  # (…, d) → (…, m)
        # Project: xΩ^⊤ → (…, m)
        proj = torch.matmul(x, self.omega.t())  # (…, m)
        proj = proj + self.bias  # Broadcast bias
        return self.scale * torch.cos(proj)

    # ------------------------------------------------------------------
    # Forward (self-attention)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d) → (B, L, d)
        """Apply kernelised self-attention to *x*.

        The implementation works with real-valued inputs.  If *x* is complex
        (dtype = torch.cfloat / torch.cdouble), the real and imaginary parts are
        concatenated along the feature dimension so that attention operates on
        ℝ^{2d}.  The output is converted back to the original complex dtype.
        """

        is_complex = torch.is_complex(x)
        if is_complex:
            # Stack real & imag parts → real tensor of shape (B, L, 2d)
            x_real = torch.cat([x.real, x.imag], dim=-1)
        else:
            x_real = x  # type: ignore[assignment]

        B, L, d_in = x_real.shape
        if d_in != self.d_model:
            raise ValueError(
                f"Input dim {d_in} does not match expected {self.d_model}"
            )

        # ------------------------------------------------------------------
        # 1. Compute feature map for K (= Q = V = x)
        # ------------------------------------------------------------------
        phi_x = self._phi(x_real)              # (B, L, m)

        # ------------------------------------------------------------------
        # 2. Aggregations over keys (batch level)
        # ------------------------------------------------------------------
        Z = phi_x.sum(dim=1)                   # (B, m)
        # S: Σ_j ϕ(K_j) V_j^⊤ → (B, m, d_in)
        S = torch.einsum("blm,bld->bmd", phi_x, x_real)

        # ------------------------------------------------------------------
        # 3. Per-query computations (linear in L)
        # ------------------------------------------------------------------
        num = torch.einsum("blm,bmd->bld", phi_x, S)  # (B, L, d_in)
        den = torch.einsum("blm,bm->bl", phi_x, Z).unsqueeze(-1) + 1e-8  # (B, L, 1)
        out_real = num / den                      # (B, L, d_in)

        if is_complex:
            # Split back to real & imag components
            d_half = self.d_model // 2  # original feature dimension
            real_part, imag_part = out_real[..., :d_half], out_real[..., d_half:]
            return torch.complex(real_part, imag_part)
        return out_real


if __name__ == "__main__":
    """Quick sanity check for the attention layer.

    Creates random real & complex tensors and verifies that the output shapes
    match the input shapes.  Run with:

        python -m Code.models.kernel_attention
    """
    torch.manual_seed(0)

    # Real-valued test ------------------------------------------------------
    B, L, d = 2, 128, 64
    x = torch.randn(B, L, d)
    attn = KernelizedLinearAttention(d_model=d, m=256)
    out = attn(x)
    print("[REAL]  input  shape:", x.shape)
    print("[REAL]  output shape:", out.shape)

    # Complex-valued test ---------------------------------------------------
    x_c = torch.randn(B, L, d, dtype=torch.float32) + 1j * torch.randn(B, L, d)
    attn_c = KernelizedLinearAttention(d_model=2 * d, m=256,trainable_features=True)  # 2*d for real+imag
    out_c = attn_c(x_c)
    print("[CMPLX] input  shape:", x_c.shape)
    print("[CMPLX] output shape:", out_c.shape)

    total_params = sum(p.numel() for p in attn_c.parameters())
    print("可学习参数总数:", total_params)  
