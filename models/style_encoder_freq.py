"""Frequency Domain Style Encoder

This module implements the Fourier-domain style encoder described in the
paper.  It maps *K* reference coordinate sequences (each resampled to *L*
points) into their complex frequency spectrum, projects them into a
higher-dimensional latent space via an isometric embedding, and refines the
representation using *N* stacked complex self-attention blocks
(Kernelised Linear Attention) followed by complex feed-forward networks.

The encoder outputs the amplitude part of the refined spectrum, which is
later fused with content features by the StyleAggregation module.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from models.kernel_attention import KernelizedLinearAttention
from models.complex_ffn import ComplexFeedForwardNetwork, ModReLU

__all__ = ["FrequencyStyleEncoder"]


# ---------------------------------------------------------------------------
# Encoder building block -----------------------------------------------------
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """One layer of complex self-attention + FFN (Add & Norm)."""

    def __init__(self, d_channels: int, *, m: int = 256) -> None:
        super().__init__()
        # Attention over real+imag parts → 2*d_channels dims
        self.attn = KernelizedLinearAttention(d_model=2 * d_channels, m=m, trainable_features=True)
        self.attn_norm = ModReLU(d_channels)
        self.ffn = ComplexFeedForwardNetwork(d_channels)
        self.ffn_norm = ModReLU(d_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (K, L, D) complex
        y = self.attn(z)                       # Self-attention in complex domain
        z = self.attn_norm(z + y)              # Add & Norm (modReLU)
        y2 = self.ffn(z)                       # Complex FFN
        z = self.ffn_norm(z + y2)              # Add & Norm (modReLU)
        return z


# ---------------------------------------------------------------------------
# Frequency Domain Style Encoder --------------------------------------------
# ---------------------------------------------------------------------------

class FrequencyStyleEncoder(nn.Module):
    """Fourier-domain style encoder.

    Parameters
    ----------
    seq_len : int
        Number of points *L* per coordinate sequence (after resampling).
    d_channels : int, optional
        Dimensionality *D* of the latent spectrum (default: 128).
    num_layers : int, optional
        Number of `_EncoderBlock` stacked (default: 3).
    m : int, optional
        Number of random Fourier features in attention (default: 256).
    scale : float, optional
        Std of the random init for the projection matrix (default: 0.02).
    """

    def __init__(
        self,
        seq_len: int,
        d_channels: int = 128,
        *,
        num_layers: int = 3,
        m: int = 256,
        scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_channels = d_channels

        # ------------------------------------------------------------------
        # 1. Isometric embedding (learnable orthonormal projection) ---------
        # ------------------------------------------------------------------
        proj = scale * torch.randn(d_channels, 2, dtype=torch.cfloat)
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        if d_channels >= 1:
            proj[0, 0] = sqrt2_inv + 0j
            proj[0, 1] = sqrt2_inv + 0j
        if d_channels >= 2:
            proj[1, 0] = sqrt2_inv + 0j
            proj[1, 1] = -sqrt2_inv + 0j
        # Raw projection → orthonormalised every forward pass via QR
        self.raw_proj = nn.Parameter(proj)

        # ------------------------------------------------------------------
        # 2. Stacked attention + FFN blocks --------------------------------
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList([
            _EncoderBlock(d_channels, m=m) for _ in range(num_layers)
        ])

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode reference coordinate sequences.

        Parameters
        ----------
        seq : torch.Tensor (real) shape (K, L, 2)
            *K* reference trajectories, each resampled to *L* points.

        Returns
        -------
        torch.Tensor (real) shape (K, L, D)
            Amplitude \( \mathcal{A}(F_s) \) of the refined spectrum.
        """
        if seq.dim() != 3 or seq.size(-1) != 2:
            raise ValueError("Input must have shape (K, L, 2).")
        K, L, _ = seq.shape
        if L != self.seq_len:
            raise ValueError(f"Sequence length mismatch: expected {self.seq_len}, got {L}")

        # ------------------------------------------------------------------
        # 1. Discrete Fourier Transform (per coordinate dimension) ---------
        # ------------------------------------------------------------------
        seq_fft = torch.fft.fft(seq, dim=1)  # (K, L, 2) complex

        # ------------------------------------------------------------------
        # 2. Orthonormal projection into C^D --------------------------------
        # ------------------------------------------------------------------
        A, _ = torch.linalg.qr(self.raw_proj, mode="reduced")  # (D, 2)
        z = torch.einsum("klc,dc->kld", seq_fft, A)            # (K, L, D) complex

        # ------------------------------------------------------------------
        # 3. Self-attention stack ------------------------------------------
        # ------------------------------------------------------------------
        for layer in self.layers:
            z = layer(z)

        # ------------------------------------------------------------------
        # 4. Output amplitude ----------------------------------------------
        # ------------------------------------------------------------------
        amp = z.abs()                                          # (K, L, D) real
        return amp
