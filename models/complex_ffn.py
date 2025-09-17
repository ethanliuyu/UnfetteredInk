"""Complex Feedforward Network (FFN) with modReLU activation.

Implements the two-layer feed-forward module described in the paper:
    H = σ(W1 · Z̃ + B1)
    FFN(Z̃) = W2 · H + B2
where σ is the modReLU non-linearity operating in the complex domain.

All computations are performed using complex tensors (dtype torch.cfloat /
cdouble).  Real tensors are automatically promoted to complex by adding an
imaginary part of zero.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["ComplexFeedForwardNetwork", "ModReLU"]


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------

class ComplexLinear(nn.Module):
    """Linear layer for complex inputs and weights.

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool, optional (default=True)
    init_scale : float, optional (default=0.02)
        Standard deviation of the normal distribution used for weight init.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, init_scale: float = 0.02) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = init_scale * torch.randn(out_features, in_features, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

        if bias:
            bias_param = init_scale * torch.randn(out_features, dtype=torch.cfloat)
            self.bias = nn.Parameter(bias_param)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (…, in) -> (…, out)
        if not torch.is_complex(x):
            x = x.to(dtype=torch.cfloat)
        y = torch.einsum("...i,oi->...o", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class ModReLU(nn.Module):
    """modReLU activation for complex inputs.

    For each complex element z and learnable bias b ::
        σ(z) = ReLU(|z| + b) · z / (|z| + ε)
    ε is a small constant to avoid divide-by-zero.
    """

    def __init__(self, d_model: int, *, eps: float = 1e-8, bias_init: float = 0.0) -> None:
        super().__init__()
        self.eps = eps
        # Real-valued bias per feature dimension
        bias = torch.full((d_model,), bias_init, dtype=torch.float32)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = x.to(dtype=torch.cfloat)
        magnitude = torch.abs(x)  # (…)
        scale = torch.relu(magnitude + self.bias) / (magnitude + self.eps)
        return x * scale


# ---------------------------------------------------------------------------
# Complex Feedforward Network
# ---------------------------------------------------------------------------

class ComplexFeedForwardNetwork(nn.Module):
    """Two-layer FFN operating in the complex domain with modReLU activation."""

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        *,
        activation: Optional[nn.Module] = None,
        init_scale: float = 0.02,
    ) -> None:
        """Parameters
        ----------
        d_model : int
            Input / output dimensionality.
        d_hidden : int, optional
            Hidden dimensionality.  Defaults to 4·d_model as in Transformer FFN.
        activation : nn.Module, optional
            Custom activation module.  If *None*, uses ModReLU.
        init_scale : float, optional
            Weight initialisation scale for *ComplexLinear* layers.
        """
        super().__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model
        if activation is None:
            activation = ModReLU(d_hidden)

        self.fc1 = ComplexLinear(d_model, d_hidden, init_scale=init_scale)
        self.act = activation
        self.fc2 = ComplexLinear(d_hidden, d_model, init_scale=init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        if not torch.is_complex(z):
            z = z.to(dtype=torch.cfloat)
        h = self.act(self.fc1(z))
        return self.fc2(h)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, L, d = 2, 128, 64
    x = torch.randn(B, L, d) + 1j * torch.randn(B, L, d)

    ffn = ComplexFeedForwardNetwork(d_model=d, d_hidden=256)
    out = ffn(x)
    print("input shape:", x.shape)
    print("output shape:", out.shape)
    # Parameter count
    print("可学习参数总数:", sum(p.numel() for p in ffn.parameters()))


