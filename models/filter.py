from typing import Optional

import torch
import torch.nn as nn

# Typing compatibility for Python <3.9
from typing import List, Optional, Tuple, Union

class StaticFrequencyFilter(nn.Module):
    """Static (global) frequency-domain filter.

    Given frequency coefficients 𝒁̂ of shape (B, L, D),
    the filter learns a per-frequency gate σ(𝑊_S[k]) ∈ (0, 1)
    that is shared across all channels D and all samples in the batch.

    𝔼[k] = σ(𝑊_S[k]) · 𝒁̂[k]

    Parameters
    ----------
    seq_len: int
        Number of discrete frequency bins L (same as input sequence length).
    init_scale: float, optional (default=0.02)
        Standard deviation multiplier for weight initialisation.
    """

    def __init__(self, seq_len: int, init_scale: float = 0.02):
        super().__init__()
        self.seq_len = seq_len

        # Real-valued learnable weights (L, 1). After sigmoid → (0, 1).
        self.W_s = nn.Parameter(init_scale * torch.randn(seq_len, 1))

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Apply static filter.

        Args
        ----
        z_hat: torch.Tensor, shape (B, L, D)
            Frequency-domain representation before filtering.

        Returns
        -------
        torch.Tensor, same shape as *z_hat*
            Filtered frequency coefficients.
        """
        if z_hat.dim() != 3 or z_hat.size(1) != self.seq_len:
            raise ValueError(
                f"Expected z_hat shape (B, {self.seq_len}, D), got {tuple(z_hat.shape)}")

        # (1, L, 1) gate broadcast across batch and channel dims
        gate = torch.sigmoid(self.W_s).unsqueeze(0)  # (1, L, 1)
        return z_hat * gate


# -----------------------------------------------------------------------------
# Dynamic frequency filter
# -----------------------------------------------------------------------------


class DynamicFrequencyFilter(nn.Module):
    """Dynamic, sample-adaptive frequency filter.

    Generates a gating matrix *A* with local receptive fields on the 2-D
    frequency map (axes: length *L* × channels *D*). For every position (k, d),
    it extracts an *R_L × R_D* patch, vectorises it and feeds it through a
    shared MLP to obtain a scalar gate in (0, 1).

    Notes
    -----
    • Operates on *real-valued* magnitudes of the complex spectrum; the resulting
      gate is then applied (broadcast) to the complex coefficients.
    • Compared with a 2-D convolution, this offers higher-order, non-linear
      interactions with modest parameters.
    """

    def __init__(
        self,
        seq_len: int,
        d_channels: int,
        patch_size: Union[Tuple[int, int], List[Tuple[int, int]]] = (3, 3),
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.d_channels = d_channels

        if hidden_dims is None:
            hidden_dims = [32, 16]
        if activation is None:
            activation = nn.ReLU()

        # Allow a single patch size tuple or a list of tuples (multi-scale)
        if isinstance(patch_size, list):
            self.patch_sizes = patch_size
        else:
            self.patch_sizes = [patch_size]

        # Store original argument for backward compatibility
        self.patch_size = patch_size

        # Build a 2D convolution (single-channel in/out) for each patch size
        # Keep the original signature but ignore hidden_dims/activation.
        self.convs: nn.ModuleList = nn.ModuleList()

        for ps in self.patch_sizes:
            # Use padding=(k//2) which preserves size for odd kernels; for even
            # kernels this yields +1 which we will crop later in forward.
            self.convs.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=ps,
                    padding=(ps[0] // 2, ps[1] // 2),
                    bias=True,
                )
            )

    def forward(self, z_hat: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Apply dynamic filter.

        Parameters
        ----------
        z_hat : torch.Tensor, shape (B, L, D)
            Frequency representation used to compute gates.
        E : torch.Tensor, shape (B, L, D)
            Input to be modulated (typically output of static filter).

        Returns
        -------
        torch.Tensor, shape (B, L, D)
            Dynamically filtered output.
        """

        if z_hat.dim() != 3 or z_hat.size(1) != self.seq_len or z_hat.size(2) != self.d_channels:
            raise ValueError(
                f"Expected z_hat shape (B, {self.seq_len}, {self.d_channels}), got {tuple(z_hat.shape)}")
        if E.shape != z_hat.shape:
            raise ValueError("E and z_hat must have the same shape for element-wise gating.")

        B = z_hat.size(0)

        # Use magnitude for gating computation (real-valued): (B, 1, L, D)
        mag = z_hat.abs().unsqueeze(1)

        all_gates: List[torch.Tensor] = []

        for conv_layer in self.convs:
            # Conv over (B, 1, L, D) → (B, 1, L', D')
            gates_i = torch.sigmoid(conv_layer(mag))

            # Bring to (B, L', D')
            gates_i = gates_i.squeeze(1)

            # Handle even kernel sizes causing one extra row/col.
            # Crop or pad with ones (neutral gate) to match (L, D).
            B_, Lp, Dp = gates_i.size()

            # Adjust length dimension
            if Lp > self.seq_len:
                gates_i = gates_i[:, :self.seq_len, :]
            elif Lp < self.seq_len:
                pad_len = self.seq_len - Lp
                pad_tensor = torch.ones(B_, pad_len, Dp, device=gates_i.device, dtype=gates_i.dtype)
                gates_i = torch.cat([gates_i, pad_tensor], dim=1)

            # Adjust channel dimension
            if Dp > self.d_channels:
                gates_i = gates_i[:, :, :self.d_channels]
            elif Dp < self.d_channels:
                pad_w = self.d_channels - Dp
                pad_tensor = torch.ones(B_, self.seq_len, pad_w, device=gates_i.device, dtype=gates_i.dtype)
                gates_i = torch.cat([gates_i, pad_tensor], dim=2)

            all_gates.append(gates_i)

        # Aggregate gates from different patch sizes (mean)
        gates = torch.stack(all_gates, dim=0).mean(dim=0)

        # Element-wise modulation (broadcast over complex components)
        return E * gates

# -----------------------------------------------------------------------------
# Combined filter: static + dynamic
# -----------------------------------------------------------------------------


class CombinedFrequencyFilter(nn.Module):
    """Convenience wrapper that sequentially applies Static and Dynamic filters.

    Workflow:
        1. E = StaticFrequencyFilter( Ẑ )
        2. D = DynamicFrequencyFilter( Ẑ, E )

    By default it returns the dynamically filtered output *D*; caller can set
    `return_intermediate=True` to also obtain the intermediate static output.
    """

    def __init__(
        self,
        seq_len: int,
        d_channels: int,
        *,
        static_init_scale: float = 0.02,
        patch_size: Tuple[int, int] = (3, 3),
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.static_filter = StaticFrequencyFilter(seq_len, init_scale=static_init_scale)
        self.dynamic_filter = DynamicFrequencyFilter(
            seq_len,
            d_channels,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            activation=activation,
        )

    def forward(self, z_hat: torch.Tensor, *, return_intermediate: bool = False):
        """Apply static then dynamic filtering.

        Parameters
        ----------
        z_hat : torch.Tensor, shape (B, L, D)
            Input frequency coefficients.
        return_intermediate : bool, optional
            If True, return both (D, E). Otherwise only D.
        """
        E = self.static_filter(z_hat)          # static-filtered output
        D = self.dynamic_filter(z_hat, E)      # dynamic-filtered output

        if return_intermediate:
            return D, E
        return D

