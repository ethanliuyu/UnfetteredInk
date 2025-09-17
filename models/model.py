
import torch
import torch.nn as nn
import math
import os

from models.filter import CombinedFrequencyFilter
from models.kernel_attention import KernelizedLinearAttention
from models.style_aggregation import StyleAggregation
from models.style_encoder_freq import FrequencyStyleEncoder
from dataset import svg_path_to_image
import torchvision.transforms as T
import numpy as np
from models.complex_ffn import ModReLU, ComplexFeedForwardNetwork





class TrajectoryFrequencyModel(nn.Module):

    class FusionBlock(nn.Module):
        """Encapsulates Kernelised Linear Attention + Style Aggregation."""
        def __init__(self, seq_len: int, d_channels: int, *, m: int = 256, num_heads: int = 8):
            super().__init__()
            # Attention works on complex tensor directly
            self.attn = KernelizedLinearAttention(d_model=2 * d_channels, m=m, trainable_features=True)
            # Static + dynamic frequency filter per block
            self.freq_filter = CombinedFrequencyFilter(seq_len, d_channels, patch_size=(7, 14))
            # Normalisation (ModReLU) after attention residual
            self.attn_norm = ModReLU(d_channels)
            # Feed-forward network operating in complex domain
            self.ffn = ComplexFeedForwardNetwork(d_channels)
            # Normalisation (ModReLU) after FFN residual
            self.ffn_norm = ModReLU(d_channels)
            # Style aggregation operates on amplitude (image-style removed)
            self.style_agg = StyleAggregation(d_model=d_channels, num_heads=num_heads)

        def forward(self, x_fft: torch.Tensor, style_amp: torch.Tensor) -> torch.Tensor:
            """x_fft : (B, L, D) complex
            style_amp : (K, L, D) real or (B, K, L, D) real
            Returns updated complex tensor.
            """
            # 1. Frequency filtering
            filt_fft = self.freq_filter(x_fft)
            # 2. Self-attention in complex domain
            attn_fft = self.attn(filt_fft)  # (B, L, D) complex
            # 3. Residual connection & ModReLU norm
            res_fft = self.attn_norm(filt_fft + attn_fft)

            # 4. Amplitude-phase decomposition
            amp = res_fft.abs()
            phase_unit = res_fft / (amp + 1e-8)

            # 5. Style aggregation on amplitude (image-style features removed)
            amp_updated, _ = self.style_agg(amp, style_amp)

            # 6. Recompose complex tensor
            comp_fft = amp_updated * phase_unit  # (B, L, D) complex

            # 7. Complex Feed-Forward Network
            ffn_out = self.ffn(comp_fft)

            # 8. Residual add & ModReLU norm
            out_fft = self.ffn_norm(comp_fft + ffn_out)
            return out_fft


    def __init__(self, seq_len: int, d_channels: int = 128, scale: float = 0.02):
        super().__init__()
        self.seq_len = seq_len
        self.d_channels = d_channels

        # Build style-attention fusion block(s) – each contains its own freq filter
        self.fusion_blocks = nn.ModuleList([
            TrajectoryFrequencyModel.FusionBlock(seq_len, d_channels)
        ])

        # Construct initial projection matrix following the paper:
        #   A = 1/sqrt(2) * [[1,  1],
        #                    [1, -1],
        #                    [... random ...]]
        proj = scale * torch.randn(d_channels, 2, dtype=torch.cfloat)

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        if d_channels >= 1:
            proj[0, 0] = sqrt2_inv + 0j
            proj[0, 1] = sqrt2_inv + 0j
        if d_channels >= 2:
            proj[1, 0] = sqrt2_inv + 0j
            proj[1, 1] = -sqrt2_inv + 0j


        # Raw complex projection matrix (learnable). It will be orthogonalized
        # to obtain an orthonormal basis A ∈ ℂ^{D×2} at every forward pass.
        self.raw_proj = nn.Parameter(proj)

        # Frequency-domain style encoder (learned from reference trajectories)
        self.style_encoder = FrequencyStyleEncoder(seq_len, d_channels)

        # Image-based Style Encoder removed; no image feature extraction.

    # ----------------------------------------------------------------------
    # Loss computation -----------------------------------------------------
    # ----------------------------------------------------------------------
    def compute_loss(
        self,
        traj_pred: torch.Tensor,  # (B, L, 2)
        traj_target: torch.Tensor,  # (B, L, 2)
        style_seq_ref: torch.Tensor,  # (K, L, 2)
        *,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        lambda4: float = 1.0,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute total loss as described in the paper.

        Parameters
        ----------
        traj_pred : torch.Tensor (B, L, 2)
            Predicted trajectory coordinates \(\hat{S}\).
        traj_target : torch.Tensor (B, L, 2)
            Ground-truth trajectory coordinates \(Y\).
        style_seq_ref : torch.Tensor (K, L, 2)
            Reference style trajectories \(S_r\).
        lambda1, lambda2, lambda3, lambda4 : float, optional
            Weights for each loss term.

        Returns
        -------
        torch.Tensor
            Scalar total loss \(\mathcal{L}\).
        """
        # ------------------------------------------------------------------
        # 1. Coordinate-space MSE ------------------------------------------
        # ------------------------------------------------------------------
        L_mse = torch.mean((traj_target - traj_pred) ** 2)

        # ------------------------------------------------------------------
        # 2. Image-domain style loss removed
        L_img = torch.tensor(0.0, device=traj_pred.device)

        # ------------------------------------------------------------------
        # 3. Frequency-domain style losses ---------------------------------
        # ------------------------------------------------------------------
        # Amplitude spectra of reference, generated, and target trajectories
        amp_r = self.style_encoder(style_seq_ref)              # (K, L, D)
        amp_g = self.style_encoder(traj_pred.detach())         # (B, L, D)
        amp_t = self.style_encoder(traj_target.detach())       # (B, L, D)

        # ------------------------------------------------------
        # Amplitude energy normalisation (scale invariance) ----
        # ------------------------------------------------------
        def _norm_amp(a: torch.Tensor):
            # a: (N,L,D) real → divide by per-sample RMS to suppress scale mismatch
            rms = torch.sqrt((a ** 2).mean(dim=(1, 2), keepdim=True) + 1e-8)
            return a / rms

        amp_r_n = _norm_amp(amp_r)            # (K,L,D)
        amp_g_n = _norm_amp(amp_g)            # (B,L,D)
        amp_t_n = _norm_amp(amp_t)            # (B,L,D)

        # Compute losses (average over all dims)
        L_rg = torch.mean((amp_r_n.mean(dim=0) - amp_g_n.mean(dim=0)) ** 2)
        L_rt = torch.mean((amp_r_n.mean(dim=0) - amp_t_n.mean(dim=0)) ** 2)

        # ------------------------------------------------------------------
        # 4. Total ----------------------------------------------------------
        # ------------------------------------------------------------------
        total_loss = (
            lambda1 * L_mse +
            lambda2 * L_img +
            lambda3 * L_rg +
            lambda4 * L_rt
        )
        if return_components:
            comps = {
                'mse': L_mse.detach(),
                'img': L_img.detach(),
                'rg': L_rg.detach(),
                'rt': L_rt.detach()
            }
            return total_loss, comps
        return total_loss

    # ----------------------------------------------------------------------
    # Helper: end-to-end loss preparation from model outputs ---------------
    # ----------------------------------------------------------------------
    def prepare_and_compute_loss(
        self,
        recon_fft: torch.Tensor,          # (B, L, 2) complex – output of forward()
        traj_target: torch.Tensor,        # (B, L, 2) real
        style_seq_ref: torch.Tensor,      # (B, K, L, 2) real
        *,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        lambda4: float = 1.0,
    ) -> torch.Tensor:
        """Prepare all inputs and delegate to *compute_loss*.

        This method performs:
        1. Inverse FFT of *recon_fft* to obtain predicted trajectories.
        2. Rasterisation of trajectories to 256×256 images via *svg_path_to_image*.
        3. Per-sample loss aggregation with *compute_loss*.
        """
        device = recon_fft.device
        B, L, _ = recon_fft.shape

        # Prepare image normalisation transform (match dataset)
        img_transform = T.Compose([
            T.ToTensor(),                 # (H,W) → (1,H,W) float [0,1]
            T.Normalize((0.5,), (0.5,))   # match training normalisation
        ])

        losses = []
        mse_list, img_list, rg_list, rt_list = [], [], [], []
        for i in range(B):
            # ------------------------------------------------------------------
            # 1. Trajectory prediction (time domain) --------------------------
            # ------------------------------------------------------------------
            traj_pred_complex = torch.fft.ifft(recon_fft[i], dim=0)  # (L, 2) complex
            traj_pred = traj_pred_complex.real                      # take real part

            # Clamp to drawing canvas (1–256)
            coord_np = traj_pred.detach().cpu().numpy()
            coord_np = np.clip(np.round(coord_np), 1, 256).astype(int)

            # Build SVG path string: M x0 y0 L x1 y1 ...
            path_parts = [f"M {coord_np[0,0]} {coord_np[0,1]}"] + [
                f"L {x} {y}" for x, y in coord_np[1:]
            ]
            path_str = " ".join(path_parts)

            # Rasterise to grayscale image (numpy H×W)
            img_np = svg_path_to_image(path_str, image_size=(256, 256))
            img_tensor = img_transform(img_np)  # (1,256,256)

            # Image feature extraction removed

            # ------------------------------------------------------------------
            # 3. Compute loss for this sample ---------------------------------
            # ------------------------------------------------------------------
            loss_i, comps_i = self.compute_loss(
                traj_pred=traj_pred.unsqueeze(0).to(device),
                traj_target=traj_target[i].unsqueeze(0).to(device),
                style_seq_ref=style_seq_ref[i].to(device),
                F_r_img=torch.empty(0, device=device),
                F_g_img=torch.empty(0, device=device),
                lambda1=lambda1,
                lambda2=lambda2,
                lambda3=lambda3,
                lambda4=lambda4,
                return_components=True,
            )
            losses.append(loss_i)
            mse_list.append(comps_i['mse'])
            img_list.append(comps_i['img'])
            rg_list.append(comps_i['rg'])
            rt_list.append(comps_i['rt'])

        total = torch.mean(torch.stack(losses))
        comps_mean = {
            'mse': torch.mean(torch.stack(mse_list)),
            'img': torch.mean(torch.stack(img_list)),
            'rg': torch.mean(torch.stack(rg_list)),
            'rt': torch.mean(torch.stack(rt_list)),
        }
        return total, comps_mean

    # ----------------------------------------------------------------------
    # Forward pass ---------------------------------------------------------
    # ----------------------------------------------------------------------
    def forward(
        self,
        traj: torch.Tensor,          # (B, L, 2)
        style_seq: torch.Tensor,     # (B,K,L,2) or (K,L,2)
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            traj: Tensor of shape (B, L, 2) – 2-D trajectory in Cartesian space.

        Returns:
            Tensor of shape (B, L, D) – frequency-domain representation.
        """
        if traj.dim() != 3 or traj.size(-1) != 2:
            raise ValueError("Input must have shape (B, L, 2).")
        B, L, _ = traj.shape
        if L != self.seq_len:
            raise ValueError(f"Sequence length mismatch: expected {self.seq_len}, got {L}")

        # (B, L, 2) → (B, L, 2) complex
        traj_fft = torch.fft.fft(traj, dim=1)

        # Obtain orthonormal projection matrix A via QR: AᴴA = I₂.
        A, _ = torch.linalg.qr(self.raw_proj, mode="reduced")  # (D, 2)

        # Linear projection (B, L, 2) × (2, D) → (B, L, D) in complex domain
        fft = torch.einsum('blc,dc->bld', traj_fft, A)


        # Image style features removed

        # --------------------------------------------------------------
        # Style encoding (batched or shared) ---------------------------
        # --------------------------------------------------------------
        if style_seq.dim() == 4:
            # (B,K,L,2) – individual style banks per sample
            B_s, K, L_s, _ = style_seq.shape
            if B_s != B or L_s != self.seq_len:
                raise ValueError("style_seq shape mismatch with traj")
            style_seq_flat = style_seq.reshape(B_s * K, L_s, 2)        # (B*K,L,2)
            style_amp_flat = self.style_encoder(style_seq_flat)        # (B*K,L,D)
            style_amp = style_amp_flat.reshape(B_s, K, L_s, self.d_channels)  # (B,K,L,D)
        elif style_seq.dim() == 3:
            # (K,L,2) – shared across the batch
            style_amp = self.style_encoder(style_seq)                  # (K,L,D)
        else:
            raise ValueError("style_seq must have shape (B,K,L,2) or (K,L,2)")

        # --------------------------------------------------------------
        # Fusion blocks ------------------------------------------------
        # --------------------------------------------------------------
        for block in self.fusion_blocks:
            fft = block(fft, style_amp)


        # ------------------------------------------------------------------
        # Optional: map back to original 2-channel frequency spectrum to verify
        # energy preservation / exact invertibility.
        # Ẑ  (B, L, D) × (D, 2) → (B, L, 2)
        recon_fft = torch.einsum('bld,dc->blc', fft, A.conj())


        # Return attention + style fused representation
        return recon_fft









