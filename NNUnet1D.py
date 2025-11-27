# Fixed NNUnet1D.py
# - Ensure log_norm scalar from premodule has correct shape before feeding into scale_embed.
# - Prevent accidental extra dims that caused expand(...) runtime error on MPS.
# - Kept comments and simple logic so it's easy to follow.

import torch
import torch.nn as nn
import torch.nn.functional as F
from NN import NormalizeLogRadius, evaluate
from typing import Optional, Tuple


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        premodule: Optional[str] = None,
        emb_dim=128,     # <--- time embedding size
    ):
        """
        Simple, readable 1D UNet-like architecture for time-series (length ~100-1000).

        Key points:
        - premodule: None or "NormalizeLogRadius". If NormalizeLogRadius is used,
          the premodule returns (x_normalized, log_norm) where log_norm is a scalar
          per example (shape (B,1) or (B,1,1)). We map log_norm -> emb_dim and add
          it to the time embedding so the model conditions on both time and global scale.
        """
        super().__init__()

        self.input_dim = input_dim
        assert premodule in (None, "NormalizeLogRadius")
        self.premodule = NormalizeLogRadius() if premodule == "NormalizeLogRadius" else None

        # Time embedding: map scalar t -> emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # If a premodule is used, create a small MLP that maps the scalar log-norm
        # (shape (B,1)) to a vector of size emb_dim. We'll add this vector to the
        # time embedding so conditioning is (time + lognorm).
        if self.premodule is not None:
            self.scale_embed = nn.Sequential(
                nn.Linear(1, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.scale_embed = None

        # Channel configuration for encoder/decoder
        chs = [base_channels * m for m in channel_mults]

        # ---------------- Encoder ----------------
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        in_ch = 1  # our input has 1 channel (we will concat time-embedding channels later)
        for out_ch in chs:
            # Each block expects input channels = (prev_out_ch + emb_dim) because we concat time embedding
            self.enc_blocks.append(ConvBlock1D(in_ch + emb_dim, out_ch))
            # Downsample by factor 2
            self.downs.append(
                nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            in_ch = out_ch

        # ---------------- Bottleneck ----------------
        # Middle block also receives the time embedding concatenated along channels
        self.middle = ConvBlock1D(in_ch + emb_dim, in_ch)

        # ---------------- Decoder ----------------
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for out_ch in reversed(chs):
            self.up_convs.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            # Decoder block receives: upsampled features + skip connection + time embedding
            self.dec_blocks.append(
                ConvBlock1D(out_ch * 2 + emb_dim, out_ch)
            )
            in_ch = out_ch

        # ---------------- Final projection ----------------
        self.final = nn.Conv1d(in_ch, 1, kernel_size=1)


    def forward(self, x, t):
        """
        x: (B, L) or (B, 1, L)
        t: (B,) or (B,1)  - scalar time / timestep conditioning

        Returns:
            out: (B, L) same spatial length as input (up to minor padding at decoder).
        """
        # Normalize input shape -> (B,1,L)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,L)

        # Make sure t is (B,1)
        t = t.view(-1, 1)        # (B,1)

        # Compute time embedding vector (B, emb_dim)
        t_emb = self.time_mlp(t) # (B, emb_dim)

        # If premodule provided, apply it to x to get normalized x and log_norm
        # NormalizeLogRadius returns (x_normalized, log_norm) where log_norm may be (B,1,1)
        if self.premodule is not None:
            x, log_norm = self.premodule(x)  # x: (B,1,L), log_norm: (B,1) or (B,1,1)

            # Scale normalized x to keep std consistent across lengths (same pattern as NNUnet)
            x = x * torch.sqrt(torch.tensor(x.shape[-1], dtype=log_norm.dtype, device=log_norm.device))

            # --- CRITICAL FIX ---
            # Ensure log_norm has shape (B, 1) before feeding to the linear MLP.
            # If log_norm comes as (B,1,1) we must squeeze the last dimension; if it is (B,1) keep it.
            log_norm_flat = log_norm.view(log_norm.shape[0], -1)  # becomes (B,1)
            # Now scale_embed maps (B,1) -> (B, emb_dim)
            scale_vec = self.scale_embed(log_norm_flat.to(t_emb.dtype))  # (B, emb_dim)
            # add scale vector to time embedding (both are (B, emb_dim))
            t_emb = t_emb + scale_vec

        # Helper to expand t_emb along spatial dimension for concatenation
        # t_emb currently is (B, emb_dim)
        t_emb = t_emb.unsqueeze(-1)  # (B, emb_dim, 1)
        t_emb_rep = lambda h: t_emb.expand(-1, -1, h.shape[-1])

        skips = []
        h = x  # (B,1,L)

        # -------- Encoder --------
        for block, down in zip(self.enc_blocks, self.downs):
            # Concatenate time embedding as additional channels before each conv block
            h = torch.cat([h, t_emb_rep(h)], dim=1)  # (B, C + emb_dim, L)
            h = block(h)
            skips.append(h)
            h = down(h)

        # -------- Middle --------
        h = torch.cat([h, t_emb_rep(h)], dim=1)
        h = self.middle(h)

        # -------- Decoder --------
        for up, block in zip(self.up_convs, self.dec_blocks):
            h = up(h)
            skip = skips.pop()

            # If upsampling produces a slightly smaller length due to padding, pad it.
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))

            # Concatenate skip + time embedding and run conv block
            h = torch.cat([h, skip, t_emb_rep(h)], dim=1)
            h = block(h)

        out = self.final(h).squeeze(1)  # (B, L)
        return out