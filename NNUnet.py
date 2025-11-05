

import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- colleague's helpers / UNetModel ----------
# If these live elsewhere, adjust imports accordingly.
from model.nn_utils import conv_nd, linear, SiLU, timestep_embedding  # needed here
from model.unet import UNetModel  # UNet class
from NN import NormalizeLogRadius, evaluate

from own_plotting import plots_vort
import matplotlib.pyplot as plt

scale_image = 5
plot_debug= False

# ============================================================
# 0) Flat <-> Image helpers (support C or F ordering)
# ============================================================

def flat_to_img(x: torch.Tensor, H: int, W: int, order: Literal["C","F"]="C") -> torch.Tensor:
    """
    x: (B, d) with d = H*W  ->  (B, 1, H, W)
    """
    B, d = x.shape
    x = x/scale_image  # rescale to (0,1) for NN
    assert d == H*W, f"Expected d={H*W}, got {d}"
    if order == "C":
        x = x.view(B, 1, H, W)
    else:  # "F"
        x = x.view(B, 1, W, H).transpose(2, 3).contiguous()
    
    if plot_debug:
        xcopy = x.clone()
        with torch.no_grad():
            for i in range(min(B, 5)):
                xt = xcopy[i,0,:,:].cpu().squeeze().numpy()
                plots_vort(xt,vmin=-1,vmax=1)
                plt.show(block=False)
                name_fig = "images/NNimage_In_" + str(i) + ".png"
                plt.savefig(name_fig)
                plt.pause(1)
            plt.close()
            plt.close('all')

    return x

def img_to_flat(y: torch.Tensor, order: Literal["C","F"]="C") -> torch.Tensor:
    """
    y: (B, 1, H, W)  ->  (B, H*W)
    """
    B, C, H, W = y.shape

    if plot_debug:
        xcopy = y.clone()
        with torch.no_grad():
            for i in range(min(B, 5)):
                xt = xcopy[i,0,:,:].cpu().squeeze().numpy()
                plots_vort(xt,vmin=-1,vmax=1)
                plt.show(block=False)
                name_fig = "images/NNimage_Out_" + str(i) + ".png"
                plt.savefig(name_fig)
                plt.pause(1)
            plt.close()
            plt.close('all')

    y = scale_image*y  # scale from (0,1) for NN
    assert C == 1, f"Expected 1 channel, got {C}"
    if order == "C":
        return y.reshape(B, H*W)
    else:  # "F"
        return y.transpose(2, 3).contiguous().view(B, H*W)


class UNetModelWithLogNorm(UNetModel):
    """
    Extends UNetModel by adding optional log||x|| scalar conditioning.
    If enabled, we mirror the time-embedding MLP for log||x|| and add it to `emb`.
    """
    def __init__(self, *args, use_log_norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_log_norm = use_log_norm
        if self.use_log_norm:
            time_embed_dim = self.model_channels * 4
            self.scale_embed = nn.Sequential(
                linear(self.model_channels, time_embed_dim),
                SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

    def _make_emb(self, timesteps: torch.Tensor, log_norm: Optional[torch.Tensor] = None, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y is not None and y.shape == (emb.shape[0],)
            emb = emb + self.label_emb(y)
        if self.use_log_norm:
            assert log_norm is not None, "log_norm must be provided when use_log_norm=True"
            logn = log_norm.view(-1)  # (B,)
            emb_scale = self.scale_embed(timestep_embedding(logn, self.model_channels))
            emb = emb + emb_scale
        return emb

    def forward_up_to_middle(self, x, timesteps, y=None, log_norm: Optional[torch.Tensor] = None):
        assert (y is not None) == (self.num_classes is not None), "y required iff class-conditional"
        hs = []
        emb = self._make_emb(timesteps, log_norm=log_norm, y=y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        return h, hs, emb

    def forward(self, x, timesteps, y=None, log_norm: Optional[torch.Tensor] = None):
        if self.learn_potential:
            x.requires_grad = True
        with torch.set_grad_enabled(torch.is_grad_enabled() or self.learn_potential):
            h, hs, emb = self.forward_up_to_middle(x, timesteps, y=y, log_norm=log_norm)
            if self.learn_potential:
                from torch.autograd import backward
                potential = h.mean((-1, -2, -3)).sum(0)
                backward(potential, create_graph=True)
                grad = x.grad
                x.grad = None
                x.requires_grad = False
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                return grad
            else:
                for module in self.output_blocks:
                    cat_in = torch.cat([h, hs.pop()], dim=1)
                    h = module(cat_in, emb)
                h = h.type(x.dtype)
                return self.out(h)


class VorticityUNet(nn.Module):
    """
    Wrapper keeping your (x, t) API and premodule flag.
    - premodule:
        * None                -> raw x, time-only conditioning
        * "NormalizeLogRadius"-> x/||x||, time + log||x|| conditioning
    - Accepts (B, d=H*W) or (B,1,H,W). Returns the same shape it received.
    """
    def __init__(
        self,
        base_channels: int = 32,                 # maps to model_channels
        channel_mults = (1, 2, 4),               # width per scale (e.g., 16->8->4 for 16x16)
        num_res_blocks: int = 2,
        emb_dim_ignored: int = 128,              # kept for API symmetry; not used
        dropout: float = 0.0,
        premodule: Optional[str] = None,         # None or "NormalizeLogRadius"
        in_space: int = 16,                      # H=W for reshaping flat vectors
        attention_resolutions = (2, 4),          # ds factors (2=8x8, 4=4x4 for 16x16 input)
        conv_resample: bool = True,
        num_heads: int = 1,
        use_checkpoint: bool = False,
        learn_potential: bool = False,
        flatten_order: Literal["C","F"] = "C",   # how your (B,d) vectors were flattened
    ):
        super().__init__()
        assert premodule in (None, "NormalizeLogRadius")
        self.pre = NormalizeLogRadius() if premodule == "NormalizeLogRadius" else None
        self.in_space = in_space
        assert flatten_order in ("C","F")
        self.flatten_order = flatten_order

        self.core = UNetModelWithLogNorm(
            in_channels     = 1,
            model_channels  = base_channels,
            out_channels    = 1,
            in_space        = in_space,
            num_res_blocks  = num_res_blocks,
            attention_resolutions = attention_resolutions,
            dropout         = dropout,
            channel_mult    = tuple(channel_mults),
            conv_resample   = conv_resample,
            dims            = 2,
            num_classes     = None,
            use_checkpoint  = use_checkpoint,
            num_heads       = num_heads,
            use_scale_shift_norm = False,
            learn_potential = learn_potential,
            use_log_norm    = (premodule == "NormalizeLogRadius"),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d=H*W) or (B,1,H,W)
        t: (B,) or (B,1)
        """
        t = t.view(-1)  # (B,)
        need_flat = False

        if self.pre is not None:
            x, log_norm = self.pre(x)  # (batch, learnable_network_input_dim)
            x = x * torch.sqrt(torch.tensor(x.shape[-1], dtype=log_norm.dtype, device=log_norm.device))  # scale to keep std consistent

        if x.dim() == 2:
            B, d = x.shape
            H = W = self.in_space
            assert d == H*W, f"Flat dim {d} != {H}*{W}"
            x_img = flat_to_img(x, H, W, order=self.flatten_order)
            need_flat = True
        elif x.dim() == 4:
            assert x.size(1) == 1, f"Expected (B,1,H,W), got {tuple(x.shape)}"
            x_img = x
        else:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}")
        
        if plot_debug:
            xcopy = x_img.clone()
            with torch.no_grad():
                for i in range(min(xcopy.size(0), 5)):
                    print("NNimage_Pre or NoPre_Out_" + str(i) + ".png")
                    xt = xcopy[i,0,:,:].cpu().squeeze().numpy()
                    # print(xt.shape)
                    plots_vort(xt,vmin=-1,vmax=1)
                    plt.show(block=False)
                    if self.pre is None:
                        name_fig = "images/NNimage_NoPre_Out_" + str(i) + ".png"
                    else:
                        name_fig = "images/NNimage_Pre_Out_" + str(i) + ".png"
                    plt.savefig(name_fig)
                    plt.pause(1)
                plt.close()
                plt.close('all')

        if self.pre is None:
            y_img = self.core(x_img, timesteps=t)  # time-only
        else:
            y_img = self.core(x_img, timesteps=t, log_norm=log_norm)
        
        if need_flat:
            return img_to_flat(y_img, order=self.flatten_order)  # (B, d)
        else:
            return y_img

