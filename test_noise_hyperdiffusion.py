# -*- coding: utf-8 -*-
"""
Visual check: forward SDE with ONLY the stochastic noise G active (no
drift, V0=A_beta=0), at three hyperdiffusion strengths -- none, a
fraction of the noise's own gamma0()-calibrated rate, and the full
(uncapped-unless-needed) rate -- to compare grid-scale aliasing buildup
against energy loss. Each row is one hyperdiffusion setting, columns are
snapshots over time.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import PIV
from SDEs import MSGMsde, forward_SDE
from sde_scheme import rk4_stratonovich_sampler
from transportNoise import fourier_flat_to_spatial

device = 'cpu'
npixel = 16
num_steps_forward = 128
n_snapshots = 8
hyper_fractions = [0.0, 0.2, 1.0]  # of the full gamma0()-calibrated D_hyper

sampler = PIV(npixel**2, normalized=True, largeImage=True, smoothing=2,
              localized=True, few_data=False, FFTfields=True, ntrain_max=float('inf'))

np.random.seed(0); torch.manual_seed(0)
x_init = sampler.sample(64).to(device)
inf_sde = MSGMsde(x_init, beta_min=1, beta_max=1, t_epsilon=0.004, T=torch.tensor(1.0),
                   num_steps_forward=num_steps_forward, device=device,
                   estim_cst_norm_dens_r_T=False, norm_sampler='ecdf', norm_map='log',
                   denseTensor=False, sparse_tensor_type="AMSGM", plot_validate=False,
                   disable_noise=False, k_pattern=None, V0=0, A_beta=0)
D_hyper_full = inf_sde.D_hyper.clone()
x0 = sampler.sampletest(1).to(device)

fig, axs = plt.subplots(2*len(hyper_fractions), n_snapshots,
                         figsize=(2*n_snapshots, 4*len(hyper_fractions)))
step_inds = np.linspace(0, num_steps_forward, n_snapshots).round().astype(int)

for row, frac in enumerate(hyper_fractions):
    np.random.seed(1); torch.manual_seed(1)
    inf_sde.D_hyper = D_hyper_full * frac
    for_sde = forward_SDE(inf_sde, inf_sde.T)
    xs = rk4_stratonovich_sampler(for_sde, x0, num_steps=num_steps_forward,
                                   keep_all_samples=True, include_t0=True)
    energy = (xs**2).sum(dim=(2, 3))
    print(f'hyper_fraction={frac}: energy at t=[0, mid, end] = '
          f'{energy[0].item():.2f}, {energy[xs.shape[0]//2].item():.2f}, {energy[-1].item():.2f}')

    imgs = [fourier_flat_to_spatial(xs[i]).reshape(npixel, npixel).numpy() for i in step_inds]
    vmax_row = max(np.abs(im).max() for im in imgs)
    for col, (i, img) in enumerate(zip(step_inds, imgs)):
        axs[2*row, col].imshow(img, cmap='RdBu_r', vmin=-vmax_row, vmax=vmax_row)
        # spectrum (log |FFT|) to make grid-scale aliasing content visible directly
        spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))
        axs[2*row+1, col].imshow(spec, cmap='inferno')
        for r in (2*row, 2*row+1):
            axs[r, col].set_xticks([]); axs[r, col].set_yticks([])
        if row == 0:
            axs[2*row, col].set_title(f't={i/num_steps_forward:.2f}')
    axs[2*row, 0].set_ylabel(f'hyper x{frac}\nframe')
    axs[2*row+1, 0].set_ylabel(f'hyper x{frac}\nlog|FFT|')

fig.suptitle(f'{inf_sde.name_SDE} -- noise only, hyperdiffusion fractions {hyper_fractions}')
plt.tight_layout()

import os
os.makedirs('results', exist_ok=True)
out_path = f'results/{sampler.name}_{inf_sde.name_SDE}_noise_hyperdiffusion_check.png'
plt.savefig(out_path, dpi=120)
print('saved to', out_path)
