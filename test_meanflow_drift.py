# -*- coding: utf-8 -*-
"""
Visual check: forward SDE with the mean-flow (traveling-wave) Stratonovich
drift F, noise disabled entirely (disable_noise=True) -- so the only thing
moving the field is pure advection by a moving Taylor-Green 4-vortex
streamfunction psi_w=4*A_beta*sin(2pi(x-V0t))*sin(2pi(y-V0t)) (V0, A_beta
auto-sized to the CFL limit). Run this, then look at the saved PNG: each
row is one sample, columns are snapshots over time -- the pattern should
translate/rotate smoothly with no added noise texture, and (per the exact
skew-symmetry of the drift, see SDEs.py) energy should stay visibly
constant across columns.
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import PIV
from SDEs import MSGMsde, forward_SDE
from sde_scheme import rk4_stratonovich_sampler
from transportNoise import fourier_flat_to_spatial

np.random.seed(0); torch.manual_seed(0)

device = 'cpu'
npixel = 16
num_steps_forward = 128
n_samples = 4
n_snapshots = 8  # columns in the figure

sampler = PIV(npixel**2, normalized=True, largeImage=True, smoothing=2,
              localized=True, few_data=False, FFTfields=True, ntrain_max=float('inf'))

x_init = sampler.sample(64).to(device)
inf_sde = MSGMsde(x_init, beta_min=1, beta_max=1, t_epsilon=0.004, T=torch.tensor(1.0),
                   num_steps_forward=num_steps_forward, device=device,
                   estim_cst_norm_dens_r_T=False, norm_sampler='ecdf', norm_map='log',
                   denseTensor=False, sparse_tensor_type="AMSGM", plot_validate=False,
                   disable_noise=True, k_pattern=None, V0=None, A_beta=None)
print('name_SDE:', inf_sde.name_SDE)
print('auto V0:', inf_sde.V0, ' auto A_beta:', inf_sde.A_beta)

for_sde = forward_SDE(inf_sde, inf_sde.T)
x0 = sampler.sampletest(n_samples).to(device)

t0 = time.time()
xs = rk4_stratonovich_sampler(for_sde, x0, num_steps=num_steps_forward,
                               keep_all_samples=True, include_t0=True)
print(f'integration done in {time.time()-t0:.1f}s, xs shape {tuple(xs.shape)}')

energy = (xs**2).sum(dim=(2, 3))  # (time, batch)
print('energy per sample over time (should stay ~constant, pure advection):')
print(energy[[0, xs.shape[0]//2, -1]])

step_inds = np.linspace(0, xs.shape[0] - 1, n_snapshots).round().astype(int)
# The Taylor-Green forcing only nudges energy between *neighboring* Fourier
# modes each step (see transportNoise.grid_k.build_sparse_forcing) -- on a
# spatially-smooth field the raw image barely changes, so also plot
# frame[t]-frame[0] (independently rescaled per column) to make that
# genuine, if subtle, drift-induced redistribution visible.
fig, axs = plt.subplots(2*n_samples, n_snapshots, figsize=(2*n_snapshots, 4*n_samples))
imgs = np.stack([[fourier_flat_to_spatial(xs[i, row:row+1]).reshape(npixel, npixel).numpy()
                   for i in step_inds] for row in range(n_samples)])  # (sample, snap, npixel, npixel)
for row in range(n_samples):
    vmax_row = np.abs(imgs[row]).max()
    for col, i in enumerate(step_inds):
        axs[2*row, col].imshow(imgs[row, col], cmap='RdBu_r', vmin=-vmax_row, vmax=vmax_row)
        diff = imgs[row, col] - imgs[row, 0]
        dmax = max(np.abs(diff).max(), 1e-12)
        axs[2*row+1, col].imshow(diff, cmap='PuOr', vmin=-dmax, vmax=dmax)
        for r in (2*row, 2*row+1):
            axs[r, col].set_xticks([]); axs[r, col].set_yticks([])
        if row == 0:
            axs[2*row, col].set_title(f't={i/num_steps_forward:.2f}')
    axs[2*row, 0].set_ylabel(f'sample {row}\nframe')
    axs[2*row+1, 0].set_ylabel(f'sample {row}\nframe-frame[0]')

fig.suptitle(f'{inf_sde.name_SDE} -- drift only (V0={inf_sde.V0:.3g}, A_beta={inf_sde.A_beta:.3g}, terms={inf_sde._meanflow_terms})')
plt.tight_layout()

import os
os.makedirs('results', exist_ok=True)
out_path = f'results/{sampler.name}_{inf_sde.name_SDE}_meanflow_check.png'
plt.savefig(out_path, dpi=120)
print('saved to', out_path)
