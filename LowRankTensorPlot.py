import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------------------------------
# Parameters
# -------------------------------------------------------------
T = 100.0
N = 20000
dt = T/N
d = 4
sqrt_d = np.sqrt(d)

lambda1 = 1.0
lambda2 = 10.0

trajectory = False # trajectory or iid latent variables
from_uniform = False # from uniform or from Brownian motion
random_init = False # random initial condition or fixed initial condition
init_value = 1.0 # init = init_value * [1 1 1 1] 
kill_dim34 = False # kill dimensions 3 and 4 (only plot dim 1 and 2)

# ---------------------------------------------------------------------
# Matplotlib global styling for ICLR-level figures
# ---------------------------------------------------------------------
rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "savefig.transparent": True,
})
fig_size = 2  # inches
# Use a clean contrasting color instead of black
COL = "#1f77b4"   # matplotlib blue

# -------------------------------------------------------------

if trajectory:
    print("Simulating Brownian motion trajectories")
    # Brownian motion for trajectories 
    N_T = N
    N_init = 1  # to store initial condition
    np.random.seed(0)
    dW = np.sqrt(dt) * np.random.randn(N)
    B = np.empty(N+1)
    B[0] = 0.0
    B[1:] = np.cumsum(dW)
    from_uniform = False  # irrelevant for trajectories
else:
    print("Sampling independent Brownian motion latent variables")
    # Brownian motion for iid latent variables  
    N_T = 1 
    N_init = N+1  
    np.random.seed(0)
    if from_uniform:
        U = 1e3 * np.random.rand(N+1) 
    else:
        B = np.sqrt(T) * np.random.randn(N+1)
        

# initialize arrays to store trajectories
x1 = np.zeros(N+1)
x2 = np.zeros(N+1)
x3 = np.zeros(N+1)
x4 = np.zeros(N+1)
x01 = np.zeros(N_init)
x02 = np.zeros(N_init)
x03 = np.zeros(N_init)
x04 = np.zeros(N_init)

if random_init:
    print("Using random initial condition")
    # random initial condition
    mean_init = 1.0
    std_init = 0.1
    x01[0:N_init] = mean_init + std_init * np.random.randn(N_init)
    x02[0:N_init] = mean_init + std_init * np.random.randn(N_init)
    x03[0:N_init] = mean_init + std_init * np.random.randn(N_init)
    x04[0:N_init] = mean_init + std_init * np.random.randn(N_init)
else:   
    print("Using fixed initial condition")
    # fix initial condition
    if kill_dim34:
        init_value *= np.sqrt(2)
    x01[0:N_init] = init_value * np.ones(N_init)
    x02[0:N_init] = init_value * np.ones(N_init)
    x03[0:N_init] = (1-kill_dim34) * init_value * np.ones(N_init)
    x04[0:N_init] = (1-kill_dim34) * init_value * np.ones(N_init)

    
# X0 = np.array([1,1,1,1], dtype=float)

for k in range(N+1):
    if from_uniform:
        th1 = lambda1 * U[k]
        th2 = lambda2 * U[k]
    else:
        th1 = sqrt_d * lambda1 * B[k]
        th2 = sqrt_d * lambda2 * B[k]

    idx_init = 0 if trajectory else k

    x1[k] = x02[idx_init]*np.sin(th1) + x01[idx_init]*np.cos(th1)
    x2[k] = x02[idx_init]*np.cos(th1) - x01[idx_init]*np.sin(th1)

    x3[k] = x04[idx_init]*np.sin(th2) + x03[idx_init]*np.cos(th2)
    x4[k] = x04[idx_init]*np.cos(th2) - x03[idx_init]*np.sin(th2)

if not trajectory:
    x1=x1[1:-1]
    x2=x2[1:-1]
    x3=x3[1:-1]
    x4=x4[1:-1]

# ---------------------------------------------------------------------
# 2D scatter: x1 vs x2
# ---------------------------------------------------------------------
plt.figure(figsize=(fig_size, fig_size))
plt.scatter(x1, x2, s=3, c=COL, alpha=0.25, edgecolors="none")
plt.axis('equal')
xlim=plt.xlim()
ylim=plt.ylim()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.tight_layout()
plt.savefig("x1_x2.png")

# ---------------------------------------------------------------------
# 2D scatter: x1 vs x3
# ---------------------------------------------------------------------
plt.figure(figsize=(fig_size, fig_size))
plt.scatter(x1, x3, s=3, c=COL, alpha=0.25, edgecolors="none")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_3$")
plt.tight_layout()
plt.savefig("x1_x3.png")

# ---------------------------------------------------------------------
# 2D scatter: x1 vs x4
# ---------------------------------------------------------------------
plt.figure(figsize=(fig_size, fig_size))
plt.scatter(x1, x4, s=3, c=COL, alpha=0.25, edgecolors="none")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_4$")
plt.tight_layout()
plt.savefig("x1_x4.png")

# ---------------------------------------------------------------------
# 2D scatter: x4 vs x3
# ---------------------------------------------------------------------
plt.figure(figsize=(fig_size, fig_size))
plt.scatter(x4, x3, s=3, c=COL, alpha=0.25, edgecolors="none")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r"$x_4$")
plt.ylabel(r"$x_3$")
plt.tight_layout()
plt.savefig("x4_x3.png")  

# ---------------------------------------------------------------------
# 3D scatter
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(4, 2))
ax = fig.add_subplot(111, projection="3d")

# --- depth-based fading for visibility ---
z_norm = (x3 - x3.min()) / (x3.max() - x3.min() + 1e-9)
colors = plt.cm.Blues(0.3 + 0.7 * z_norm)

ax.scatter(
    x1, x2, x3,
    s=6,
    c=colors,
    edgecolors="none",
    depthshade=True,
    alpha=0.85
)
plt.axis('equal')

# --- labels (reduced fontsize) ---
ax.set_xlabel(r"$x_1$", labelpad=3, fontsize=8)
ax.set_ylabel(r"$x_2$", labelpad=3, fontsize=8)
ax.set_zlabel(r"$x_3$", labelpad=3, fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=7)

# --- clean background ---
ax.xaxis.pane.set_alpha(0.06)
ax.yaxis.pane.set_alpha(0.06)
ax.zaxis.pane.set_alpha(0.06)
ax.grid(False)
# --- the ONLY reliable fix ---
fig.subplots_adjust(
    left=0.0,     # increase left margin
    right=0.95,
    bottom=0.23,   # increase bottom
    top=0.95
)

# IMPORTANT: DO NOT USE bbox_inches="tight" for 3D!
fig.savefig(
    "x1_x2_x3.png",
    dpi=300,
    pad_inches=0.1
)