# config.py
import numpy as np
import torch
from types import SimpleNamespace

def fourier_flat_to_spatial(y_fft_flat):
    """(B,n,2) [re,im] Fourier flat -> (B,n) real spatial flat, via ifft2().
    'F'-order reshape matches grid_k's I=p1+p2*N convention."""
    B, n, _ = y_fft_flat.shape
    N = int(round(n ** 0.5))
    y_img = torch.complex(y_fft_flat[..., 0], y_fft_flat[..., 1]).view(B, N, N).transpose(1, 2)
    y_sp_img = torch.fft.ifft2(y_img, dim=(1, 2)).real
    return y_sp_img.transpose(1, 2).reshape(B, n)

def spatial_flat_to_fourier(y_sp_flat):
    """Inverse of fourier_flat_to_spatial: (B,n) real -> (B,n,2) [real,imag], via fft2()."""
    B, n = y_sp_flat.shape
    N = int(round(n ** 0.5))
    y_sp_img = y_sp_flat.view(B, N, N).transpose(1, 2)
    y_fft_img = torch.fft.fft2(y_sp_img, dim=(1, 2))
    y_fft_flat = y_fft_img.transpose(1, 2).reshape(B, n)
    return torch.stack([y_fft_flat.real, y_fft_flat.imag], dim=-1)

# MPS chokes on large (B*nnz) gathers/scatters well below actual memory
# capacity; chunk the index dimension to stay under that.
_MPS_SAFE_INDEX_BUDGET = 64 * 2_097_152

def mps_chunk_size(nnz, batch):
    return max(1, _MPS_SAFE_INDEX_BUDGET // max(batch, 1))

def mps_safe_gather(tensor, dim, index):
    """torch.index_select, chunked on MPS for large indices."""
    if tensor.device.type != 'mps':
        return torch.index_select(tensor, dim, index)
    chunk = mps_chunk_size(index.numel(), tensor.shape[0])
    if index.numel() <= chunk:
        return torch.index_select(tensor, dim, index)
    pieces = [torch.index_select(tensor, dim, index[i:i + chunk]) for i in range(0, index.numel(), chunk)]
    return torch.cat(pieces, dim=dim)

def mps_safe_scatter_add_(dest, dim, index, src):
    """dest.scatter_add_, chunked on MPS for large indices."""
    if dest.device.type != 'mps':
        return dest.scatter_add_(dim, index, src)
    n = index.shape[dim]
    chunk = mps_chunk_size(n, dest.shape[0])
    if n <= chunk:
        return dest.scatter_add_(dim, index, src)
    for start in range(0, n, chunk):
        idx_slice = [slice(None)] * index.dim()
        idx_slice[dim] = slice(start, start + chunk)
        src_slice = [slice(None)] * src.dim()
        src_slice[dim] = slice(start, start + chunk)
        dest.scatter_add_(dim, index[tuple(idx_slice)], src[tuple(src_slice)])
    return dest

def swap_channels(y):
    """(B,n,2) [re,im] -> (-im,re): multiplication by i. Used by the sparse
    advection tensors' I-channel (G^I, F^I)."""
    return torch.stack([-y[..., 1], y[..., 0]], dim=-1)

def sparse_yJ_products(y, J, V_R, V_I):
    """Shared gather+scale step for the sparse (I,J,K) advection tensors
    (G noise and F drift): R-channel (V_R) acts on y, I-channel (V_I) acts
    on swap_channels(y). Returns (V_R*y[J], V_I*(iy)[J])."""
    yJ_R = mps_safe_gather(y, 1, J)
    yJ_I = mps_safe_gather(swap_channels(y), 1, J)
    V_R_b = V_R.reshape(1, -1, *([1] * (y.dim() - 2)))
    V_I_b = V_I.reshape(1, -1, *([1] * (y.dim() - 2)))
    return V_R_b * yJ_R, V_I_b * yJ_I

def meanflow_beta_vectors(t, beta_map, n, device, dtype):
    """Builds (B,n) beta_R(t), beta_I(t) from a {K: (coefficient, omega)}
    map (grid_k.meanflow_beta): beta_R=coefficient*cos(omega*t),
    beta_I=coefficient*sin(omega*t), 0 elsewhere."""
    t_flat = t.reshape(-1)
    beta_R = torch.zeros(t_flat.shape[0], n, device=device, dtype=dtype)
    beta_I = torch.zeros(t_flat.shape[0], n, device=device, dtype=dtype)
    for K, (coefficient, omega) in beta_map.items():
        phase = omega * t_flat
        beta_R[:, K] = coefficient * torch.cos(phase)
        beta_I[:, K] = coefficient * torch.sin(phase)
    return beta_R, beta_I

def compute_ito_correction(indices, valuesR, valuesI, n):
    """Ito correction L_G = 0.5*sum_k(G^R_k^2 - G^I_k^2) for a sparse
    (I,J,K) advection tensor (see grid_k.build_sparse_G): each K-slice's
    Ghat^{k,R}/Ghat^{k,I} are skew, so their squares are negative
    semidefinite, matching the Laplacian form L_PQ=-(a0/2)*K^2*delta_PQ.

    indices: (3,nnz) int64 [I,J,K]. valuesR, valuesI: (nnz,) float (G^R, G^I).
    Returns a CPU torch.Tensor (n,n); caller moves it to device.
    """
    L_G = torch.zeros(n, n)
    for k in range(n):
        mask = (indices[2] == k)
        Ik, Jk = indices[0][mask], indices[1][mask]
        VRk, VIk = valuesR[mask], valuesI[mask]
        MRk = torch.zeros(n, n)
        MRk.index_put_((Ik, Jk), VRk, accumulate=True)
        MIk = torch.zeros(n, n)
        MIk.index_put_((Ik, Jk), VIk, accumulate=True)
        L_G += 0.5 * (MRk @ MRk - MIk @ MIk)
    return L_G

class grid_k():
    def __init__(self, N=2, Lx=1, L_LS=1, rho_s=5/3, smoothing=False, anti_aliasing = True):
        # see "New Numerical Results for the Surface Quasi-Geostrophic
        # Equation", Constantin et al., J. Sci. Comput. (2012).
        alpha = 36.
        order = 19.
        K_max = np.pi * N / Lx
        
        self.rho_s: float = rho_s            # spatial spectrum slope of advecting velocity

        self.dim = 2
        self.N = N
        self.Lx = Lx
        self.L_LS = L_LS
        self.PN = N//2                                 # aliased wavevector index
        self.dx = self.Lx / N

        theta_1 = self.rho_s + self.dim + 1            # spectrum slope of the bidirectionnal spatial spectrum
        K_Lx = 2 * np.pi / self.Lx                     # largest-scale wavenumber
        K_LS = 2 * np.pi / self.L_LS                   # large-scale wavenumber
        self.K_LS = K_LS

        # Wavenumbers
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        if self.dim == 2 :
            KX, KY = np.meshgrid(kx, kx, indexing="ij")
            K = np.sqrt(KX**2 + KY**2)
            self.KX = KX
            self.KY = KY
        elif self.dim == 3 :
            KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
            K = np.sqrt(KX**2 + KY**2 + KZ**2)
            Kv = np.concat((KX, KY, KZ), axis= 3)
            self.KX = KX
            self.KY = KY
            self.KZ = KZ
            self.Kv = Kv
        else:
            raise ValueError("Incorrect dim")
        self.K = K

        # Spectrum of advecting velocity
        self.alpha2_K = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
        if anti_aliasing:
            self.alpha2_K *= np.exp( - alpha * (K/K_max)**order)
        self.cleanAliasing(self.alpha2_K)
        a0_temp = self.a0()
        self.alpha2_K /= a0_temp  # normalize to a0=1

        # f_Kp_Kq[p1,p2,q1,q2] = p^perp . q, an antisymmetric bilinear form
        # of the two wavevectors (broadcast over a full (N,N,N,N) grid).
        KX_p = KX[:, :, np.newaxis, np.newaxis]
        KY_p = KY[:, :, np.newaxis, np.newaxis]
        KX_q = KX[np.newaxis, np.newaxis, :, :]
        KY_q = KY[np.newaxis, np.newaxis, :, :]
        self.f_Kp_Kq = KX_p * KY_q - KY_p * KX_q
        if anti_aliasing:
            self.f_Kp_Kq *= np.exp( - alpha * ( (KX_q**2+KY_q**2)/(K_max**2) )**(order/2) )


    def a0(self):
        """Diffusion coefficient of the advecting velocity (Resseguier,
        Hascoet & Chapron 2024, JFM eq. 4.7); dt<=dx^2/a0. Denominator is
        N^(2*dim+2), not N^(2*dim) -- verified against L_G's own diagonal
        (diag(L_G)/K^2 is only N-independent with the extra N^2)."""
        return np.sum(self.K**2 * self.alpha2_K) / (2 * self.N**(2*self.dim+2))

    def gamma0(self):
        """Rate of velocity-gradient creation by the advecting noise (same
        reference, eq. 4.8); calibrates the targeted hyperdiffusion."""
        factor = 1
        return factor * np.sum(self.K**4 * self.alpha2_K) / (8 * self.N**(2*self.dim+2))

    # Empirical margin below the true instability onset (energy growth
    # turns severe around cst~0.16-0.2, diverges by cst~0.3); kept low to
    # favor robustness over maximal noise strength.
    _DIFFUSIVE_CFL_SAFETY = 0.1

    def cfl_dt_max(self, beta):
        """Diffusive CFL bound dt <= cst*dx^2/(beta*a0)."""
        return self._DIFFUSIVE_CFL_SAFETY * self.dx**2 / (beta * self.a0())

    def cleanAliasing(self,S1):
        if self.dim == 2 :
            S1[0, 0] = 0
            S1[self.PN, :] = 0
            S1[:, self.PN] = 0
        elif self.dim == 3 :
            S1[0, 0, 0] = 0
            S1[self.PN, :, :] = 0
            S1[:, self.PN, :] = 0
            S1[:, :, self.PN] = 0
        else:
            raise ValueError("Incorrect dim")

    def rescale_for_cfl(self, beta_max, dt):
        """Rescales alpha2_K (hence G) down to meet the diffusive CFL
        exactly if (beta_max, dt) violates it; a no-op otherwise. Mutates
        self.alpha2_K in place. Returns (rescale or None, a0_before, dt_max)."""
        a0_before = self.a0()
        dt_max = self.cfl_dt_max(beta_max)
        if dt > dt_max:
            target_a0 = self._DIFFUSIVE_CFL_SAFETY * self.dx**2 / (beta_max * dt)
            rescale = target_a0 / a0_before
            self.alpha2_K *= rescale
            return rescale, a0_before, dt_max
        return None, a0_before, dt_max

    def meanflow_gamma0(self, terms):
        """Mean-flow drift's own gamma0 analogue: 1/tau_w ~ ||grad(w)|| ~
        ||k||^2*amplitude (w=curl(psi), so grad(w) carries one extra ||k||
        over w~||k||*amplitude itself). No extra IFFT2/N^2 factor: amplitude
        is already physical, and ||k||^2*amplitude is already a rate (1/time).
        terms: [(k1,k2,amplitude),...] (e.g. self._meanflow_terms)."""
        if not terms:
            return 0.0
        return max(np.hypot(self.KX[k1 % self.N, k2 % self.N], self.KY[k1 % self.N, k2 % self.N])**2 * abs(amp)
                    for k1, k2, amp in terms)

    def build_hyperdiffusion(self, z=1, extra_gamma0=0.0):
        """Hyperdiffusion rate, diagonal in Fourier space:
        D_hyper = gamma0*(dx*K)^(2z) (dx*K in [0,pi], so ~0 at large scale,
        ~gamma0*pi^(2z) at Nyquist). gamma0 = self.gamma0() + extra_gamma0
        (e.g. the mean-flow drift's own analogous rate, meanflow_gamma0,
        needed since the noise's gamma0 alone ignores a deterministic
        advecting velocity). Returns (D_hyper, gamma0), D_hyper flattened
        in 'F' order to match the sparse tensor's I=p1+p2*N convention."""
        gamma0 = self.gamma0() + extra_gamma0
        D_hyper = gamma0 * (self.dx * self.K) ** (2 * z)   # (N,N), same (k1,k2) layout as self.K
        return D_hyper.flatten(order='F'), gamma0

    def build_hyperdiffusion_capped(self, z, extra_gamma0, dt, margin=2.0):
        """build_hyperdiffusion, then cap gamma0 so dt*D_hyper_max never
        exceeds `margin`: hyperdiffusion is stepped EXPLICITLY by RK4 (see
        SDEs.MSGMsde.f_strato), whose real-axis stability limit is
        |z|<~2.785 -- past that it doesn't under-damp, it blows up the
        highest wavenumbers. Returns (D_hyper, gamma0)."""
        D_hyper, gamma0 = self.build_hyperdiffusion(z=z, extra_gamma0=extra_gamma0)
        dt_D_hyper_max = dt * D_hyper.max()
        if dt_D_hyper_max > margin:
            cap_rescale = margin / dt_D_hyper_max
            print(f"Capping hyperdiffusion by {cap_rescale:.3g} (gamma0: {gamma0:.3g} -> {gamma0*cap_rescale:.3g}) "
                  f"to keep dt*D_hyper_max={dt_D_hyper_max:.3g} within RK4's explicit stability margin ({margin:.3g}).")
            D_hyper = D_hyper * cap_rescale
            gamma0 = gamma0 * cap_rescale
        return D_hyper, gamma0

    def _build_sparse_kpq(self, k_list, weight_by_k=None):
        """Shared {k,-k}-folded (I,J,K) builder for both sparse advection
        tensors from f_Kp_Kq: the noise G (build_sparse_G, weight=
        sqrt(alpha2_K)) and the drift F (build_sparse_forcing, weight=1,
        i.e. F^{k,.}=G^{k,.}/alpha(k)). Each K-slice's R-channel is exactly
        skew, I-channel exactly symmetric. k_list=ALL (k1,k2) gives G's
        full O(N^4) tensor; a few explicit wavevectors gives F's O(N^2)-
        per-k one. Returns (i_list, j_list, k_list_out, vR_list, vI_list).
        """
        N = self.N
        if weight_by_k is None:
            weight_by_k = lambda k1, k2: 1.0
        i_list, j_list, k_list_out, vR_list, vI_list = [], [], [], [], []
        for (k1z, k2z) in k_list:
            K = (k1z % N) + (k2z % N) * N
            w = weight_by_k(k1z % N, k2z % N)
            for p1 in range(N):
                for p2 in range(N):
                    I = p1 + p2 * N
                    q1, q2 = (p1 + k1z) % N, (p2 + k2z) % N
                    coef1 = -w * self.f_Kp_Kq[p1, p2, q1, q2] / N**3
                    i_list.append(I); j_list.append(q1 + q2*N); k_list_out.append(K)
                    vR_list.append(coef1); vI_list.append(coef1)

                    q1n, q2n = (p1 - k1z) % N, (p2 - k2z) % N
                    coef2 = -w * self.f_Kp_Kq[p1, p2, q1n, q2n] / N**3
                    i_list.append(I); j_list.append(q1n + q2n*N); k_list_out.append(K)
                    vR_list.append(coef2); vI_list.append(-coef2)
        return i_list, j_list, k_list_out, vR_list, vI_list

    def build_sparse_G(self):
        """G^R (skew, per-channel) and G^I (symmetric, cross-channel)
        sparse tensors driving the two independent real noises dB^R_t(k),
        dB^I_t(k), weighted by sqrt(alpha2_K) (the advecting velocity's
        spectrum) -- see _build_sparse_kpq. Returns (i_list, j_list,
        k_list, vR_list, vI_list)."""
        N = self.N
        all_k = [(k1, k2) for k1 in range(N) for k2 in range(N)]
        return self._build_sparse_kpq(all_k, weight_by_k=lambda k1, k2: np.sqrt(self.alpha2_K[k1, k2]))

    def build_sparse_forcing(self, k_list):
        """Same construction as build_sparse_G but for explicit k's only
        and unweighted (F^{k,.}=G^{k,.}/alpha(k)), combined into ONE
        sparse tensor with K set per term so a per-mode beta(t) vector can
        be gathered via K like dW is for the noise (see sparse_yJ_products)."""
        return self._build_sparse_kpq(k_list)

    def taylor_green_terms(self, A_beta=1.0):
        """[(k1,k2,amplitude), ...] reproducing the moving 4-vortex
        streamfunction psi_w = 4*A_beta*sin(2*pi*(x-V0*t))*sin(2*pi*(y-V0*t))
        via sin(A)sin(B)=0.5*(cos(A-B)-cos(A+B)): "A-B" at k_a=(1,-1),
        "A+B" at k_b=(1,1) with a flipped amplitude sign."""
        return [(1, -1, A_beta), (1, 1, -A_beta)]

    def meanflow_beta(self, terms, V0):
        """Per-mode {K: (coefficient, omega)}, the deterministic analogue
        of the noise's dB^R_t(k)/dB^I_t(k): beta_R(t,K)=coefficient*cos
        (omega*t), beta_I(t,K)=coefficient*sin(omega*t), with
        coefficient=-N^3*amplitude (matching _build_sparse_kpq's 1/N^3 and
        the unnormalized-FFT convention) and omega=2*pi*(k1+k2)*V0."""
        N = self.N
        return {(k1 % N) + (k2 % N) * N: (-N**3 * amp, 2 * np.pi * (k1 + k2) * V0)
                for (k1, k2, amp) in terms}

    def meanflow_w_max(self, k_pattern, n_grid=256):
        """True peak |w| of the streamfunction psi = sum 2*amp*cos(2*pi*
        (k1*x+k2*y)) reconstructed from k_pattern, evaluated on a fine
        grid: the true peak of several superposed modes can exceed any
        single mode's own ||k||*|amplitude| (constructive interference),
        e.g. exactly 2*sqrt(2)x for taylor_green_terms."""
        x = np.linspace(0, 1, n_grid, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        dpsidx = np.zeros_like(X)
        dpsidy = np.zeros_like(X)
        for k1, k2, amp in k_pattern:
            phase = 2 * np.pi * (k1 * X + k2 * Y)
            dpsidx += -2 * amp * 2 * np.pi * k1 * np.sin(phase)
            dpsidy += -2 * amp * 2 * np.pi * k2 * np.sin(phase)
        return float(np.hypot(dpsidx, dpsidy).max())

    def meanflow_cfl(self, V0, A_beta, dt, k_pattern=None):
        """Auto-sizes (or diagnoses) V0 and A_beta against two CFL bounds:
        the pattern's phase speed (dt<=dx/(V0*sqrt(2))) and its true peak
        velocity (dt<=dx/||w||, from meanflow_w_max). None values are sized
        to sit exactly at their limit. k_pattern defaults to
        taylor_green_terms's unit amplitudes. Returns (V0, A_beta, terms)."""
        if k_pattern is None:
            k_pattern = self.taylor_green_terms(1.0)
        if V0 is None:
            V0 = self.dx / dt / np.sqrt(2) / 2
        unit_w_max = self.meanflow_w_max(k_pattern)
        if A_beta is None:
            A_beta = (self.dx / dt) / unit_w_max if unit_w_max > 0 else 0.0

        terms = [(k1, k2, u * A_beta) for k1, k2, u in k_pattern]
        V0_norm = abs(V0) * np.sqrt(2)
        dt_max_V0 = self.dx / V0_norm if V0_norm > 0 else float('inf')
        w_max = unit_w_max * abs(A_beta)
        dt_max_w = self.dx / w_max if w_max > 0 else float('inf')
        dt_max = min(dt_max_V0, dt_max_w)
        print(f"Mean-flow drift: terms={[(k1, k2, round(a, 4)) for k1, k2, a in terms]}, V0={V0:.3g}, "
              f"||w||~{w_max:.3g}, dt={dt:.3g}, dt_max(V0)={dt_max_V0:.3g}, dt_max(||w||)={dt_max_w:.3g}"
              + (f" -- WARNING: dt > dt_max={dt_max:.3g}" if dt > dt_max else ""))
        return V0, A_beta, terms

def build_advection_state(N, anti_aliasing, T, num_steps_forward, beta_max,
                           V0, A_beta, k_pattern, device):
    """Builds a grid_k plus every sparse tensor MSGMsde.sparse_G_advection
    needs (noise G, drift F, hyperdiffusion, Ito correction), already on
    `device`. Returns a SimpleNamespace: G_I/J/K/V/V_I, G_sparse_cpu, L_G,
    D_hyper, V0, A_beta, meanflow_terms, and (only if A_beta!=0)
    F_I/J/K/V_R/V_I, meanflow_beta_map."""
    grid = grid_k(N=N, anti_aliasing=anti_aliasing)
    dt = T / num_steps_forward

    rescale, a0_before, dt_max = grid.rescale_for_cfl(beta_max, dt)
    if rescale is not None:
        print(f"Rescaling G amplitude by {rescale:.3g} (a0: {a0_before:.3g} -> {grid.a0():.3g}) "
              f"to meet the diffusive CFL condition (dt={dt:.3g} > dt_max={dt_max:.3g} otherwise).")
    print(f"A-MSGM SPDE info: dx={grid.dx:.3g}, L={grid.Lx:.3g}, T={T:.3g}, "
          f"num_steps_forward={num_steps_forward}, dt={dt:.3g}, beta={beta_max:.3g}, "
          f"a0={grid.a0():.3g}, dt_max={dt_max:.3g}")

    state = SimpleNamespace()
    state.V0, state.A_beta, state.meanflow_terms = grid.meanflow_cfl(V0, A_beta, dt, k_pattern=k_pattern)

    if state.A_beta != 0:
        i, j, k, vR, vI = grid.build_sparse_forcing([(k1, k2) for k1, k2, _ in state.meanflow_terms])
        state.F_I = torch.tensor(i, dtype=torch.int64, device=device)
        state.F_J = torch.tensor(j, dtype=torch.int64, device=device)
        state.F_K = torch.tensor(k, dtype=torch.int64, device=device)
        state.F_V_R = torch.tensor(vR, dtype=torch.float32, device=device)
        state.F_V_I = torch.tensor(vI, dtype=torch.float32, device=device)
        state.meanflow_beta_map = grid.meanflow_beta(state.meanflow_terms, state.V0)

    # Always on: the noise's own gamma0() alone (grid-scale content from
    # the stochastic advection) needs damping even with no drift, or
    # aliasing artifacts build up over time steps (gamma0_w adds the
    # drift's own contribution on top, 0 when there's no drift).
    gamma0_w = grid.meanflow_gamma0(state.meanflow_terms)
    D_hyper, gamma0 = grid.build_hyperdiffusion_capped(z=1, extra_gamma0=gamma0_w, dt=dt)
    state.D_hyper = torch.from_numpy(D_hyper).float().to(device)
    if gamma0 > 0:
        hyperdiff_dt_max = 1 / gamma0
        print(f"Hyperdiffusion: z=1, gamma0={gamma0:.3g} (of which mean-flow gamma0_w={gamma0_w:.3g} pre-cap), "
              f"(1/gamma0={hyperdiff_dt_max:.3g} s), max D_hyper={D_hyper.max():.3g} (at Nyquist), dt={dt:.3g}"
              + (f" -- WARNING: dt > 1/gamma0={hyperdiff_dt_max:.3g}" if dt > hyperdiff_dt_max else ""))
    else:
        print("Hyperdiffusion: disabled (gamma0=0)")

    n = N * N
    i, j, k, vR, vI = grid.build_sparse_G()
    indices = torch.tensor([i, j, k], dtype=torch.int64)
    valuesR = torch.tensor(vR, dtype=torch.float32)
    valuesI = torch.tensor(vI, dtype=torch.float32)
    try:
        state.G_sparse_cpu = torch.sparse_coo_tensor(indices, valuesR, size=(n, n, n)).coalesce()
    except Exception:
        state.G_sparse_cpu = None
    state.G_I = indices[0].to(device)
    state.G_J = indices[1].to(device)
    state.G_K = indices[2].to(device)
    state.G_V = valuesR.to(device)
    state.G_V_I = valuesI.to(device)
    state.L_G = compute_ito_correction(indices, valuesR, valuesI, n).to(device)
    print(f"trace(L_G) = {torch.trace(state.L_G).item():.3g} (was -0.5*dim = {-0.5*n:.3g} for the placeholder elsewhere)")
    return state
