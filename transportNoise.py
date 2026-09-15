# config.py
import numpy as np
import torch

def fourier_flat_to_spatial(y_fft_flat):
    """
    (B,n,2) [real,imag] Fourier-domain flat vector -> (B,n) real
    spatial-domain flat vector, via ifft2().real. Uses 'F'-order reshaping
    (row index = p1, column index = p2) to match grid_k's flat-index
    convention I=p1+p2*N (see coef_G_KPQ/build_sparse_G). Distinct from
    NNUnet.flat_to_img/img_to_flat, which wrap the same ifft2/fft2 step for
    feeding a CNN (extra scale_image rescaling, image-shaped output,
    "C"/"F" order flexibility) rather than converting raw SDE state.
    """
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

class grid_k():
    def __init__(self, N=2, Lx=1, L_LS=1, rho_s=5/3, smoothing=False, anti_aliasing = True):
        # % see "New Numerical Results for the Surface Quasi-Geostrophic
        # % Equation", Constantin et al., J. Sci. Comput. (2012).
        alpha = 36.
        order = 19.
        K_max = np.pi * N / Lx
        # maskx = exp(-alpha*( (2./model.grid.MX(1)).*abs(nx) ).^order);
        # masky = exp(-alpha*( (2./model.grid.MX(2)).*abs(ny) ).^order);
        
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
            # KX = KX[:,:,np.newaxis]
            # KY = KY[:,:,np.newaxis]
            K = np.sqrt(KX**2 + KY**2)
            self.KX = KX
            self.KY = KY
        elif self.dim == 3 :
            KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
            # KX = KX[:,:,:,np.newaxis]
            # KY = KY[:,:,:,np.newaxis]
            # KZ = KZ[:,:,:,np.newaxis]
            K = np.sqrt(KX**2 + KY**2 + KZ**2)
            Kv = np.concat((KX, KY, KZ), axis= 3)
            # Kv = Kv[:,:,:,:,np.newaxis]
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

        # dual grid: broadcast KX,KY over a full (N,N,N,N) grid of (k1,k2,q1,q2)
        # pairs -- KX_k/KY_k vary only over the first two axes (k1,k2), KX_q/KY_q
        # only over the last two (q1,q2), so f_Kk_Kq[k1,k2,q1,q2] = k^perp . q
        # (a cross-product-like antisymmetric bilinear form of the two wavevectors).
        KX_p = KX[:, :, np.newaxis, np.newaxis]
        KY_p = KY[:, :, np.newaxis, np.newaxis]
        KX_q = KX[np.newaxis, np.newaxis, :, :]
        KY_q = KY[np.newaxis, np.newaxis, :, :]
        self.f_Kp_Kq = KX_p * KY_q - KY_p * KX_q
        if anti_aliasing:
            self.f_Kp_Kq *= np.exp( - alpha * ( (KX_q**2+KY_q**2)/(K_max**2) )**(order/2) )

    def coef_G_KPQ(self,k1,k2,p1,p2,q1,q2):
        # 1/N^3 discrete-normalization factor: the advection term is a
        # real-space product v*grad(q), so building it as a discrete
        # Fourier-space convolution sum (as sparse_G_advection does) needs
        # the usual 1/N^2 (np.fft's unnormalized convention: pointwise
        # product in real space <-> convolution/N^dim in Fourier space, here
        # dim=2), plus one more 1/N tied to how alpha2_K's own normalization
        # relates to that convolution sum. Verified empirically: without
        # this, L_G's spectral radius exceeds the theoretical Laplacian
        # Ito-correction magnitude -(a0/2)*K^2 by a factor growing like N^6
        # across resolutions; with it, the ratio is O(1) and stable across N
        # (residual ~0.5, still being tracked down).
        return - np.sqrt(self.alpha2_K[k1,k2]) * self.f_Kp_Kq[p1,p2,q1,q2] / self.N**3

    def a0(self):
        """
        Characteristic diffusion coefficient a0 of the advecting velocity
        field: a0 = (1/(2 dt)) E||sigma dB_t||^2, the pointwise real-space
        variance rate (Resseguier, Hascoet & Chapron 2024, JFM, eq. 4.7),
        used in the diffusive CFL condition dt <= dx^2/a0 (their PhD thesis
        appendix C.1). It is a property of the velocity spectrum alone (not
        of the advection tensor's directional structure f_Kp_Kq).

        From the A-MSGM tensor note, alpha2_K[k1,k2] = E||sigma dB_hat_t(k)||^2
        / (dt ||k||^2), i.e. E||sigma dB_hat_t(k)||^2/dt = K[k1,k2]^2 *
        alpha2_K[k1,k2]. Discrete Parseval for np.fft's convention
        (sum_x |v|^2 = (1/N^dim) sum_k |v_hat|^2) plus homogeneity (every
        point has the same pointwise variance) gives:
            a0 = (1/(2 N^(2*dim))) * sum_{k1,k2} K[k1,k2]^2 * alpha2_K[k1,k2]
        """
        return np.sum(self.K**2 * self.alpha2_K) / (2 * self.N**(2*self.dim))

    def gamma0(self):
        """
        Characteristic (squared) rate of velocity-gradient creation by the
        advecting noise: gamma0 = (1/(8 dt)) E||grad_x(sigma dB_t)^T||^2
        (Resseguier, Hascoet & Chapron 2024, JFM, eq. 4.8); 1/gamma0 is the
        characteristic time of gradient creation. Same discrete-Parseval
        derivation as a0() (eq. 4.7), but weighted by an extra ||k||^2 since
        differentiating multiplies the Fourier coefficient by k:
            gamma0 = (1/(8 N^(2*dim))) * sum_{k1,k2} K[k1,k2]^4 * alpha2_K[k1,k2]

        Used to calibrate a targeted hyperdiffusion (see
        SDEs.sparse_G_advection) that damps grid-scale aliasing from the
        stochastic advection's direct cascade without affecting large scales.
        """
        factor = 1 
        return factor * np.sum(self.K**4 * self.alpha2_K) / (8 * self.N**(2*self.dim))

    def cfl_dt_max(self, beta):
        """
        Diffusive CFL bound on the time step, dt <= dx^2 / (beta * a0), for
        an SDE forward-integration diffusion coefficient scaled by beta(t)
        (see SDEs.MSGMsde.g). Returns the max stable dt for the given beta.
        """
        return self.dx**2 / (beta * self.a0())

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
        """
        Diffusive CFL condition (Resseguier et al. 2024 JFM eq. 4.7; PhD
        thesis appendix C.1: dt <= dx^2/(beta*a0)). If violated for the
        given (beta_max, dt), rescales alpha2_K (and hence G, since a0 is
        linear in alpha2_K) down just enough to meet it exactly -- this
        replaces the __init__-time a0-normalization (which only forces
        mean(alpha2_K)=1, unrelated to CFL/beta/dt). Mutates self.alpha2_K
        in place if violated; a no-op otherwise. Returns (rescale or None,
        a0_before, dt_max) for the caller to log.
        """
        a0_before = self.a0()
        dt_max = self.cfl_dt_max(beta_max)
        if dt > dt_max:
            target_a0 = self.dx**2 / (beta_max * dt)
            rescale = target_a0 / a0_before
            self.alpha2_K *= rescale
            return rescale, a0_before, dt_max
        return None, a0_before, dt_max

    def build_hyperdiffusion(self, z=1):
        """
        Targeted hyperdiffusion rate, diagonal in Fourier space, to
        stabilize the SPDE against the direct cascade / aliasing the
        stochastic advection creates at grid scale:
            D_PQ = coef_D * K(P)^(2z) * delta_PQ, coef_D = dx^(2z) * gamma0
                 = gamma0 * (dx*K(P))^(2z)
        (dx*K) is dimensionless, in [0, pi], giving a rate ~gamma0*pi^(2z)
        at the Nyquist/grid scale and ~0 at large scales. gamma0 =
        1/(characteristic time of gradient creation), eq. 4.8 of the same
        reference as a0() (eq. 4.7).

        Returns (D_hyper, gamma0), D_hyper flattened in 'F' order (row
        index p1, column index p2) to match the sparse advection tensor's
        flat-index convention I=p1+p2*N.
        """
        gamma0 = self.gamma0()
        D_hyper = gamma0 * (self.dx * self.K) ** (2 * z)   # (N,N), same (k1,k2) layout as self.K
        return D_hyper.flatten(order='F'), gamma0

    def build_sparse_G(self):
        """
        Builds the sparse advection tensor G[I,J,K] (flat index I=p1+p2*N,
        matching K=k1+k2*N and J=q1+q2*N the same way) from coef_G_KPQ.
        Each (p,q) pair contributes at both K=(q-p)%N and K=(p-q)%N (i.e.
        {k,-k} folded together), which makes each G_k slice exactly
        skew-symmetric (verified: max|Q^T G_k Q| ~ 1e-13 for random Q) --
        a plain single-delta shift matrix (nonzero only at q=p+k) can't be
        skew-symmetric on its own, since its (p,q) and (q,p) entries live
        on disjoint diagonals unless 2k=0.

        Returns (i_list, j_list, k_list, v_list), plain Python lists ready
        for torch.tensor construction (I,J,K as int64, V as float32).
        """
        N = self.N
        i_list, j_list, k_list, v_list = [], [], [], []
        for p1 in range(N):
            for p2 in range(N):
                for q1 in range(N):
                    for q2 in range(N):
                        k1 = (q1-p1) % N
                        k2 = (q2-p2) % N
                        K = k1 + k2*N
                        J = q1 + q2*N
                        I = p1 + p2*N
                        i_list.append(I); j_list.append(J); k_list.append(K)
                        v_list.append(self.coef_G_KPQ(k1, k2, p1, p2, q1, q2))

                        k1n, k2n = (N-k1) % N, (N-k2) % N
                        Kn = k1n + k2n*N
                        i_list.append(I); j_list.append(J); k_list.append(Kn)
                        v_list.append(self.coef_G_KPQ(k1n, k2n, p1, p2, q1, q2))
        return i_list, j_list, k_list, v_list

