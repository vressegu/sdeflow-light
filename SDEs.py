# -*- coding: utf-8 -*-
"""Copy of sdeflow_equivalent_sdes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Tx_Yt90NRgHve--ocIXi6SGR-0ebwH0N
"""


import time
import numpy as np
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.mstats import mquantiles
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sde_scheme import euler_maruyama_sampler,heun_sampler,rk4_stratonovich_sampler

# init device
if torch.cuda.is_available():
    device = 'cuda'
    # print('use gpu\n')
elif torch.backends.mps.is_available():
    device = 'mps'
    # print('use mps\n')
else:
    device = 'cpu'
    # print('use cpu\n')

# class OU_SDE(torch.nn.Module): 
class forward_SDE(torch.nn.Module): 

    def __init__(self, base_sde, T):
        super().__init__()
        self.base_sde = base_sde
        self.T = T
    
    # Drift
    def mu(self, s, y, lmbd=0.):
        return self.base_sde.f(s, y) 
    
    # Stratonovich Drift
    def mu_Strato(self, t, y, lmbd=0.):
        return self.mu(t, y) - 0.5 * self.base_sde.div_Sigma(t, y)
    
    # Diffusion
    def sigma(self, s, y, lmbd=0.):
        return self.base_sde.g(s, y)

class SDE(torch.nn.Module):
    """
    parent class for SDE
    """
    # This class need to be changed since the forward SDE cannot be solved analitically
    def __init__(self, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.T = T
        self.t_epsilon = t_epsilon
        # self.forward_SDE = forward_SDE(self, self.T).to(device)

    def sample_scheme(self, t, y0, return_noise=False):
        """
        sample yt | y0
        """
        # num_steps_tot = 1000
        ## our_sde = forward_SDE(self, self.T).to(device)
        # y_allt = euler_maruyama_sampler(our_sde, y0, num_steps_tot, 0, True) # sample

        num_steps_tot = 100
        our_sde = forward_SDE(self, self.T).to(device)
        # y_allt = euler_maruyama_sampler(our_sde, y0, num_steps_tot, 0, True) # sample
        # y_allt = heun_sampler(our_sde, y0, num_steps_tot, 0, True) # sample
        y_allt = rk4_stratonovich_sampler(our_sde, y0, num_steps_tot, 0, True) # sample

        num_steps_floats = num_steps_tot * t/self.T
        num_steps_int = torch.trunc(num_steps_floats).to(torch.int)

        # WARNING : this sampler is used (many times) for t=T 
        # another method should be used here instead
        for k in range(y0.shape[0]):
            if t[k] >= self.T: 
                num_steps_int[k] = num_steps_tot - 1

        yt = torch.zeros_like(y0)
        for k in range(y0.shape[0]):
            yt[k,:] = y_allt[num_steps_int[k]][k,:]

        return yt
    
    def slow_sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        # mu = self.mean_weight(t) * y0
        # std = self.var(t) ** 0.5
        # epsilon = torch.randn_like(y0)
        # yt = epsilon * std + mu

        #y_0 = torch.randn(num_samples, 2, device=device) # init from prior
        dt =0.01

        # # print(t.shape())
        # print(y0)
        # print(type(y0))
        # print(y0.shape)
        # exit()
        num_steps_floats = t /dt 
        yt = torch.zeros_like(y0)
        #self.our_sde.T = t
        # TODO: Make this in parallel
        for k, discretization_info in enumerate(zip(t,num_steps_floats,y0)):  
            print("k = ", k)
            tt, ns, y0_k= discretization_info
            if tt < dt: 
                ns = 1
            else:
                ns = int(ns)
            our_sde = forward_SDE(self, tt).to(device)
            # self.forward_SDE.T = tt
            yt[k,:] = own_euler_maruyama_sampler(our_sde, y0_k, ns, 0,0)[-1] # sample
            

        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_Song_et_al(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        raise NotImplementedError('See the official repository.')
        # return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)


Log2PI = float(np.log(2 * np.pi))

def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z

class VariancePreservingSDE(SDE):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    # This class need to be changed since the forward SDE cannot be solved analitically
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__(T, t_epsilon)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.name_SDE = "VariancePreservingSDE"

    @property
    def logvar_mean_T(self):
        logvar = torch.zeros(1)
        mean = torch.zeros(1)
        return logvar, mean

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def div_Sigma(self, t, y):
        return torch.zeros_like(y)
        # return L_G * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        # return self.sample_scheme(t, y0)
        return self.sample_Song_et_al(t, y0, return_noise)

    def latent_sample(self,num_samples, n, device=device):
        # init from prior
        return torch.randn(num_samples, n, device=device) 
    
    def cond_latent_sample(self,t_, T, x):
        # conditionnal latent sample of yT knowing x=y0
        yT = self.sample(torch.ones_like(t_) * T, x)
        return yT
    
    def log_latent_pdf(self,yT):
        # log of latent pdf
        return log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT))
    

##########################################

def delta(i,j): 
    if i == j: 
        return 1.
    else: 
        return 0.
    
def G_k_e_l(alpha,F,n,k,l) : 
    return torch.tensor([alpha[k] * F[k,l] * delta(i, (l-k)%n)  for i in range(n)])

## temporary
def new_G(n) : 
    # from n independent random matrices 
    G = torch.zeros(n,n,n) 
    for k in range(n): 
        F = torch.randn(n,n)
        F = 0.5 * (F - F.T)
        G[:,:,k] = F
    
    # normalisation to control how fast the dynamic is
    L_G = 0.5*torch.einsum('ijk, jmk -> im', G, G)   # ito correction tensor
    tr_L = torch.trace(L_G)
    G = torch.sqrt( - 0.5 * n / tr_L ) * G
    
    # check
    validate = False
    if validate:
        print(tr_L)
        for l in range(n): 
            print("G[:,l,:] of rank d-1 ?")
            print(G[:,l,:])    
        for k in range(n): 
            print("G[:,:,k] skew sym ?")
            print(G[:,:,k])

    return G.to(device)

def new_G_physics(n) : 
    # Jacobian from advection term
    F = torch.randn(n,n)
    F = 0.5 * (F - F.T)
    # small-scale streamfunction spectrum
    alpha = torch.randn(n)
    alpha = torch.pow(alpha,2) 

    G = torch.zeros(n,n,n) 
    for k in range(n): 
        for l in range(n): 
            G[:,l,k] = G_k_e_l(alpha,F,n,k,l)
    
    # check
    validate = False
    if validate:
        for l in range(n): 
            print("G[:,l,:] of rank d-1 ?")
            print(G[:,l,:])
        for k in range(n): 
            print("G[:,:,k] skew sym ?")
            print(G[:,:,k])

    return G.to(device)

class multiplicativeNoise(SDE):
    """
    d Y = G(Y) o dB_t
    """
    # This class need to be changed since the forward SDE cannot be solved analitically
    # def __init__(self, n=2, G = new_G(2), T=1.0, t_epsilon=0.001):
    # def __init__(self, n=2, T=1.0, t_epsilon=0.001):
    def __init__(self, y0, beta=1.0, T=1.0, t_epsilon=0.001, plot_validate = False):
        super().__init__(T, t_epsilon)
        self.r_T = torch.linalg.norm(torch.tensor(y0), dim= 1)
        r_T = self.r_T.reshape(len(self.r_T),1)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.002).fit(r_T)
        self.dim = y0.shape[1]
        self.G = np.sqrt(beta) * new_G(self.dim)
        self.L_G = 0.5*torch.einsum('ijk, jmk -> im', self.G, self.G).to(device)   # ito correction tensor
        self.name_SDE = "multiplicativeNoise"

        if plot_validate :   
            beta_G = - 2*torch.trace(self.L_G)/self.dim            
            print("G")
            print(self.G[:,:,0])
            print(self.G[:,:,1])
            print("L_G")
            print(self.L_G)
            print("beta_G = " + str(beta_G))
                
            r_T = self.r_T.reshape(len(r_T),1)
            kde = KernelDensity(kernel='gaussian', bandwidth=0.002).fit(r_T)
            X_plot = torch.linspace(0,max(r_T[:,0]) + 0.1*abs(max(r_T[:,0])), 1000).reshape(1000,1)
            log_dens = torch.tensor(kde.score_samples(X_plot))
            plt.plot(X_plot[:,0], torch.exp(log_dens))
            plt.savefig("ecdf_n_kde.png")

    def f(self, t, y):
        # return 0.5 * div_Sigma(t, y)
        drift = torch.einsum('ij, bj -> bi', self.L_G, y)
        return drift

    def div_Sigma(self, t, y):
        drift = torch.einsum('ij, bj -> bi', 2*self.L_G, y)
        return drift

    def g(self, t, y):
        Gy = torch.einsum('ijk, bj -> bik', self.G, y)         # diffusion part 
        return Gy
    
    def sample(self, t, y0, return_noise=False):
        return self.sample_scheme(t, y0)

    def generate_uniform_on_sphere(self,num_samples): 
        # Let X_i be N(0,1) and  lambda^2 =2 sum X_i^2, then (X_1,...,X_d) / lambda  is uniform in S^{d-1}
        X = torch.randn(num_samples, self.dim).to(device)
        X_norm = torch.linalg.norm(X, dim = 1).reshape(num_samples,1)
        X =  X / X_norm 
        return X

    def gen_radial_distribution(self,num_samples): 
        Z = torch.randn(num_samples)
        U = norm.cdf(Z)    # map gaussian to uniform 
        r_gen = np.quantile(self.r_T, U).reshape(num_samples,1)

        validate = False
        if validate:
            r_plot = r_gen.clone().detach()
            plt.hist(r_plot, bins = 100, alpha = 0.5, density = True)
            time.sleep(0.5)
            plt.show(block=False)
            plt.savefig("radial_distribution.png")
            plt.pause(1)
            plt.close()

        return torch.tensor(r_gen).to(torch.float32).to(device)

    def latent_sample(self,num_samples, n, device=device):
        # init from prior
        r = self.gen_radial_distribution(num_samples)
        s = self.generate_uniform_on_sphere(num_samples)
        x0 = r * s 

        validate = False
        if validate:
            x0_plot = x0.clone().detach().cpu()
            if self.dim == 2: 
                plt.plot(x0_plot[:,0], x0_plot[:,1], 'or', markersize = 1, alpha = 0.5)
                plt.gca().set_aspect('equal', 'box')
            elif self.dim == 3: 
                #ax = plt.figure().add_subplot(projection='3d')
                ax = plt.axes(projection='3d')
                x_gen_plot = np.array(x0_plot)
                ax.plot3D(x_gen_plot[:,0], x_gen_plot[:,1], x_gen_plot[:,2], 'or', markersize = 1, alpha = 0.5)
                plt.gca().set_aspect('equal', 'box')
            else: 
                raise NotImplemented("dim = 2 and 3 supported")
            plt.show()
            plt.savefig("latent_sample_multNoise.png")
            plt.close()

        return x0
        
    def cond_latent_sample(self,t_, T, x):
        # conditionnal latent sample of yT knowing x=y0
        r_x = torch.linalg.norm(x.clone().detach(), dim= 1).reshape(x.shape[0],1)
        s = self.generate_uniform_on_sphere(x.shape[0])
        yT =  r_x * s
        return yT.to(torch.float32).to(device)
    
    def log_latent_pdf(self,yT):
        r_T = self.r_T.reshape(len(self.r_T),1)
        X_plot = torch.linspace(0,max(r_T[:,0]) + 0.1*abs(max(r_T[:,0])), 1000).reshape(1000,1)
        r_yT = torch.linalg.norm(yT.clone().detach(), dim= 1)
        r_yT = r_yT.reshape(len(r_yT),1)
        log_dens_yT = torch.tensor(self.kde.score_samples(r_yT.cpu())) - np.log(2*np.pi)
        return log_dens_yT.to(torch.float32).to(device)

###################################################################################################

### Reverse SDE
def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1

def sample_gaussian(shape):
    return torch.randn(*shape)

def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')

class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """
    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, lmbd=0.):
        return self.ga_m_drift(self.T-t, y, lmbd)

    # Drift of reserve generative SDE
    def ga_m_drift(self, s, y, lmbd=0.):
        return (1. - 0.5 * lmbd) * self.ga(s, y) - self.base_sde.f(s, y) + (1. - lmbd) * self.base_sde.div_Sigma(s, y)
    
    def ga(self, s, y):
        if len(self.base_sde.g(s, y).shape)>2 :
            ga = torch.einsum('bij, bj -> bi', self.base_sde.g(s, y), self.a(y, s.squeeze()))
        else :
            ga = self.base_sde.g(s, y) * self.a(y, s.squeeze())
        return ga

    # Stratonovich Drift
    def mu_Strato(self, t, y, lmbd=0.):
        return self.mu(t, y, lmbd) - 0.5 * (1. - lmbd) * self.base_sde.div_Sigma(self.T-t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    # # WARNING : DSM is not relevant in MSGM
    # # SSM needs to be defined instead
    # @torch.enable_grad()
    # def dsm(self, x):
    #     """
    #     denoising score matching loss
    #     """
    #     if self.debias:
    #         t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
    #     else:
    #         t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
    #     y, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
    #     a = self.a(y, t_.squeeze())

    #     return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2
    #     # / g is not convenient for g being a dense matrix of rank d-1...

    @torch.enable_grad()
    def ssm(self, x):
        """
        estimating the SSM loss of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        # Is self.debias case needed as in DSM ???
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze())

        # OU case
        # mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)
        # mu_to_div = mu

        # General case
        mu = self.ga_m_drift(t_, y, 0.)
        mu_to_div = mu - 0.5 * self.base_sde.div_Sigma(t_, y)

        # Simpler and faster way for MSGM
        #mu_to_div = self.ga(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        mMu = (
            torch.autograd.grad(mu_to_div, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False)

        mNu = (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

        return mMu + mNu

    @torch.enable_grad()
    def elbo_random_t_slice(self, x):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        yT = self.cond_latent_sample(t_, self.base_sde.T, x)
        lp = self.base_sde.log_latent_pdf(yT).view(x.size(0), -1).sum(1)

        return lp - self.ssm(x) / qt
    
    def latent_sample(self,num_samples, n, device=device):
        # init from prior
        return self.base_sde.latent_sample(num_samples, n, device) 
    
    def cond_latent_sample(self,t_, T, x):
        # conditionnal latent sample of yT knowing x=y0
        return self.base_sde.cond_latent_sample(t_, T, x)
