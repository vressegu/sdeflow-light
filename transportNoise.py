# config.py
import numpy as np
from dataclasses import dataclass

class grid_k(): 
    def __init__(self, N=2, Lx=1, L_LS=1e3, rho_s=5/3):
    # def __init__(self, N=2, Lx=1, L_LS=1):

        self.rho_s: float = rho_s            # spatial spectrum slope of advecting velocity
    
        self.dim = 2
        self.N = N
        self.Lx = Lx
        self.L_LS = L_LS
        self.PN = N//2                                 # aliased wavevector index
        self.dx = self.Lx / N
        self.idx = np.linspace(0, self.N, endpoint=False)

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
            Kv = np.concat((KX, KY), axis= 2)
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
        
        # Spectrum of advecting velocity
        alpha2_K = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
        self.cleanAliasing(alpha2_K)
        a0_temp = np.sum(alpha2_K[:])/(self.N**self.dim) # discrete parseval theorem (?)
        alpha2_K /= a0_temp

        # dual grid
        KXq= np.transpose(KX,(2,3,0,1))
        KYq= np.transpose(KY,(2,3,0,1))
        print(np.shape(KX))
        print(np.shape(KXq))
        f_Kk_Kq = KX * KYq - KY * KXq
        print(np.shape(f_Kk_Kq))

        k1 = np.transpose(self.idx.copy(),(0,1,2,3))
        k2 = np.transpose(self.idx.copy(),(3,0,1,2))
        q1 = np.transpose(self.idx.copy(),(2,3,0,1))
        q2 = np.transpose(self.idx.copy(),(1,2,3,0))
        q1_k1 = (q1-k1)%self.N
        q2_k2 = (q2-k2)%self.N
        # dirac = (p1==q1m)&(p2==q2m)

        self.K = K
        self.alpha2_K = alpha2_K
        self.f_Kk_Kq = f_Kk_Kq
        self.q1_k1 = q1_k1
        self.q2_k2 = q2_k2

    def G_KPQ(self,P):
        # p1,p2,q1,q2 : vector 
        p1 = P%self.N
        p2 = P//self.N
        return np.reshape( self.G_k1k2p1p2q1q2(p1,p2), (self.N**2, self.N**2), order='F') 

    def G_k1k2p1p2q1q2(self,p1,p2):
        # p1,p2,q1,q2 : vector 
        return self.alpha2_K * self.f_Kk_Kq * self.delta_pqk(p1,p2)

    def delta_pqk(self,p1,p2):
        # p1,p2,q1,q2 : vector 
        # k1,k2 scalaire
        return ((self.q1_k1==p1)&(self.q2_k2==p2))

    # def delta_pqk(self,p1,p2,q1,q2,k1,k2):
    #     q1m = (q1-k1)%self.N
    #     q2m = (q2-k2)%self.N
    #     return ((p1==q1m)&(p2==q2m))
    
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

    def cleanAliasing_W(self,S1):
        S1[0, 0, 0] = 0
        S1[0, self.PN, :] = 0
        S1[0, :, self.PN] = 0

    def zeroSpatialField(self):
        if self.dim == 2 :
            zeroF = np.zeros((self.N, self.N))
        elif self.dim == 3 :
            zeroF = np.zeros((self.N, self.N, self.N))
        else:
            raise ValueError("Incorrect dim")
        return zeroF
    
    def whiteNoise(self, scalar = False):
        if self.dim == 2 :
            xi = np.random.randn(self.N, self.N)
        elif self.dim == 3 :
            xi = np.random.randn(self.N, self.N, self.N, 3*(not scalar) + scalar)
            if scalar:
                xi = xi[:,:,:,0,:]
        else:
            raise ValueError("Incorrect dim")
        return xi
