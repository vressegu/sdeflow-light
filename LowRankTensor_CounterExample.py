
"""
Simulation SDE simple :
U = I_d
X0 = (1,1,1,1)
lambda1 = 1, lambda2 = 10
X_t(j) = exp(i * sqrt(4) * lambda_j * B_t) * X0_j
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Paramètres
# -------------------------------------------------------------
d = 4
T = 100.0
N = 20000
dt = T/N
times = np.linspace(0, T, N+1)
scale_fig = 0.5

# valeurs propres (paires ±)
lambda1 = 1.0
lambda2 = 10.0
Lambda = np.array([lambda1, -lambda1, lambda2, -lambda2])

# X0 = np.array([1,1,1,1], dtype=complex)
sqrt_d = np.sqrt(d)

# Mouvement brownien
np.random.seed(0)
dW = np.sqrt(dt) * np.random.randn(N)
B = np.empty(N+1)
B[0] = 0.0
B[1:] = np.cumsum(dW)


N = len(B)-1  # nombre d'itérations
x1 = [0]*(N+1)
x2 = [0]*(N+1)
x3 = [0]*(N+1)
x4 = [0]*(N+1)

X0 = np.array([1,1,1,1], dtype=float)
for k in range(N+1):
    x1[k]=X0[1]*np.sin(sqrt_d*lambda1*B[k])+X0[0]*np.cos(sqrt_d*lambda1*B[k])
    x2[k]=X0[1]*np.cos(sqrt_d*lambda1*B[k])-X0[0]*np.sin(sqrt_d*lambda1*B[k])
    
    x3[k]=X0[3]*np.sin(sqrt_d*lambda2*B[k])+X0[2]*np.cos(sqrt_d*lambda2*B[k])
    x4[k]=X0[3]*np.cos(sqrt_d*lambda2*B[k])-X0[2]*np.sin(sqrt_d*lambda2*B[k])
    

# ----- Scatter X1 vs X2 -----
plt.figure(figsize=(5*scale_fig,5*scale_fig))
plt.scatter(x1, x2, s=2, c ='Black', alpha=0.5)
# plt.title("Projection dans le plan (Re X1, Re X2)")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.grid(True)
plt.savefig('x1_x2.png')

# ----- Scatter X1 vs X2 -----
plt.figure(figsize=(5*scale_fig,5*scale_fig))
plt.scatter(x1, x3, s=2, c ='Black', alpha=0.5)
# plt.title("Projection dans le plan (Re X1, Re X2)")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_3$")
plt.grid(True)
plt.savefig('x1_x3.png')

 


fig = plt.figure(figsize=(8*scale_fig,6*scale_fig))
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(x1, x2, x3, c='black' )  

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
plt.savefig('x1_x2_x3.png')
plt.show()


