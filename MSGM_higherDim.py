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
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from netCDF4 import Dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sde_scheme import euler_maruyama_sampler,heun_sampler,rk4_stratonovich_sampler
from own_plotting import plot_selected_inds
from SDEs import VariancePreservingSDE,PluginReverseSDE,multiplicativeNoise


np.random.seed(0)
DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

if __name__ == '__main__':

    ## 1. Initialize dataset
    
    class eof_pressure:
        def __init__(self, dim = 8):
            self.dim = dim
            self.name='eof_pressure_NA'
            self.name = self.name + str(self.dim)

            pathData = '../MultiplicativeDiffusion/'
            dataset = Dataset(pathData + 'pcs.nc', 'r')
            pseudo_pcs = dataset.variables['pseudo_pcs'][:] # var can be 'Theta', 'S', 'V', 'U' etc..
            npdata = np.array(pseudo_pcs)

            npdata = npdata[0:-1:1,0:self.dim]/250000
            n_test = npdata.shape[0] // 3


            # npdata = npdata/10
            # npdatatest = npdatatest/10
            self.npdata = npdata[0:-n_test:1,:]
            self.npdatatest = npdata[-n_test:-1:1,:]

            self.max_nsamples = self.npdata.shape[0]
            self.max_nsamplestest = self.npdatatest.shape[0]

        def sample(self, n):               
            idx = np.random.randint(0,self.npdata.shape[0], size = n) #% self.max_nsamples
            return torch.from_numpy(self.npdata[idx,:]).to(torch.float32)

        def sampletest(self, n):               
            idx = np.random.randint(0,self.npdatatest.shape[0], size = n) #% self.max_nsamples
            return torch.from_numpy(self.npdatatest[idx,:]).to(torch.float32)
    
    class Lorenz96:
        def __init__(self, n_dim_L96 = 100, dim = 8):
            self.dim = dim
            # n_dim_L96 = 100
            # # n_dim_L96 = 4
            self.name='L96ß_n' + str(n_dim_L96)
            # Re='300'
            # Re = str(Re)
            self.name = self.name + str(self.dim)

            pathData = '../MultiplicativeDiffusion/'

            # npdata = np.load('./L96_n' + str(n_dim_L96) + '_data.npy')
            pathData = pathData + './L96_n' + str(n_dim_L96) + '_data'
            npdata = np.load(pathData + '.npy')
            npdatatest = np.load(pathData + '_test.npy')
            npdata = npdata/10
            npdatatest = npdatatest/10
            self.npdata = npdata[:,0:self.dim]
            self.npdatatest = npdatatest[:,0:self.dim]

            self.max_nsamples = npdata.shape[0]
            self.max_nsamplestest = npdatatest.shape[0]

        def sample(self, n):               
            idx = np.random.randint(0,self.npdata.shape[0], size = n) #% self.max_nsamples
            return torch.from_numpy(self.npdata[idx,:]).to(torch.float32)

        def sampletest(self, n):               
            idx = np.random.randint(0,self.npdatatest.shape[0], size = n) #% self.max_nsamples
            return torch.from_numpy(self.npdatatest[idx,:]).to(torch.float32)

    class PODmodes:
        def __init__(self, Re = 300, dim = 8):
            self.dim = dim
            # self.dim = 16
            self.name='POD'
            # Re='300'
            Re = str(Re)
            self.name = self.name + Re + str(self.dim)

            pathData = '../MultiplicativeDiffusion/'
            pathData = pathData + 'tempPODModes/LES_Re' + Re + '/temporalModes_16modes'
            # pathData = pathData + 'tempPODModes/LES_Re3900/temporalModes_16modes'
            npdata = np.load(pathData + '/U.npy')
            npdatatest = np.load(pathData + '_test/U.npy')
            npdata = npdata/10
            npdatatest = npdatatest/10
            self.npdata = npdata[:,0:self.dim]
            self.npdatatest = npdatatest[:,0:self.dim]

            self.max_nsamples = npdata.shape[0]
            self.max_nsamplestest = npdatatest.shape[0]

        def sample(self, n):               
            idx = np.random.randint(0,self.npdata.shape[0], size = n) #% self.max_nsamples
            return torch.from_numpy(self.npdata[idx,:]).to(torch.float32)

        def sampletest(self, n):               
            idx = np.random.randint(0,self.npdatatest.shape[0], size = n) #% self.max_nsamples
            return torch.from_numpy(self.npdatatest[idx,:]).to(torch.float32)

    class SwissRoll:
        """
        Swiss roll distribution sampler.
        noise control the amount of noise injected to make a thicker swiss roll
        """
        def __init__(self): 
            self.dim = 2
            self.name='swiss'
        def sample(self, n, noise=0.5):
            if noise is None:
                noise = 0.5
            return torch.from_numpy(
                make_swiss_roll(n, noise=noise)[0][:, [0, 2]].astype('float32') / 5.) # Changed: Pass noise as a keyword argument
        
        def sampletest(self, n, noise=0.5):
            return self.sample(n, noise)

    # sampler = SwissRoll()
    dims = [2,4,8,16]
    # dims = [8]
    # Res=[300,3900]
    Res=[100,1000,10000]
    # Res=[300]
    for dim in dims:
        
        for Re in Res:
            # sampler = PODmodes(Re,dim)
            sampler = Lorenz96(Re,dim)
            # sampler = eof_pressure(dim)
            # num_samples = 100000
            num_samples = 10000
            xtest = sampler.sampletest(num_samples).data.numpy()
            
            plt.close('all')
            # # fig = plt.figure(figsize=(5, 5))
            # fig = plt.figure(figsize=(10, 10))

            pddatatest = pd.DataFrame(xtest, columns=range(1,1+xtest.shape[1]))
            # pd.plotting.scatter_matrix(pddata, diagonal='kde',s=1,hist_kwds={"bins": 20},
            # color='red') 
            dim = pddatatest.shape[1]
            fig, axes = plt.subplots(nrows=dim, ncols=dim, figsize=(2*dim,dim))
            color='blue'
            ssize = 2
            scatter = pd.plotting.scatter_matrix(pddatatest, diagonal=None,s=ssize,hist_kwds={"bins": 20},
                color=color, ax=axes) 
            # Customize the diagonal manually
            for i, col in enumerate(pddatatest.columns):
                ax = scatter[i, i]
                ax.clear()
                pddatatest[col].plot.kde(ax=ax, color=color, label='test')
                ax.legend(fontsize=8, loc='upper right')

            plt.tight_layout()
            # plt.show()

            # _ = plt.hist2d(x[:,0], x[:,1], 200, range=((-5,5), (-5,5)))
            # plt.axis('off')
            # plt.tight_layout()

            plt.show(block=False)    
            plt.pause(0.1)
            plt.savefig(sampler.name + ".png")
            plt.pause(0.1)
            plt.close()
            

            ## 2. Define models

            ### 2.2. Define MLP
            class Swish(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return torch.sigmoid(x)*x

            class MLP(nn.Module):
                def __init__(self,
                            input_dim=2,
                            index_dim=1,
                            hidden_dim=128,
                            act=Swish(),
                            ):
                    super().__init__()
                    self.input_dim = input_dim
                    self.index_dim = index_dim
                    self.hidden_dim = hidden_dim
                    self.act = act

                    self.main = nn.Sequential(
                        nn.Linear(input_dim+index_dim, hidden_dim),
                        act,
                        nn.Linear(hidden_dim, hidden_dim),
                        act,
                        nn.Linear(hidden_dim, hidden_dim),
                        act,
                        nn.Linear(hidden_dim, input_dim),
                        )

                def forward(self, input, t):
                    # init
                    sz = input.size()
                    input = input.view(-1, self.input_dim)
                    t = t.view(-1, self.index_dim).float()

                    # forward
                    h = torch.cat([input, t], dim=1) # concat
                    output = self.main(h) # forward
                    return output.view(*sz)

            ### 2.3. Define evaluate function (compute ELBO)
            @torch.no_grad()
            def evaluate(gen_sde, x_test):
                gen_sde.eval()
                num_samples = x_test.size(0)
                test_elbo = gen_sde.elbo_random_t_slice(x_test)
                gen_sde.train()
                return test_elbo.mean(), test_elbo.std() / num_samples ** 0.5

            ## 3. Train
            # init device
            if torch.cuda.is_available():
                device = 'cuda'
                print('use gpu\n')
            elif torch.backends.mps.is_available():
                device = 'mps'
                print('use mps\n')
            else:
                device = 'cpu'
                print('use cpu\n')

            # iterations = 10000
            # iterationss = [100000, 10000, 1000, 100, 10]
            iterationss = [10000]
            # for iterations in iterationss:
            # batch_sizes = [256, 128, 64, 32, 16, 8, 4]
            batch_sizes = [256]

            # nosamples = np.linspace( n0, sampler.max_nsamples, sampler.max_nsamples/10)  1000 2000 3000.... 10000
            # if iter = k * 1000     (assume iterations = 10000)
            #  -> choose nosamples[k] samples
            # Technique to "generate" new sampler : boostrapping
            

            for batch_size in batch_sizes:

                # arguments
                T0 = 1
                vtype = 'rademacher'
                lr = 0.001
                # batch_size = 256
                # #iterations = 100000
                iterations = 10000
                # iterations = 1
                # print_every = 50
                print_every = 1000

                # init models
                # drift_q = MLP(input_dim=2, index_dim=1, hidden_dim=128).to(device)
                drift_q = MLP(input_dim=sampler.dim, index_dim=1, hidden_dim=128).to(device)
                T = torch.nn.Parameter(torch.FloatTensor([T0]), requires_grad=False)
                x_init = sampler.sample(iterations*batch_size).data.numpy()
                inf_sde = multiplicativeNoise(x_init,beta=1, T=T).to(device)
                # inf_sde = VariancePreservingSDE(beta_min=1, beta_max=1, T=T).to(device)
                gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=vtype, debias=False).to(device)

                print("iterations = " + str(iterations) )
                print("name_SDE = " + inf_sde.name_SDE )

                # init optimizer
                optim = torch.optim.Adam(gen_sde.parameters(), lr=lr)

                # train
                start_time = time.time()
                for i in range(iterations):
                    optim.zero_grad() # init optimizer
                    x = sampler.sample(batch_size).to(device) # sample data
                    loss = gen_sde.ssm(x).mean() # forward and compute loss
                    loss.backward() # backward
                    optim.step() # update

                    # print
                    if (i == 0) or ((i+1) % print_every == 0):
                        # elbo
                        elbo, elbo_std = evaluate(gen_sde, x)

                        # print
                        elapsed = time.time() - start_time
                        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} | elbo {:8.3f} | elbo std {:8.3f} '
                            .format(i+1, elapsed*1000/print_every, loss.item(), elbo.item(), elbo_std.item()))
                        start_time = time.time()


                ## 4. Visualize

                ### 4.3. Simulate SDEs
                """
                Simulate the generative SDE by using RK4 method
                """
                # num_stepss = [1000, 100, 50, 20, 10, 5, 3, 2]
                num_stepss = [1000, 100, 50]
                # num_stepss = [2]
                for num_steps in num_stepss:
                    print("Generation : num_steps = " + str(num_steps))
                    # init param
                    # num_samples = 100000

                    # lambdas
                    lmbds = [0.]
                    # lmbds = [0., 1.0]

                    # indices to visualize
                    num_figs = 10
                    if num_figs > num_steps:
                        num_figs = num_steps
                    fig_step = int(num_steps/50) #100
                    if fig_step < 1:
                        fig_step = 1
                    inds = [i-1 for i in range(num_steps-(num_figs-1)*fig_step, num_steps+1, fig_step)]

                    # sample and plot
                    plt.close('all')
                    for lmbd in lmbds:
                        # x_0 = gen_sde.latent_sample(num_samples, 2, device=device) # init from prior
                        x_0 = gen_sde.latent_sample(num_samples, x.shape[1], device=device) # init from prior
                        # xs = euler_maruyama_sampler(gen_sde, x_0, num_steps, lmbd=lmbd) # sample
                        # xs = heun_sampler(gen_sde, x_0, num_steps, lmbd=lmbd) # sample
                        xs = rk4_stratonovich_sampler(gen_sde, x_0, num_steps, lmbd=lmbd) # sample

                        # pddata = pd.DataFrame(npdata, columns=['A', 'B', 'C', 'D'])
                        pddatagen = pd.DataFrame(xs[num_steps-1], columns=range(1,1+x.shape[1]))

                        # pddatatest = pd.DataFrame(xtest, columns=range(1,1+xtest.shape[1]))
                        # # pd.plotting.scatter_matrix(pddata, diagonal='kde',s=1,hist_kwds={"bins": 20},
                        # # color='red') 
                        dim = pddatatest.shape[1]
                        fig, axes = plt.subplots(nrows=dim, ncols=dim, figsize=(2*dim,dim))
                        color='blue'
                        scatter = pd.plotting.scatter_matrix(pddatatest, diagonal=None,s=ssize,hist_kwds={"bins": 20},
                            color=color, ax=axes) 
                        color='red'
                        scatter = pd.plotting.scatter_matrix(pddatagen, diagonal=None,s=ssize/2,hist_kwds={"bins": 20},
                            color=color, ax=axes) 
                        # Customize the diagonal manually
                        for i, col in enumerate(pddatatest.columns):
                            ax = scatter[i, i]
                            ax.clear()
                            color='blue'
                            pddatatest[col].plot.kde(ax=ax, color=color, label='test')
                            color='red'
                            pddatagen[col].plot.kde(ax=ax, color=color, label='gen')
                            ax.legend(fontsize=8, loc='upper right')

                        # pd.plotting.scatter_matrix(pddatagen, diagonal='kde',s=1,hist_kwds={"bins": 20})
                        # # pd.plotting.scatter_matrix(pddata, diagonal="kde")

                        plt.tight_layout()
                        # plt.show()
                        time.sleep(0.5)
                        plt.show(block=False)
                        name_fig = sampler.name + "_" \
                            + gen_sde.base_sde.name_SDE + "_" + str(iterations) + "iteLearning_" \
                            + str(batch_size) + "batchSize_" \
                            + str(num_steps) + "stepsBack_lmbd=" + str(lmbd) + "_multDim.png" 
                        plt.savefig(name_fig)
                        plt.pause(1)
                        plt.close()

                        # plot_selected_inds(xs[:,0:1,:], inds, True, True, lmbd) # plot

                        # time.sleep(0.5)
                        # plt.show(block=False)
                        # name_fig = sampler.name + "_" \
                        #     + gen_sde.base_sde.name_SDE + "_" + str(iterations) + "iteLearning_" \
                        #     + str(batch_size) + "batchSize_" \
                        #     + str(num_steps) + "stepsBack_lmbd=" + str(lmbd) + ".png" 
                        # plt.savefig(name_fig)
                        # plt.pause(1)
                        # plt.close()
