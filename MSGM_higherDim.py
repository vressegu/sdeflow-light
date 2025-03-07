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
from data import ncar_weather_station,weather_station,eof_pressure,Lorenz96,PODmodes,SwissRoll


np.random.seed(0)
torch.manual_seed(0) 

DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

# arguments

# Train
T0 = 1
vtype = 'rademacher'
lr = 0.001
iterations = 10000
print_every = 1000

# Inference
num_stepss_backward = [10,100]
include_t0 = False
num_samples = 10000

# Dataset
dims = [30]
dims = [4]
Res=[300]
MSGMs = [0,1]
normalized_data = True

# Plots
scatter_plots = True
denoising_plots = True
save_results = True
plot_xlim = 3.0
plot_xlim = 2.0
plot_ylim_row = plot_xlim
plot_xlim_col = plot_xlim
ssize = 1

# Load results 
justLoad = False


if __name__ == '__main__':

    for MSGM in MSGMs:

        if not MSGM:
            normalized_data = True
        else:
            normalized_data = False
        for dim in dims:
            
            for Re in Res:
                ## 1. Initialize dataset
                sampler = SwissRoll()
                # sampler = PODmodes(Re,dim, normalized=normalized_data)
                # sampler = Lorenz96(Re,dim)
                # sampler = eof_pressure(dim)
                # sampler = weather_station()
                # sampler = weather_station(dim) 
                # sampler = ncar_weather_station()
                # sampler = ncar_weather_station(dim) 

                xtest = sampler.sampletest(num_samples).data.numpy()
                sampler.dim = xtest.shape[1]
                std_test = xtest.std(axis=0)

                plt.close('all')
                # # fig = plt.figure(figsize=(5, 5))
                # fig = plt.figure(figsize=(10, 10))
                dimplot = np.min([8,xtest.shape[1]])
                pddatatest = pd.DataFrame(xtest[:,0:dimplot], columns=range(1,1+dimplot))
                # pd.plotting.scatter_matrix(pddata, diagonal='kde',s=1,hist_kwds={"bins": 20},
                # color='red') 
                # dim = pddatatest.shape[1]
                fig, axes = plt.subplots(nrows=dimplot, ncols=dimplot, figsize=(2*dimplot,dimplot))
                color='blue'
                scatter = pd.plotting.scatter_matrix(pddatatest, diagonal=None,s=ssize,hist_kwds={"bins": 20},
                    color=color, ax=axes) 
                # Customize the diagonal manually
                for i, col in enumerate(pddatatest.columns):
                    ax = scatter[i, i]
                    ax.clear()
                    pddatatest[col].plot.kde(ax=ax, color=color, label='test')
                    ax.legend(fontsize=8, loc='upper right')
                    if not normalized_data:
                        plot_ylim_row = plot_xlim * std_test[i]
                    for j, col in enumerate(pddatatest.columns):
                        ax = scatter[i, j]
                        if not normalized_data:
                            plot_xlim_col = plot_xlim * std_test[j]
                        ax.axis(xmin=-plot_xlim_col,xmax=plot_xlim_col)
                        if (i != j):
                            ax.axis(ymin=-plot_ylim_row,ymax=plot_ylim_row)
                            # ax.set_aspect('equal', 'box')

                plt.tight_layout()
                # plt.show()

                # _ = plt.hist2d(x[:,0], x[:,1], 200, range=((-5,5), (-5,5)))
                # plt.axis('off')
                # plt.tight_layout()

                plt.show(block=False)    
                plt.pause(0.1)
                plt.savefig("results/" + sampler.name + ".png")
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
                    num_samples_ = x_test.size(0)
                    test_elbo = gen_sde.elbo_random_t_slice(x_test)
                    gen_sde.train()
                    return test_elbo.mean(), test_elbo.std() / num_samples_ ** 0.5

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

                # iterationss = [100000, 10000, 1000, 100, 10]
                # for iterations in iterationss:
                # batch_sizes = [256, 128, 64, 32, 16, 8, 4]
                batch_sizes = [256]

                # nosamples = np.linspace( n0, sampler.max_nsamples, sampler.max_nsamples/10)  1000 2000 3000.... 10000
                # if iter = k * 1000     (assume iterations = 10000)
                #  -> choose nosamples[k] samples
                # Technique to "generate" new sampler : boostrapping
                

                for batch_size in batch_sizes:

                    # init models
                    drift_q = MLP(input_dim=sampler.dim, index_dim=1, hidden_dim=128).to(device)
                    T = torch.nn.Parameter(torch.FloatTensor([T0]), requires_grad=False)
                    if MSGM:
                        x_init = sampler.sample(iterations*batch_size).data.numpy()
                        inf_sde = multiplicativeNoise(x_init,beta=1, T=T, num_steps_forward=num_steps_forward).to(device)
                    else:
                        inf_sde = VariancePreservingSDE(beta_min=1, beta_max=1, T=T, num_steps_forward=num_steps_forward).to(device)
                    gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=vtype, debias=False).to(device)

                    print("iterations = " + str(iterations) )
                    print("name_SDE = " + inf_sde.name_SDE )

                    if (not justLoad):
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
                    # num_stepss_backward = [1000, 100, 50, 20, 10, 5, 3, 2]

                    # num_stepss_backward = [2]
                    for num_steps_backward in num_stepss_backward:
                        print("Generation : num_steps_backward = " + str(num_steps_backward))
                        # init param
                        # num_samples = 100000

                        # lambdas
                        lmbds = [0.]
                        # lmbds = [0., 1.0]

                        # indices to visualize
                        fig_step = int(num_steps_backward/10) #100
                        if fig_step < 1:
                            fig_step = 1
                        if include_t0:
                            inds = range(0, num_steps_backward+1, fig_step)
                        else:
                            inds = range(fig_step-1, num_steps_backward, fig_step)
                        # sample and plot
                        plt.close('all')
                        for lmbd in lmbds:
                            folder_results = "results"
                            name_simu = folder_results + "/" + sampler.name + "_" \
                                + gen_sde.base_sde.name_SDE + "_" + str(iterations) + "iteLearning_" \
                                + str(batch_size) + "batchSize_" \
                                + str(num_steps_backward) + "stepsBack_lmbd=" + str(lmbd) 
                            if (justLoad):
                                save_results = False
                                xs = torch.load(name_simu + ".pt", weights_only=True)
                            else:
                                x_0 = gen_sde.latent_sample(num_samples, sampler.dim, device=device) # init from prior
                                xs = rk4_stratonovich_sampler(gen_sde, x_0, num_steps_backward, lmbd=lmbd,include_t0=include_t0) # sample
                            xgen = xs[-1]

                            if (scatter_plots):
                                pddatagen = pd.DataFrame(xgen[:,0:dimplot], columns=range(1,1+dimplot))

                                fig, axes = plt.subplots(nrows=dimplot, ncols=dimplot, figsize=(2*dimplot,dimplot))
                                color='red'
                                scatter = pd.plotting.scatter_matrix(pddatagen, diagonal=None,s=ssize,hist_kwds={"bins": 20},
                                    color=color, ax=axes) 
                                color='blue'
                                scatter = pd.plotting.scatter_matrix(pddatatest, diagonal=None,s=ssize/2,hist_kwds={"bins": 20},
                                    color=color, ax=axes) 
                                for i, col in enumerate(pddatatest.columns):
                                    ax = scatter[i, i]
                                    ax.clear()
                                    color='blue'
                                    pddatatest[col].plot.kde(ax=ax, color=color, label='test')
                                    if not normalized_data:
                                        plot_ylim_row = plot_xlim * std_test[i]
                                    for j, col in enumerate(pddatatest.columns):
                                        ax = scatter[i, j]
                                        if not normalized_data:
                                            plot_xlim_col = plot_xlim * std_test[j]
                                        ax.axis(xmin=-plot_xlim_col,xmax=plot_xlim_col)
                                        if (i != j):
                                            ax.axis(ymin=-plot_ylim_row,ymax=plot_ylim_row)
                                plt.tight_layout()
                                time.sleep(0.5)
                                plt.show(block=False)
                                plt.pause(1)
                                # Customize the diagonal manually
                                for i, col in enumerate(pddatatest.columns):
                                    ax = scatter[i, i]
                                    color='red'
                                    pddatagen[col].plot.kde(ax=ax, color=color, label='gen')
                                    ax.legend(fontsize=8, loc='upper right')
                                plt.tight_layout()
                                # plt.show()
                                time.sleep(0.5)
                                plt.show(block=False)
                                name_fig = name_simu + "_multDim.png" 
                                plt.savefig(name_fig)
                                plt.pause(1)
                                plt.close()

                            if (denoising_plots):
                                plot_selected_inds(xs, inds, True, False, lmbd, include_t0=include_t0) # plot
                                time.sleep(0.5)
                                plt.show(block=False)
                                name_fig = name_simu + ".png" 
                                plt.savefig(name_fig)
                                plt.pause(1)
                                plt.close()

                            if (save_results):
                                torch.save(xs, name_simu + ".pt")
