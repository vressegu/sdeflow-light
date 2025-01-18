

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

class eof_pressure:
    def __init__(self, dim = 8):
        self.dim = dim
        self.name='eof_pressure_NA'
        self.name = self.name + str(self.dim)

        pathData = '../MultiplicativeDiffusion/'
        dataset = Dataset(pathData + 'pcs2.nc', 'r')
        # dataset = Dataset(pathData + 'pcs.nc', 'r')
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