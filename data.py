

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
from pathlib import Path
import random

class PIV:
    def __init__(self, dim = 2, normalized = False, localized = False, few_data = False, ntrain_max = np.inf):
        self.dim = dim
        self.name='PIV'
        self.name += str(self.dim)
        if localized:
            self.name += 'loc'
        if few_data:
            # self.name = self.name + 'fewData'
            self.name += str(ntrain_max) + 'pts'
        if normalized:
            self.name += '_norm'
    
        folder_str = "/Users/vresseiguier/Coding/MultiplicativeDiffusion/newPIV"
        if localized:
            folder_str += '2'
        folder = Path(folder_str)
        prefix = "Serie_"

        npdata = np.empty((32, 0))   # if not already

        print("Loading PIV data from folder:", folder)
        for file in sorted(folder.glob(prefix + "*_vortdiv.npy")):
            # print("Processing", file.name)
            dataPt = np.load(folder / f"{file.stem}.npy")  
            npdata = np.concatenate((npdata, dataPt.reshape(-1, 1)), axis=1)
            # print("data shape:", dataPt.shape)
            # print(dataPt)
            if any(np.isnan(dataPt.flatten())):
                print("Processing", file.name)
                print("data shape:", npdata.shape)
                print(dataPt)
        npdata = npdata.transpose() /2.5

        # center and mormalize data
        npdata = npdata-npdata.mean(axis=0)
        # keep only dim dimension
        npdata = npdata[:,0:self.dim]

        if few_data:
            # n_train = 1000
            n_train= min([2*npdata.shape[0]// 3, ntrain_max])
            n_test = npdata.shape[0] - n_train 
        else:
            n_test = npdata.shape[0] // 3

        self.npdata = npdata[0:-n_test:1,:]
        self.npdatatest = npdata[-n_test:-1:1,:]

        self.max_nsamples = self.npdata.shape[0]
        self.max_nsamplestest = self.npdatatest.shape[0]

        self.std = npdata.std(axis=0)
        if normalized:
            self.npdata = self.npdata/self.std
            self.npdatatest = self.npdatatest/self.std

    def sample(self, n):               
        idx = np.random.randint(0,self.npdata.shape[0], size = n) #% self.max_nsamples
        return torch.from_numpy(self.npdata[idx,:]).to(torch.float32)

    def sampletest(self, n):               
        idx = np.random.randint(0,self.npdatatest.shape[0], size = n) #% self.max_nsamples
        return torch.from_numpy(self.npdatatest[idx,:]).to(torch.float32)
    
    def get_std(self):
        return torch.from_numpy(self.std).to(torch.float32)


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

class Cauchy:
    """
    multi-dimensional Cauchy distribution sampler.
    """
    def __init__(self, dim = 2, correlation = False, normalized = False):
        self.dim = dim
        self.name='cauchy' + str(self.dim)
        if correlation:
            self.A = torch.randn(dim, dim)
            self.name = self.name + "cor"
        else:
            self.A = torch.eye(dim)
        cov = self.A @ self.A.T
        self.std = torch.sqrt(torch.diag(cov))
        if normalized:
            self.name = self.name + '_norm'
            self.A = torch.diag(1/self.std) @ self.A 
            cov = self.A @ self.A.T

        scale = (1.0/50)
        self.cauchy = torch.distributions.Cauchy(0.0, scale)

    def sample(self, n):
        return  self.cauchy.sample((n, self.dim)) @ self.A.T
    
    def sampletest(self, n):
        return self.sample(n)
    
    def get_std(self):
        return self.std
    

class Gaussian:
    """
    multi-dimensional Gaussian distribution sampler.
    """
    def __init__(self, dim = 2, correlation = True, normalized = False):
        self.dim = dim
        self.name='gaussian' + str(self.dim)
        if correlation:
            self.A = torch.randn(dim, dim)
            self.name = self.name + "cor"
        else:
            self.A = torch.eye(dim)
        cov = self.A @ self.A.T
        self.std = torch.sqrt(torch.diag(cov))
        if normalized:
            self.name = self.name + '_norm'
            self.A = torch.diag(1/self.std) @ self.A 
            cov = self.A @ self.A.T
        self.normal = torch.distributions.Normal(0.0, 1.0)

    def sample(self, n):
        return  self.normal.sample((n, self.dim)) @ self.A.T
    
    def sampletest(self, n):
        return self.sample(n)
    
    def get_std(self):
        return self.std
