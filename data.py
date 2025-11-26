

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
from own_plotting import plots_vort

pathData = '../MSGM-data/'

class PIV:
    def __init__(self, dim = 2, normalized = False, 
                 localized = False, 
                 largeImage = False, 
                 smoothing = 0,
                 few_data = False, 
                 ntrain_max = np.inf):
        self.dim = dim
        self.name='PIV'
        self.name += str(self.dim)
        if largeImage:
            self.name += 'largeIm'
            if smoothing == 1:
                self.name += 'smooth'
            if smoothing == 2:
                self.name += 'superSmooth'
            localized = True
            npixelx = np.int32( np.sqrt(dim) )
        elif localized:
            self.name += 'loc'
        if few_data:
            # self.name = self.name + 'fewData'
            self.name += str(ntrain_max) + 'pts'
        if normalized:
            self.name += '_norm'
    
        folder_str = pathData
        if largeImage:
            folder_str += 'largerImage'
        else:
            folder_str += "newPIV"
            if localized:
                folder_str += '2'
        folder = Path(folder_str)
        prefix = "Serie_"

        if largeImage:
            npixelx_max = 64
        else:
            npixelx_max = 4
        dmax = 2*(npixelx_max**2)
        npdata = np.empty((dmax, 0))   # if not already

        print("Loading PIV data from folder:", folder)
        # if largeImage:
        #     file = folder_str + "/vortdivtot.npy"
        #     npdata = np.load(file)  
        # else:
        for file in sorted(folder.glob(prefix + "*_vortdiv.npy")):
            # print("Processing", file.name)
            dataPt = np.load(folder / f"{file.stem}.npy")  
            npdata = np.concatenate((npdata, dataPt.reshape(-1, 1)), axis=1)
            if any(np.isnan(dataPt.flatten())):
                print("Processing", file.name)
                print("data shape:", npdata.shape)
                print(dataPt)
        npdata = npdata.transpose() /2.5

        # center and mormalize data
        npdata = npdata-npdata.mean(axis=0)
        # keep only dim dimension
        if largeImage :
            if not (dim == npixelx**2):
                raise ValueError("Incorrect dim to subsample: {}".format(dim))
            npdata = npdata.reshape(([npdata.shape[0],npixelx_max,npixelx_max,2]),order='F')

            time_id = 0
            plots_vort(npdata[time_id,:,:,0])
            name_fig = "images/originalimageAtt" + str(time_id) + ".png" 
            plt.savefig(name_fig)
            plt.close()
            plt.close('all')

            npdata = npdata[:,:,:,0] # keeping only vorticity
            
            if smoothing>0:
                print("Filtered images")
                from scipy.ndimage import gaussian_filter
                if smoothing == 1:
                    sigmax = npdata.shape[1]//(3*npixelx)
                elif smoothing == 2:
                    sigmax = npdata.shape[1]//(npixelx)
                    npdata *= 4
                # sigmay = npdata.shape[2]//npixelx
                # print('npdata.shape = ', npdata.shape)
                for i in range(npdata.shape[0]):
                    npdata[i,:,:] = gaussian_filter(npdata[i,:,:], sigma=sigmax)
                    # npdata[i,:,:] = gaussian_filter(npdata[i,:,1], sigma=sigmay)
                    # print('npdata.shape = ', npdata.shape)
                plots_vort(npdata[time_id,:,:])
                name_fig = "images/smoothedimageAtt" + str(time_id) + ".png" 
                plt.savefig(name_fig)
                plt.close()
                plt.close('all')
            # else:
            print("Subsample images to match the required dimension")
            ix = np.linspace(0,npdata.shape[1]-1,npixelx,dtype=int)
            iy = np.linspace(0,npdata.shape[2]-1,npixelx,dtype=int)
            npdata = npdata[:,ix,:] # subsampling 
            npdata = npdata[:,:,iy] # subsampling 
            plots_vort(npdata[time_id,:,:])
            name_fig = "images/subsampleimageAtt" + str(time_id) + ".png" 
            plt.savefig(name_fig)
            plt.close()
            plt.close('all')

            npdata = npdata.reshape(([npdata.shape[0],dim]),order='F')
        else:
            npdata = npdata[:,0:self.dim]

        if few_data:
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
