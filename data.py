

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


class weather_station:
    def __init__(self, dim = 30):
        self.dim = dim
        self.name='weather'
        self.name = self.name + str(self.dim)

        pathData = '../MultiplicativeDiffusion/'
        folder = pathData + 'weather-data-2022-12-05-to-2023-02-27'
        file_names1 = [f'CR300-{i}_Mesures_0{i}' for i in range(14,16)]  
        file_names2 = [f'CR300-{i}_Mesures_{i}' for i in range(559,572)]  
        file_names = file_names1 + file_names2 
        begin_time = '"2022-12-07 00:00:00"'
        last_time = '"2023-02-26 23:55:00"'
        init = True
        for file in file_names:
            print('read file ' + file)
            file_path = folder + '/' + file + '.dat'
            # Load the data with mixed data types
            data = np.genfromtxt(
                file_path,
                delimiter=",",
                skip_header=4,  # Skip the first 4 lines of metadata
                dtype=None,  # Automatically infer types
                encoding="utf-8",  # Handle text encoding
                missing_values='"NAN"',  # Treat "NAN" as missing
                filling_values=np.nan,  # Fill missing values with NaN
            )
            # Convert to structured array
            timestamps = data['f0']  # First column (timestamps)
            npdata = np.array([list(row)[1:] for row in data], dtype=float)  # Exclude first column
            begin_indice = np.where(timestamps == begin_time)[0][0]
            last_indice = np.where(timestamps == last_time)[0][0]
            timestamps = timestamps[begin_indice:last_indice+1]
            npdata = npdata[begin_indice:last_indice+1,:]
            npdata = npdata[:,1:3] # keep only velocity
            # cartesiaan coordinate
            npdata0 = npdata.copy()
            npdata[:,0] = npdata0[:,0]*np.cos((np.pi/180.0)*npdata0[:,1])
            npdata[:,1] = npdata0[:,0]*np.sin((np.pi/180.0)*npdata0[:,1])
            # column_names = ['vx', 'vy', 'T']  # Generate column names like Col_1, Col_2, ...

            if init :
                timestamps_keep = timestamps  
                npdata_all = npdata.copy()
                init = False
            else:
                # Find common timestamps
                previous_timestamps_keep = timestamps_keep
                timestamps_keep = np.intersect1d(timestamps_keep, timestamps)

                # Extract synchronized data
                indices1 = np.where(np.isin(timestamps, timestamps_keep))[0]
                indices2 = np.where(np.isin(previous_timestamps_keep, timestamps_keep))[0]

                npdata = npdata[indices1,:]
                npdata_all = npdata_all[indices2,:]
                
                npdata_all = np.concatenate( (npdata_all,npdata.copy()), axis=1)

        npdata = npdata_all

        # center and mormalize data
        npdata = (npdata-npdata.mean(axis=0))/npdata.std(axis=0)
        # keep only dim dimension
        npdata = npdata[0:-1:1,0:self.dim]

        n_test = npdata.shape[0] // 3

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