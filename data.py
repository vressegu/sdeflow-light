

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

class ERA5:
    def __init__(self, dim = 40, \
                variables = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "vorticity"],
                # cities = ["Paris", "London", "Berlin", "Madrid", "Rome", "Vienna", "Amsterdam", "Stockholm", "Athens", "Warsaw"], # old order
                cities= ["Paris",  "Warsaw", "Berlin", "Vienna", "Amsterdam", "Stockholm", "Athens", "London","Madrid", "Rome"], # new order
                season: str = "all",            # "all" or "winter" (DJF)
                start_date: str = "2010-01-01T00",
                end_date: str   = "2020-12-31T18",
                use_deseason: bool = False,     # remove annual (DOY) + diurnal cycles BEFORE winter filtering
                bool_check_plot: bool = False,  # plots for one city/variable at three stages
                plot_city: str = "Berlin",
                plot_variable: str = "2m_temperature",
                mixedTimes = False,
                ):
        self.dim = dim
        self.name='ERA5'
        if len(variables)*len(cities)<self.dim:
            self.dim = len(variables)*len(cities)
        if len(variables)<4:
            self.name = self.name + str(len(variables)) + 'vars'
        if len(cities)<10:
            self.name = self.name + str(len(cities)) + 'cities'
        self.name = self.name + str(self.dim)
        if use_deseason:
            self.name = self.name + "_deseason"
        if season == "winter":
            self.name = self.name + "_DJF"
        if mixedTimes:
            self.name += 'mix'

        folder = os.path.join(pathData, 'ERA5-cities')

        # ------------- Load all city/variable series -------------
        city_data = {}
        for city in cities:
            city_data[city] = {}
            print(city)
            for var in variables:
                filename = os.path.join(folder, f"{city}_{var}_2010_2020.npy")
                print(filename)
                if os.path.exists(filename):
                    arr = np.load(filename)
                    # if vorticity has a second axis (levels), take the first level
                    if var == "vorticity" and arr.ndim == 2 and arr.shape[1] > 1:
                        # arr = arr[:, 1]
                        arr = arr[:, 0]
                    # scale (your choices)
                    if var == "vorticity":
                        arr = arr / 0.00003
                    elif var == "10m_u_component_of_wind":
                        arr = arr / 3.0
                    elif var == "10m_v_component_of_wind":
                        arr = arr / 3.0
                    elif var == "2m_temperature":
                        arr = arr / 7.0
                    city_data[city][var] = arr.astype(np.float64, copy=False)
                else:
                    print(f"⚠️ Warning: File {filename} not found!")

        # Determine timeline length from first loaded series
        num_timesteps = next(iter(city_data["Paris"].values())).shape[0]

        # Build 6-hourly time vector from start_date
        t0 = np.datetime64(start_date)
        times = np.arange(t0, t0 + np.timedelta64(num_timesteps * 6, 'h'), np.timedelta64(6, 'h'))

        # --------- Step A: mask invalid times (NaNs in vorticity across cities) ---------
        valid_mask = np.ones(num_timesteps, dtype=bool)
        if ("vorticity" in variables):
            for city in cities:
                vorticity_values = city_data[city]["vorticity"]
                valid_mask &= ~np.isnan(vorticity_values)
        print(f"NaN/vorticity mask keeps {valid_mask.sum()} / {num_timesteps} steps")

        # Apply NaN mask to series (for all vars) and to times
        for city in cities:
            for var in variables:
                city_data[city][var] = city_data[city][var][valid_mask]
        times = times[valid_mask]
        num_timesteps = valid_mask.sum()

        # --------- Pack into array [city, var, time] then reshape to (T, V*C) ----------
        T = num_timesteps
        V = len(variables)
        C = len(cities)
        data_array = np.zeros((C, V, T), dtype=np.float64)
        for i, city in enumerate(cities):
            for j, var in enumerate(variables):
                data_array[i, j, :] = city_data[city][var]

        # Build (T, V*C) consistent with Fortran-order stacking (var-major inside city-major)
        X_TVC = np.transpose(data_array, (2, 1, 0))   # (T, V, C)
        X = np.reshape(X_TVC, (T, V * C), order='F')  # (T, V*C)

        # --------- Step B: remove annual + diurnal cycles over the full year (if requested) ---------
        if use_deseason:
            X = self._deseasonalize_seasonal_diurnal(X, times)

        # SAVE the full (pre-winter) timeline for plotting & debug
        times_full = times.copy()  

        # --------- Step C: only after deseasonalization, optionally select winter (DJF) ----------
        if season == "winter":
            months = (times.astype('datetime64[M]').astype(int) % 12) + 1  # 1..12
            winter_mask = (months == 12) | (months == 1) | (months == 2)
            X = X[winter_mask, :]
            times = times[winter_mask]
            T = X.shape[0]
            print(f"Winter filter keeps {T} steps")

        # --------- Optional che# --------- Optional check plots for one city/variable at 3 stages ----------
        if bool_check_plot:
            var_index = {v: j for j, v in enumerate(variables)}
            city_index = {c: k for k, c in enumerate(cities)}
            k = city_index.get(plot_city, 0)
            j = var_index.get(plot_variable, 0)

            # raw (after NaN mask, before deseason & before winter)
            series_raw = data_array[k, j, :]  # length matches times_full

            if use_deseason:
                # compute intermediate series on the full pre-winter timeline
                s_raw = series_raw.reshape(-1, 1)
                s_ann_removed, s_full_removed = self._deseasonalize_debug_series(s_raw, times_full)
                series_ann = s_ann_removed[:, 0]
                series_full = s_full_removed[:, 0]
            else:
                series_ann = series_raw.copy()
                series_full = series_raw.copy()

            # If winter selected, **subset for display** using a mask built on times_full
            if season == "winter":
                months_all = (times_full.astype('datetime64[M]').astype(int) % 12) + 1
                djf_mask = (months_all == 12) | (months_all == 1) | (months_all == 2)

                series_raw_plot  = series_raw[djf_mask]
                series_ann_plot  = series_ann[djf_mask]
                series_full_plot = series_full[djf_mask]
                times_plot       = times_full[djf_mask]
            else:
                series_raw_plot  = series_raw
                series_ann_plot  = series_ann
                series_full_plot = series_full
                times_plot       = times_full

            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(times_plot, series_raw_plot);  axes[0].set_title(f"{plot_city} – {plot_variable} (raw)")
            axes[1].plot(times_plot, series_ann_plot);  axes[1].set_title("After removing annual seasonal cycle (DOY mean)")
            axes[2].plot(times_plot, series_full_plot); axes[2].set_title("After removing diurnal cycle (hourly mean)")
            for ax in axes: ax.grid(True, alpha=0.3)
            fig.autofmt_xdate(); plt.tight_layout()

            # Save before show (safer), then show
            folder_results = "results"
            directory = os.path.join(folder_results, self.name)
            os.makedirs(directory, exist_ok=True)
            fig.savefig(os.path.join(directory, "deseasonality.png"), dpi=150)
            plt.show()



        # --------- Center columns, keep dim, split train/test ----------
        X = X - X.mean(axis=0)
        X = X[:, :self.dim]
        n_test = X.shape[0] // 3

        if mixedTimes:
            idx = random.sample(range(X.shape[0]),n_test)
            noidx = [i for i in range(X.shape[0]) if i not in idx]
            self.npdatatest = X[idx,:]
            self.npdata = X[noidx,:]
        else:
            self.npdata     = X[:-n_test, :]
            self.npdatatest = X[-n_test:-1, :]

        self.max_nsamples     = self.npdata.shape[0]
        self.max_nsamplestest = self.npdatatest.shape[0]

    def sample(self, n):
        idx = np.random.randint(0, self.npdata.shape[0], size=n)
        return torch.from_numpy(self.npdata[idx, :]).to(torch.float32)

    def sampletest(self, n):
        idx = np.random.randint(0, self.npdatatest.shape[0], size=n)
        return torch.from_numpy(self.npdatatest[idx, :]).to(torch.float32)

    @staticmethod
    def _deseasonalize_seasonal_diurnal(X, times):
        """
        Remove annual (day-of-year mean) and diurnal (hour-of-day mean) cycles.
        X: (T, F), times: (T,) datetime64 array.
        Returns: X after removing both cycles.
        """
        ts = pd.to_datetime(times)
        df = pd.DataFrame(index=ts, data=X)

        # Annual cycle (by Day-Of-Year)
        doy = ts.dayofyear
        trend_doy = df.groupby(doy).mean()                    # (<=366, F)
        X_ann = df.values - trend_doy.reindex(doy).values     # (T, F)

        # Diurnal cycle (by hour of day)
        df_ann = pd.DataFrame(index=ts, data=X_ann)
        hours = ts.hour
        trend_hour = df_ann.groupby(hours).mean()             # (<=24, F)
        X_full = X_ann - trend_hour.reindex(hours).values     # (T, F)
        return X_full

    @staticmethod
    def _deseasonalize_debug_series(s, times):
        """
        Return intermediate series for plotting: after removing
        (1) annual DOY mean, and (2) diurnal mean.
        s: (T,1), times: (T,)
        """
        ts = pd.to_datetime(times)
        df = pd.DataFrame(index=ts, data=s)

        # Annual
        doy = ts.dayofyear
        trend_doy = df.groupby(doy).mean()
        s_ann = df.values - trend_doy.reindex(doy).values

        # Diurnal
        df_ann = pd.DataFrame(index=ts, data=s_ann)
        hours = ts.hour
        trend_hour = df_ann.groupby(hours).mean()
        s_full = s_ann - trend_hour.reindex(hours).values
        return s_ann, s_full

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


class ncar_weather_station:
    def __init__(self, dim = 90):
        self.dim = dim
        self.name='ncar_weather'
        self.name = self.name + str(self.dim)

        folder = pathData + 'isfs_m2hats_qc_geo_hr_202309'
        file = 'subsample_data'
        file_path = folder + '/' + file + '.npy'
        npdata = np.load(file_path) 

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


class weather_station:
    def __init__(self, dim = 30):
        self.dim = dim
        self.name='weather'
        self.name = self.name + str(self.dim)

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
    def __init__(self, n_dim_L96 = 100, dim = 8, normalized = False):
        self.dim = dim
        # n_dim_L96 = 100
        # # n_dim_L96 = 4
        self.name='L96ß_n' + str(n_dim_L96)
        # Re='300'
        # Re = str(Re)
        self.name = self.name + str(self.dim)

        pathData = pathData + './L96_n' + str(n_dim_L96) + '_data'
        npdata = np.load(pathData + '.npy')
        npdatatest = np.load(pathData + '_test.npy')
        npdata = npdata/10
        npdatatest = npdatatest/10

        if normalized:
            std = npdata.std(axis=0)
            npdata = npdata/std
            npdatatest = npdatatest/std

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

def load_POD_data(Re):
    pathData = pathData + 'tempPODModes/LES_Re' + str(Re) + '/temporalModes_16modes'
    npdata = np.load(pathData + '/U.npy')
    npdatatest = np.load(pathData + '_test/U.npy')
    return npdata, npdatatest

class PODmodes:
    def __init__(self, Re = 300, dim = 8, normalized = False, mixedTimes = False, concatenateRe = False, few_data = False, ntrain_max = np.inf):
        self.dim = dim
        # self.dim = 16
        self.name='POD'
        # Re='300'
        if concatenateRe:
            Re = '300-3900'
        else:
            Re = str(Re)
        self.name = self.name + Re + str(self.dim)
        if few_data:
            mixedTimes = True
            self.name += str(ntrain_max) + 'pts'
        if mixedTimes:
            self.name += 'mix'
        if normalized:
            self.name = self.name + '_norm'

        if concatenateRe:
            Re1 = 300
            Re2 = 3900
            npdata1, npdatatest1 = load_POD_data(Re1)
            npdata2, npdatatest2 = load_POD_data(Re2)
            npdata = np.concatenate(( npdata1, (Re2/Re1) * npdata2), axis=0)
            npdatatest = np.concatenate(( npdata1, (Re2/Re1) * npdata2), axis=0)
        else:
            npdata, npdatatest = load_POD_data(int(Re))

        if mixedTimes:
            npdataall = np.concatenate((npdata, npdatatest), axis=0)
        else:
            npdataall = npdata

        if few_data:
            n_train= min([2*npdataall.shape[0]// 3, ntrain_max])
            n_test = npdataall.shape[0] - n_train 
        else:
            n_test = npdataall.shape[0] // 3

        if mixedTimes:
            idx = random.sample(range(npdataall.shape[0]),n_test)
            noidx = [i for i in range(npdataall.shape[0]) if i not in idx]
            npdatatest = npdataall[idx,:]
            npdata = npdataall[noidx,:]
            
        npdata = npdata/10
        npdatatest = npdatatest/10

        npdata = npdata[:,0:self.dim]
        npdatatest = npdatatest[:,0:self.dim]

        self.std = npdata.std(axis=0)
        self.mean = npdata.mean(axis=0)
        print(self.mean/self.std)
        if normalized:
            npdata = npdata/self.std
            npdatatest = npdatatest/self.std

        self.npdata = npdata
        self.npdatatest = npdatatest

        self.max_nsamples = npdata.shape[0]
        self.max_nsamplestest = npdatatest.shape[0]
        print("max nb train samples = " + str(self.max_nsamples))
        print("max nb test samples = " + str(self.max_nsamplestest))

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
    
class GaussianCauchy:
    """
    multi-dimensional Gaussian distribution scaled with one-dim Cauchy.
    """
    def __init__(self, dim = 2, correlation = True, normalized = False):

        self.gaussian = Gaussian(dim, correlation, normalized)
        self.cauchy = torch.distributions.Cauchy(0.0, 1.0)
        self.dim = dim
        self.name='gaussianCauchy' + str(self.dim)
        if correlation:
            self.name = self.name + "cor"
        if normalized:
            self.name = self.name + '_norm'

    def get_std(self):
        return self.gaussian.std

    def sample(self, n):
        return (1.0/50) * self.gaussian.sample(n) * self.cauchy.sample((1, 1))
    
    def sampletest(self, n):
        return self.sample(n)
