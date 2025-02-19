import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import dask
import xarray as xr
from pathlib import Path
import glob
from torch.distributions import Normal
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from losses import WeightedMSEKLD, WeightedMSE
from losses import WeightedMSEKLDLowRess, WeightedMSELowRess
from preprocessing import align_data_and_targets, create_mask, pole_centric, reverse_pole_centric, segment, reverse_segment, pad_xarray
from preprocessing import AnomaliesScaler_v1_seasonal, AnomaliesScaler_v2_seasonal, Standardizer, Normalizer, PreprocessingPipeline, calculate_climatology, bias_adj, zeros_mask_gen
from torch_datasets import XArrayDataset
import torch.nn as nn
# from subregions import subregions
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
import glob
import gc
# specify data directories
data_dir_forecast = LOC_FORECASTS_SI




def predict(fct:xr.DataArray , observation:xr.DataArray , params, lead_months, model_dir,  test_years, NPSProj  = False,  model_year = None, ensemble_list = None, ensemble_mode = 'Mean', btstrp_it = 200, save=True):

    if 'run_set_1' in model_dir:
        from models.cvae_0127 import cVAE
    else:
        from models.cvae import cVAE

    if 'Linear' in model_dir:
            params['VAE_MLP_encoder'] = True

    if model_year is None:
        model_year_ = np.min(test_years) - 1
    else:
        model_year_ = model_year
    try:
        scale_factor_channels = params['scale_factor_channels']
    except:
        scale_factor_channels = None

    params["obs_clim"] = False
    params['forecast_range_months'] = eval(model_dir.split('_F')[1].split('_')[0])
    if 'LT' in model_dir:
        lead_time = eval(model_dir.split('_LT')[1].split('_')[0])
    else:
        lead_time = None
    if 'combined' in model_dir:
        params['combined_prediction'] = True
    else:
        params['combined_prediction'] = False

    params['VAE_latent_size'] = eval(model_dir.split('_LS')[1].split('_')[0])
    if params['version'] == 2:

        params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1_seasonal())]
        params['observations_preprocessing_steps'] = []
        
    else:
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []


    print(f"Start run for test year {test_years}...")

    ############################################## load data ##################################
    ensemble_list = params['ensemble_list']
    ensemble_features = params['ensemble_features']
    time_features = params["time_features"]
    model = params['model']
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]
    try:
        ensemble_mode = params['ensemble_mode']
    except:
        params['ensemble_mode'] = 'Mean'

    try:
        obs_clim = params["obs_clim"]
    except: 
        pass
    decoder_kernel_size = params["decoder_kernel_size"]

    print("Load forecasts")
    if params['version'] == 3:
        ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/bias_adjusted_North_1983-2020_1x1.nc')['SICN']
        #####################################################################################################
        ### if you used Standardizer make sure to pass VAE = True as an argument to the initializer below ###
        if params['version'] == 3:
            print(' Warning!!! If you used Standardizer as a preprocessing step make sure to pass "VAE = True" as an argument to the initializer!!!')
        #####################################################################################################
    else:
        if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
            ds_in = fct.sel(ensembles = ensemble_list)['SICN']
                
        else:    ## Load specified members
            ds_in = fct['SICN']

        if params['ensemble_mode'] == 'Mean': ##
            ensemble_features = False
            ds_in = ds_in.mean('ensembles').load() ##
        else:
            ds_in = ds_in.load().transpose('time','lead_time','ensembles',...)
            print('Warning: ensemble_mode is None. Predicting for large ensemble ...')
        
    ###### handle nan and inf over land ############
    if not NPSProj:
        ds_in = ds_in.where(ds_in<1000,np.nan) ### land is masked in model data with a large number
    else:
        mask_projection = (xr.open_dataset(data_dir_obs)['mask'].rename({'x':'lon','y':'lat'}))
        observation = (observation.rename({'x':'lon','y':'lat'}))
        ds_in = (ds_in.rename({'x':'lon','y':'lat'}))
    land_mask = observation.mean('time').where(np.isnan(observation.mean('time')),1).fillna(0)
    model_mask = ds_in.mean('time')[0].where(np.isnan(ds_in.mean('time')[0]),1).fillna(0).drop('lead_time')
    observation = observation.clip(0,1)
    ds_in = ds_in.clip(0,1)
    observation = observation.fillna(0)
    ds_in = ds_in.fillna(0)
    ############################################
    obs_in = observation.expand_dims('channels', axis=1)

    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3)
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    min_year = np.min(test_years)*100
    max_year = (np.min(test_years) + 1 )*100 if len(test_years) <2 else (np.max(test_years) + 1)*100
    ds_in_ = ds_in.where((ds_in.time >= min_year)&(ds_in.time <= max_year) , drop = True).isel(lead_time = np.arange(0,lead_months ))
    obs_in_ = obs_in.where((obs_in.time >= min_year), drop = True)
    
    ds_raw, obs_raw = align_data_and_targets(ds_in.where(ds_in.time <= (model_year_ + 1)*100, drop = True), obs_in, lead_months)  # extract valid lead times and usable years ## used to be np.min(test_years)
    ds_raw_, obs_raw_ = align_data_and_targets(ds_in_, obs_in_, lead_months)  # extract valid lead times and usable years ## used to be np.min(test_years)
    if ds_raw_.time.max() < ds_in_.time.max():
        print(f'test_years truncated at {ds_raw_.time.max().values} due to unavailability of corresponding observation beyond that.')

    del ds_in, obs_in, obs_in_, ds_in_
    gc.collect()

    if not ds_raw.time.equals(obs_raw.time): 
            ds_raw = ds_raw.sel(time = obs_raw.time)
    if not ds_raw_.time.equals(obs_raw_.time): 
            ds_raw_ = ds_raw_.sel(time = obs_raw_.time)
    
    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time','ensembles',...)
        ds_raw_ = ds_raw_.transpose('time','lead_time','ensembles',...)
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time',...)
        ds_raw_ = ds_raw_.transpose('time','lead_time',...)
    

    train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time <= (model_year_ + 1)*100].to_numpy()    
    ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean,ds_raw_ ], dim = 'time')
    obs_raw = xr.concat([obs_raw,obs_raw_ ], dim = 'time')
    assert obs_raw.time.equals(ds_raw_ensemble_mean.time)

    subset_dimensions = params["subset_dimensions"]
    del ds_raw_,obs_raw_
    gc.collect()

    if all([subset_dimensions is not None, NPSProj is False]):
        if subset_dimensions == 'North':
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat > 40, drop = True)
            obs_raw = obs_raw.where(obs_raw.lat > 40, drop = True)
            land_mask = land_mask.where(land_mask.lat > 40, drop = True)
            model_mask = model_mask.where(model_mask.lat > 40, drop = True)
        else:
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat < -40, drop = True)
            obs_raw = obs_raw.where(obs_raw.lat < -40, drop = True)
            land_mask = land_mask.where(land_mask.lat < -40, drop = True)
            model_mask = model_mask.where(model_mask.lat < -40, drop = True)

    ################################### apply the mask #######################
    if params['model'] not in [cVAE]:
        # land_mask = land_mask.where(model_mask == 1, 0)
        obs_raw = obs_raw * land_mask
        ds_raw_ensemble_mean = ds_raw_ensemble_mean * land_mask
    
    if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
        zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(lead_time = 0).drop('lead_time').where(obs_raw.time<test_year*100, drop = True ), 3) for test_year in test_years], dim = 'test_year').assign_coords(test_year = test_years)           
        
        for item in ['active_mask', 'full_ice_mask']:
            zeros_mask_full = zeros_mask_full.drop(item) if item not in params["time_features"] else zeros_mask_full
        zeros_mask_full = zeros_mask_full.drop('active_grid') if not params['active_grid'] else zeros_mask_full

        zeros_mask_full = zeros_mask_full.expand_dims('channels', axis=-3)
        if 'ensembles' in ds_raw.dims:
             zeros_mask_full = zeros_mask_full.expand_dims('ensembles', axis=2)
             
    try:
        if params['obs_clim']:

                obs_clim_times = np.concatenate([np.unique(np.floor(ds_raw_ensemble_mean[:len(train_years)].time.values/100))[2:],  np.unique(np.floor(ds_raw_ensemble_mean[len(train_years):].time.values/100))])

                ls = []
                for ind, yr in enumerate(obs_clim_times):

                        ref_base = obs_raw.where(obs_raw.time < (yr) * 100, drop = True)
                        target_time = ds_raw_ensemble_mean.time[24 + 12*ind:12*ind+36]
                        mask = create_mask(ref_base)
                        mask = np.broadcast_to(mask[...,None,None,None], ref_base.shape)
                        ls.append(calculate_climatology(ref_base,mask ).rename({'init_month' : 'time'}).assign_coords(time = target_time))
                clim = xr.concat(ls, dim = 'time')
                if 'ensembles' in ds_raw_ensemble_mean.dims: 
                    clim = clim.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ########
                obs_raw = obs_raw[24:]
                ds_raw_ensemble_mean = ds_raw_ensemble_mean[24:]
                ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, clim], dim = 'channels')
                train_years = train_years[24:]
    except:
        pass

    if params['combined_prediction']:
        obs_raw_ = obs_raw.where(obs_raw>=0.15,0)
        obs_raw_ = obs_raw_.where(obs_raw_ ==0 , 1)
        obs_raw = xr.concat([obs_raw, obs_raw_], dim = 'channels')
        params['loss_function'] = 'combined'
        del obs_raw_

    if NPSProj is False:
            ds_raw_ensemble_mean = pole_centric(ds_raw_ensemble_mean, subset_dimensions)
            obs_raw =  pole_centric(obs_raw, subset_dimensions)
            land_mask = pole_centric(land_mask, subset_dimensions)
            model_mask = pole_centric(model_mask, subset_dimensions)
            if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                zeros_mask_full = pole_centric(zeros_mask_full, subset_dimensions)

    ###################################################################################
    if lead_time is not None:
        full_shape = xr.full_like(ds_raw_ensemble_mean, np.nan).isel(lat = slice(1,2), lon = slice(1,2))
        ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(lead_time = slice(lead_time,lead_time))
        obs_raw = obs_raw.sel(lead_time = slice(lead_time,lead_time))


    del ds_raw
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
        zeros_mask = zeros_mask_full.sel(test_year = model_year_ + 1).drop('test_year')
    else:
        zeros_mask = None
    
    n_train = len(train_years)
    train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...]) if lead_time is None else create_mask(full_shape[:n_train,...])[:, lead_time - 1][..., None] ############

    ds_baseline = ds_raw_ensemble_mean[:n_train,...]
    obs_baseline = obs_raw[:n_train,...].isel(channels = slice(0,1))


    if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)

    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)

    preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)


    # Data preprocessing
    
    ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
    ds = ds_pipeline.transform(ds_raw_ensemble_mean)

    obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
    # if 'standardize' in ds_pipeline.steps:
    #     obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
    obs = obs_pipeline.transform(obs_raw.isel(channels = slice(0,1)))

    if params['combined_prediction']:
            obs = xr.concat([obs, obs_raw.isel(channels = slice(1,2))], dim  = 'channels')  

    del ds_baseline, obs_baseline, preprocessing_mask_obs, preprocessing_mask_fct
    gc.collect()
    if params['version']  in [3]:
        sigmoid_activation = False
    else:
        sigmoid_activation = True

    y0 = np.floor(ds[:n_train].time[0].values/100 )
    yr, mn = np.divmod(int(ds[:n_train].time[-1].values - y0*100),100)
    month_min_max = [y0, yr * 12 + mn]
    if 'land_mask' in time_features:
            ds = xr.concat([ds, land_mask.expand_dims('channels', axis = 0)], dim = 'channels')

    # TRAIN MODEL
    ds_train = ds[:n_train,...]
    obs_train = obs[:n_train,...]
    ds_test = ds[n_train: ,...]
    obs_test = obs[n_train: ,...]
        
    if NPSProj:
        weights = (np.ones_like(ds_train.lon) * (np.ones_like(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon})
        weights = weights * land_mask
        # weights_ = weights * land_mask
    else:
        weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
        ########################################################################
        # weights_ = weights * land_mask
        if params['equal_weights']:
            weights = xr.ones_like(weights)
        # if any(['land_mask' not in time_features, model not in [UNet2]]):
        weights = weights * land_mask

    del ds, obs
    gc.collect()
    weights = weights.values
    if time_features is None:
        if ensemble_features: ## PG: We can choose to add an ensemble feature.
            add_feature_dim = 1
        else:
            add_feature_dim = 0
    else:
        if ensemble_features:
            add_feature_dim = len(time_features) + 1
        else:
            add_feature_dim = len(time_features)
    if 'land_mask' in time_features:
        add_feature_dim -= 1

    ########################################### load the model ######################################
    try:
        if params['obs_clim']:
            n_channels_x = len(ds_train.channels) + 1
        else:
            n_channels_x = len(ds_train.channels)
        
    except:
        pass
    

    net = cVAE(VAE_latent_size = params['VAE_latent_size'], n_channels_x= n_channels_x+ add_feature_dim , sigmoid = sigmoid_activation, NPS_proj = NPSProj, device=device, combined_prediction = params['combined_prediction'],scale_factor_channels = scale_factor_channels, VAE_MLP_encoder = params['VAE_MLP_encoder'])


    print('Loading model ....')
    net.load_state_dict(torch.load(glob.glob(model_dir + f'/*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 
    net.to(device)
    net.eval()
    ##################################################################################################################################

    test_years_list = np.arange(1, ds_test.shape[0] + 1)
    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
    test_set = XArrayDataset(ds_test, obs_test, lead_time=lead_time,mask = None,zeros_mask = zeros_mask, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max, model = 'UNet2')
                                      
    if lead_time is None:
        lead_times = ds_test.lead_time.values
    else:
        lead_times = [lead_time]

    test_loss = np.zeros(shape=(test_set.target.shape[0]))
    target_ens = xr.concat([test_set.target.expand_dims('ensembles', axis = 0).isel(channels = 0) for _ in range(params['BVAE'])], dim = 'ensembles')

    test_results = np.zeros_like(target_ens.values)
    results_shape = xr.full_like(target_ens, fill_value = np.nan)

    if params['save_deterministic'] :
        test_results_deterministic = np.zeros_like(test_set.target.values)
        results_shape_deterministic = xr.full_like(test_set.target, fill_value = np.nan)
    
    
    del target_ens
   
    test_time_list =  np.arange(1, ds_test.shape[0] + 1) ########?
    if params['combined_prediction']:
            test_results_extent = test_results.copy()
            results_shape_extent = results_shape.copy()
            if params['save_deterministic'] :
                test_results_deterministic_extent = test_results_deterministic.copy()
                results_shape_deterministic_extent = results_shape_deterministic.copy()

    if params['active_grid']:
        zeros_mask_test = results_shape.copy()
        zeros_mask_test[:] = test_set.zeros_mask
        zeros_mask_test = zeros_mask_test.unstack('flattened').transpose('time','lead_time',...)

    dataloader = DataLoader(test_set, batch_size=len(lead_times), shuffle=False)

    model_mask_ = torch.from_numpy(model_mask.to_numpy()).unsqueeze(0)#.expand(n_channels_x + add_feature_dim,*model_mask.shape) ## uncomment if multichannel true
    if 'run_set_1' in model_dir:
        model_mask_ = model_mask_.expand(n_channels_x + add_feature_dim,*model_mask.shape)
    
    obs_mask = torch.from_numpy(land_mask.to_numpy()).unsqueeze(0)

    for time_id, (x, target) in enumerate(dataloader): 
        with torch.no_grad():
            if (type(x) == list) or (type(x) == tuple):
                test_raw = (x[0].to(device), x[1].to(device))
                model_mask_ = model_mask_.to(test_raw[0])
            else:
                test_raw = x.to(device)
                model_mask_ = model_mask_.to(test_raw)

            if (type(target) == list) or (type(target) == tuple):
                test_obs, m = (target[0].to(device), target[1].to(device))
            else:
                test_obs = target.to(device)
                m = None

            obs_mask = obs_mask.to(test_obs)
            if 'run_set_1' in model_dir:
                obs_mask = obs_mask.expand_as(test_obs[0])

            _,deterministic_output, _, _, cond_mu, cond_log_var = net(test_obs, obs_mask, test_raw, model_mask_, sample_size = 1 )
            if not params['save_deterministic'] :
                del deterministic_output
            basic_unet = net.unet(test_raw, model_mask_)
            cond_var = torch.exp(cond_log_var) + 1e-4
            cond_std = torch.sqrt(cond_var)
            z =  Normal(cond_mu, cond_std * n_stds).rsample(sample_shape=(params['BVAE'],)).squeeze().to(device)
            z = torch.flatten(z, start_dim = 0, end_dim = 1)
            out = net.generation(z)
            del z
            out = torch.unflatten(out, dim = 0, sizes = (params['BVAE'],cond_var.shape[0]))
            out = out + basic_unet.squeeze() 
            out = torch.flatten(out, start_dim = 0, end_dim = 1)
            generated_output = net.last_conv(out)
            
            if params['combined_prediction']:
                    generated_output_extent = net.last_conv2(out)
                    try:
                        (deterministic_output, deterministic_output_extent) = deterministic_output
                    except:
                        pass
                    (test_obs, test_obs_extent) = (test_obs[:,0].unsqueeze(1), test_obs[:,1].unsqueeze(1))
                    generated_output_extent = torch.unflatten(generated_output_extent, dim = 0, sizes = (params['BVAE'],cond_var.shape[0]))
                    test_results_extent[:,time_id * len(lead_times) : (time_id+1) * len(lead_times),] = generated_output_extent.squeeze().to(torch.device('cpu')).numpy()
                    if params['save_deterministic'] :
                        test_results_deterministic_extent[time_id * len(lead_times) : (time_id+1) * len(lead_times),] = deterministic_output_extent.squeeze().to(torch.device('cpu')).numpy()
            del out                
            generated_output = torch.unflatten(generated_output, dim = 0, sizes = (params['BVAE'],cond_var.shape[0]))
            test_results[:,time_id * len(lead_times) : (time_id+1) * len(lead_times),] = generated_output.squeeze().to(torch.device('cpu')).numpy()
            if params['save_deterministic'] :
                test_results_deterministic[time_id * len(lead_times) : (time_id+1) * len(lead_times),] = deterministic_output.to(torch.device('cpu')).numpy()
                                      

    del  test_raw,  x, target, test_obs, test_set, ds_test , generated_output
    try:
        del  generated_output_extent, deterministic_output_extent
    except:
        pass
    if params['save_deterministic'] :
        del deterministic_output
    gc.collect()
    ###################################################### has to be eddited for large ensembles!! #####################################################################
    results_shape[:] = test_results[:]
    test_results = results_shape.unstack('flattened').transpose('time','lead_time',...)
    test_results_untransformed = obs_pipeline.inverse_transform(test_results.values)
    result = xr.DataArray(test_results_untransformed, test_results.coords, test_results.dims, name='nn_adjusted')

    if params['save_deterministic'] :
        results_shape_deterministic[:] = test_results_deterministic[:]
        test_results_deterministic = results_shape_deterministic.unstack('flattened').transpose('time','lead_time',...)
        test_results_untransformed_deterministic = obs_pipeline.inverse_transform(test_results_deterministic.values)
        result_deterministic = xr.DataArray(test_results_untransformed_deterministic, test_results_deterministic.coords, test_results_deterministic.dims, name='nn_adjusted')

    if params['combined_prediction']:
            results_shape_extent[:] = test_results_extent[:]
            result_extent = results_shape_extent.unstack('flattened').transpose('time','lead_time',...)
            if params['save_deterministic'] :
                results_shape_deterministic_extent[:] = test_results_deterministic_extent[:]
                result_extent_deterministic = results_shape_deterministic_extent.unstack('flattened').transpose('time','lead_time',...)


    if obs_clim:
        result = result.isel(channels = 0).expand_dims('channels', axis=2)

    del  results_shape, test_results, test_results_untransformed
    if params['save_deterministic'] :
        del results_shape_deterministic, test_results_deterministic, test_results_untransformed_deterministic
    try:
        test_results_extent, test_results_deterministic_extent
    except:
        pass
    gc.collect()

    result = (result * land_mask)
    if params['save_deterministic'] :
        result_deterministic = (result_deterministic * land_mask)

    if not NPSProj:

            result = reverse_pole_centric(result, subset_dimensions)
            if params['save_deterministic'] :
                result_deterministic = reverse_pole_centric(result_deterministic, subset_dimensions)

    result = result.to_dataset(name = 'nn_adjusted') 
    if params['save_deterministic'] :
        result_deterministic = result_deterministic.to_dataset(name = 'nn_adjusted')

    if params['active_grid']:
        if not NPSProj:
                zeros_mask_test = reverse_pole_centric(zeros_mask_test)
        else:
            zeros_mask_test = zeros_mask_test.rename({'lon':'x', 'lat':'y'})
        result = xr.combine_by_coords([result * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])
        if params['save_deterministic'] :
            result_deterministic = xr.combine_by_coords([result_deterministic * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])


    if params['combined_prediction']:
        result_extent = (result_extent * land_mask)
        if params['save_deterministic'] :
            result_deterministic_extent = (result_deterministic_extent * land_mask)
        if not NPSProj:
            result_extent = reverse_pole_centric(result_extent, subset_dimensions)
            if params['save_deterministic'] :
                result_deterministic_extent = reverse_pole_centric(result_deterministic_extent, subset_dimensions)

        result_extent = result_extent.to_dataset(name = 'nn_adjusted_extent')
        result_extent = result_extent.where(result_extent >= 0.5, 0)
        result_extent = result_extent.where(result_extent ==0, 1)
        result = xr.combine_by_coords([result , result_extent])
        if params['save_deterministic'] :
            result_deterministic_extent = result_deterministic_extent.to_dataset(name = 'nn_adjusted_extent')
            result_deterministic_extent = result_deterministic_extent.where(result_deterministic_extent >= 0.5, 0)
            result_deterministic_extent = result_deterministic_extent.where(result_deterministic_extent ==0, 1)
            result_deterministic = xr.combine_by_coords([result_deterministic , result_deterministic_extent])


    if model_year is not None:
        Path(model_dir + f'/{model_year}_model_predictions').mkdir(parents=True, exist_ok=True)
        model_dir = model_dir + f'/{model_year}_model_predictions'



    if np.min(test_years) != np.max(test_years):
        result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_ENS_{np.min(test_years)}-{np.max(test_years)}_nstds{n_stds}.nc', mode='w'))
        if params['save_deterministic'] :
            result_deterministic.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_deterministic_{np.min(test_years)}-{np.max(test_years)}.nc', mode='w'))

    else:
        result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_ENS_{np.min(test_years)}_nstds{n_stds}.nc', mode='w'))
        if params['save_deterministic'] :
            result_deterministic.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_deterministic_{np.min(test_years)}.nc', mode='w'))

    if params['save_deterministic'] :
        return result, result_deterministic
    else:
        return result


def extract_params(model_dir):
    params = {}
    path = glob.glob(model_dir + '/*.txt')[0]
    file = open(path)
    content=file.readlines()
    for line in content:
        key = line.split('\t')[0]
        try:
            value = line.split('\t')[1].split('\n')[0]
        except:
            value = line.split('\t')[1]
        try:    
            params[key] = eval(value)
        except:
            if key == 'ensemble_list':
                ls = []
                for item in value.split('[')[1].split(']')[0].split(' '):
                    try:
                        ls.append(eval(item))
                    except:
                        pass
                params[key] = ls
            else:
                params[key] = value
    return params

if __name__ == "__main__":

    ############################################## Set_up ############################################

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/cVAE/run_set_1_convnext'
    out_dir    = f'{out_dir_x}/N4_M12_F12_v1_B0.1_batch10_e50_CscaleNone_CGNhybrid_50-1_LS50_NPSproj_North_lr0.0001_batch10_e50_LNone' 


    params = extract_params(out_dir)
    print(f'loaded configuration: \n')
    for key, values in params.items():
        print(f'{key} : {values} \n')
    
    version = int(out_dir.split('/')[-1].split('_')[3][1])

    params["version"] = version
    print( f'Version: {version}')
    #################################################################################################

    lead_months = 12
    bootstrap = False
    test_years = np.arange(2018,2022)
    n_stds = 1
    params['BVAE'] = 50
    params['save_deterministic'] = False
    #################################################################################################
    obs_ref = out_dir_x.split('/')[-3]
    if '1x1' in out_dir:
        NPSProj = False
        crs = '1x1'  
    else:
        NPSProj = True
        crs = 'NPS'


    if obs_ref == 'NASA':
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ f'/NASA*{crs}*.nc')[0] 
    else:
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]
    
    observation = xr.open_dataset(data_dir_obs)['SICN']
    ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*{crs}*.nc')[0]) for intial_month in range(1,13) ]
    fct = xr.concat(ls, dim = 'time').sortby('time')

    del ls
    gc.collect()
    ##################################################################################################
    # for i in range(1,13):
    # out_dir    = f'{out_dir_x}/N4_M12_F12_v1_Banealing_batch10_e50_cVAE_50-1_LS50_NPSproj_North_lr0.001_batch10_e50_LNone'  


    predict(fct, observation, params, lead_months, out_dir,  test_years, NPSProj  = NPSProj, model_year = 2017, ensemble_mode='Mean',  save=True)

    print(f'Output dir: {out_dir}')
    print('Saved!')

