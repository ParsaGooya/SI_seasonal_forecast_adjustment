import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.unet import UNet, UNetLCL,UNet_NPS
from models.unetconvnext import UNet2,UNet2_NPS
from models.convlstm import UNetLSTM, PNet, CNNLSTM, CNNLSTM_monthly, UNetLSTM_monthly
from models.cnn import CNN, RegCNN
from losses import WeightedMSE, WeightedMSEGlobalLoss
from losses import WeightedMSELowRess , WeightedMSEGlobalLossLowRess, GlobalLoss, IceextentlLoss
from preprocessing import align_data_and_targets, create_mask, config_grid, pole_centric, reverse_pole_centric, segment, reverse_segment, pad_xarray
from preprocessing import AnomaliesScaler_v1_seasonal, AnomaliesScaler_v2_seasonal, Standardizer, PreprocessingPipeline, calculate_climatology,bias_adj, zeros_mask_gen, Normalizer
from torch_datasets import XArrayDataset, ConvLSTMDataset
import torch.nn as nn
# from subregions import subregions
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
import glob
import gc
# specify data directories
data_dir_forecast = LOC_FORECASTS_SI

def HP_congif(params, obs_ref, lead_months, y_start, y_end, NPSProj = False):
    if lead_time is not None:
        assert lead_time <=lead_months, f"{lead_time} can not be greater than {lead_months}"
        if 'UNetLSTM' in params["model"].__name__:
            params['model'] = UNetLSTM_monthly
            print('Lead_time is on, using UNetLSTM_monthly ...')
        if 'CNNLSTM' in params["model"].__name__:
            params['model'] = CNNLSTM_monthly
            print('Lead_time is on, using CNNLSTM_monthly ...')

    if params['model'] in [UNet,UNetLCL, UNetLSTM, PNet,UNet2, UNetLSTM_monthly]:
        params['kernel_size'] = None
        params['decoder_kernel_size'] = None
        params['hidden_dims'] = None
        params['DSC'] = False

    if params['model'] not in [UNet, UNet2, UNetLCL]:
        params['skip_conv'] = False
    
    if NPSProj:
        crs = 'NPS'  
        if params["model"] == UNet:
            params['model'] = UNet_NPS
        if params["model"] == UNet2:
            params['model'] = UNet2_NPS
    else: 
        crs = '1x1'

    if obs_ref == 'NASA':
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ f'/NASA*{crs}*.nc')[0] 
    else:
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

    print("Start training")
    print("Load observations")
    obs_in = xr.open_dataset(data_dir_obs)['SICN']
    
    if params['version'] == 3:

        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
        ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/global_mean_bias_adjusted_1983-2020.nc')['SICN']
        if params['ensemble_list'] is not None:
            raise RuntimeError('With version 3 you are reading the bias adjusted ensemble mean as input. Set ensemble_list to None to proceed.')

    else:
        if params['ensemble_list'] is not None: ## PG: calculate the mean if ensemble mean is none
            print("Load forecasts")
            ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*{crs}*.nc')[0])['SICN'] for intial_month in range(1,13) ]
            ds_in = xr.concat(ls, dim = 'time').sortby('time').sel(ensembles = params['ensemble_list'])
            if params['ensemble_mode'] == 'Mean': 
                ds_in = ds_in.mean('ensembles') 
            else:
                ds_in = ds_in.transpose('time','lead_time','ensembles',...)
                print(f'Warning: ensemble_mode is {params["ensemble_mode"]}. Training for large ensemble ...')

        else:    ## Load specified members
            print("Load forecasts") 
            ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*{crs}*.nc')[0])['SICN'].mean('ensembles') for intial_month in range(1,13) ]
            ds_in = xr.concat(ls, dim = 'time').sortby('time')
    del ls
    gc.collect()
    ###### handle nan and inf over land ############
     ### land is masked in model data with a large number
    if not NPSProj:
        ds_in = ds_in.where(ds_in<1000,np.nan)
    else:
        mask_projection = (xr.open_dataset(data_dir_obs)['mask'].rename({'x':'lon','y':'lat'}))
        obs_in = (obs_in.rename({'x':'lon','y':'lat'}))
        ds_in = (ds_in.rename({'x':'lon','y':'lat'}))


    land_mask = obs_in.mean('time').where(np.isnan(obs_in.mean('time')),1).fillna(0)
    model_mask = ds_in.mean('time')[0].where(np.isnan(ds_in.mean('time')[0]),1).fillna(0).drop('lead_time')
    obs_in = obs_in.clip(0,1)
    ds_in = ds_in.clip(0,1)
    obs_in = obs_in.fillna(0)
    ds_in = ds_in.fillna(0)
    ############################################
    
    obs_in = obs_in.expand_dims('channels', axis=1)

    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3)
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_months)  # extract valid lead times and usable years
    del ds_in, obs_in
    gc.collect()

    if not ds_raw.time.equals(obs_raw.time):
            
            ds_raw = ds_raw.sel(time = obs_raw.time)
    
    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time','ensembles',...)
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time',...)

    if all([params["subset_dimensions"] is not None, NPSProj is False]):
        if params['subset_dimensions'] == 'North':
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
    if params['model'] not in [UNet2, UNet2_NPS]:
        # land_mask = land_mask.where(model_mask == 1, 0)
        obs_raw = obs_raw * land_mask
        ds_raw_ensemble_mean = ds_raw_ensemble_mean * land_mask
    ################################### apply the mask #######################
    
    zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(lead_time = 0).drop('lead_time').where(obs_raw.time<(test_year-params['num_val_years'])*100, drop = True ),3) for test_year in range(y_start,y_end+1)], dim = 'test_year').assign_coords(test_year = range(y_start,y_end+1))           
    zeros_mask_full = zeros_mask_full.expand_dims('channels', axis=-3)          
    if 'ensembles' in ds_raw.dims:
        zeros_mask_full = zeros_mask_full.expand_dims('ensembles', axis=2)
    
    obs_clim = params["obs_clim"]

    if obs_clim:
            
            ls = []
            for yr in np.unique(np.floor(ds_raw_ensemble_mean.time.values/100))[2:]:
        
                    ref  = obs_raw.where(obs_raw.time < (yr+1) * 100, drop = True)
                    mask = create_mask(ref[:-12])
                    mask = np.broadcast_to(mask[...,None,None,None], ref[:-12].shape)
                    ls.append(calculate_climatology(ref[:-12],mask ).rename({'init_month' : 'time'}).assign_coords(time = ref[-12:].time))
            clim = xr.concat(ls, dim = 'time')
            if 'ensembles' in ds_raw_ensemble_mean.dims: 
                clim = clim.expand_dims(ensembles = ds_raw_ensemble_mean['ensembles'], axis = 2) ########
            obs_raw = obs_raw.sel(time = clim.time)
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(time = clim.time)
            ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean, clim], dim = 'channels')

    if params['version'] == 'IceExtent':
        obs_raw = obs_raw.where(obs_raw>=0.15,0)
        obs_raw = obs_raw.where(obs_raw ==0 , 1)
        # ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean>=0.15,0)
        # ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean ==0 , 1)
        
    if params['combined_prediction']:
        obs_raw_ = obs_raw.where(obs_raw>=0.15,0)
        obs_raw_ = obs_raw_.where(obs_raw_ ==0 , 1)
        obs_raw = xr.concat([obs_raw, obs_raw_], dim = 'channels')
        params['loss_function'] = 'combined'
        del obs_raw_

    if NPSProj is False:
        if params['model'] in [RegCNN, CNNLSTM]:
            ds_raw_ensemble_mean = segment(ds_raw_ensemble_mean,  9)
            obs_raw =  segment(obs_raw,  9)
            land_mask = segment(land_mask,9)
            model_mask = segment(model_mask,9)
            if any(['active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                raise RuntimeError('You cannot add mask data in chennels with RegCNN model where you have regions stacked as channels!')
            zeros_mask_full = segment(zeros_mask_full, 9)

        else: #[UNet,UNetLCL, CNN,UNet2, UNetLSTM, UNetLSTM_monthly, PNet]:
            ds_raw_ensemble_mean = pole_centric(ds_raw_ensemble_mean, params['subset_dimensions'])
            obs_raw =  pole_centric(obs_raw, params['subset_dimensions'])
            zeros_mask_full = pole_centric(zeros_mask_full, params['subset_dimensions'])
            land_mask = pole_centric(land_mask, params['subset_dimensions'])
            model_mask = pole_centric(model_mask, params['subset_dimensions'])

    del ds_raw
    gc.collect()
    params['NPSProj'] = NPSProj
    
    if NPSProj:
        return ds_raw_ensemble_mean, obs_raw, params, zeros_mask_full, (land_mask, model_mask, mask_projection)
    else:
        return ds_raw_ensemble_mean, obs_raw, params, zeros_mask_full, (land_mask, model_mask)


def smooth_curve(list, factor = 0.9):
    smoothed_list = []
    for point in list:
        if smoothed_list:
            previous = smoothed_list[-1]
            smoothed_list.append(previous* factor + point * (1- factor))
        else:
            smoothed_list.append(point)
    return smoothed_list



def training_hp(hyperparamater_grid: dict, params:dict, ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset ,zeros_mask_full:XArrayDataset, land_masks:XArrayDataset  ,test_year, lead_time = None,  n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None):
    model_mask= land_masks[1]
    land_mask = land_masks[0]
    if params['NPSProj']:
        mask_projection = land_masks[2]

    assert params['version'] in [1,2,3, 'IceExtent']

    

    if params['version'] == 2:

        params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1_seasonal())]
        params['observations_preprocessing_steps'] = []  
    else:
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []

    if params['version'] == 'IceExtent':
        params['reg_scale'] = None

    for key, value in hyperparamater_grid.items():
            params[key] = value 

    if params['lr_scheduler']:
        start_factor = params['start_factor']
        end_factor = params['end_factor']
        total_iters = params['total_iters']

        
    if params['low_ress_loss']:
        params['active_grid'] = False
        print('Warning: active_grid turned off because low_ress_loss is on!')

    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    ensemble_features = params['ensemble_features']

    reg_scale = params["reg_scale"]
    model = params["model"]
    hidden_dims = params["hidden_dims"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    kernel_size = params["kernel_size"]
    decoder_kernel_size = params["decoder_kernel_size"]
    optimizer = params["optimizer"]
    lr = params["lr"]
    l2_reg = params['L2_reg']
    active_grid = params['active_grid']
    low_ress_loss = params['low_ress_loss']
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    loss_region = params["loss_region"]
    subset_dimensions = params["subset_dimensions"]
    num_val_years = params['num_val_years']


    test_years = test_year

    if lead_time is not None:
        full_shape = xr.full_like(ds_raw_ensemble_mean, np.nan).isel(lat = slice(1,2), lon = slice(1,2))
        ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(lead_time = slice(lead_time,lead_time))
        obs_raw = obs_raw.sel(lead_time = slice(lead_time,lead_time))
    ###################################################################################
    if n_runs > 1:
        numpy_seed = None
        torch_seed = None



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Start run with {params['model']} for test year {test_year} ...")
    if time_features is None:
        time_features = []
    if any([params['active_grid'],'active_mask' in time_features, 'full_ice_mask' in time_features]):
        for item in ['active_mask', 'full_ice_mask']:
            zeros_mask_full = zeros_mask_full.drop(item) if item not in time_features else zeros_mask_full
        zeros_mask_full = zeros_mask_full.drop('active_grid') if not time_features else zeros_mask_full
        zeros_mask = zeros_mask_full.sel(test_year = test_year).drop('test_year')
    else:
        zeros_mask = None
      

    train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time < (test_year-num_val_years) * 100].to_numpy()
    # validation_years = ds_raw_ensemble_mean.year[(ds_raw_ensemble_mean.year >= test_year-3)&(ds_raw_ensemble_mean.year < test_year)].to_numpy()
    
    n_train = len(train_years)
    train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...]) if lead_time is None else create_mask(full_shape[:n_train,...])[:, lead_time - 1][..., None] ############

    ds_baseline = ds_raw_ensemble_mean[:n_train,...]
    obs_baseline = obs_raw[:n_train,...].isel(channels = slice(0,1))

    if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
    else:
        preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)
    preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)


    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

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
    yr, mn = np.divmod(int(ds[:n_train+12*num_val_years].time[-1].values - y0*100),100)
    month_min_max = [y0, yr * 12 + mn]

    if 'land_mask' in time_features:
        ds = xr.concat([ds, land_mask.expand_dims('channels', axis = 0)], dim = 'channels')
    # TRAIN MODEL

    ds_train = ds[:n_train,...]
    obs_train = obs[:n_train,...]

    if model in [UNetLSTM_monthly, CNNLSTM_monthly]:
        ds_validation = ds[n_train - 11:n_train + num_val_years*12,...]
        obs_validation = obs[n_train - 11:n_train + num_val_years*12,...]
    else:
        ds_validation = ds[n_train:n_train + num_val_years*12,...]
        obs_validation = obs[n_train:n_train + num_val_years*12,...]

    if params['NPSProj']:
        weights = (np.ones_like(ds_train.lon) * (np.ones_like(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon})
        weights = weights * land_mask
        weights_val = weights.copy() * land_mask
    else:       
        weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove()  
        ####################################################################
        weights_val = weights.copy() * land_mask
        if params['equal_weights']:
            weights = xr.ones_like(weights)
        # if any(['land_mask' not in time_features, model not in [UNet2]]):
        weights = weights * land_mask

    if loss_region is not None:
        loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)
    
    else:
        loss_region_indices = None

    del ds, obs
    gc.collect()
    torch.cuda.empty_cache() 
    torch.cuda.synchronize() 
    weights = weights.values
    weights_val = weights_val.values

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

    n_channels_x = len(ds_train.channels)


    if model in [UNetLSTM, PNet,]:
        net = model(  n_channels_x= n_channels_x+ add_feature_dim ,seq_length = lead_months, bilinear=params['bilinear'], sigmoid = sigmoid_activation, device =  device)
    elif model in [CNNLSTM]:
        net = model(  n_channels_x= n_channels_x, add_features = add_feature_dim , seq_length = lead_months, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size, sigmoid = sigmoid_activation, device =  device)
    elif model in [UNet,UNetLCL,UNet2, UNet_NPS, UNet2_NPS]:
        net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'], sigmoid = sigmoid_activation, skip_conv = params['skip_conv'], combined_prediction = params['combined_prediction'])
    elif model in [CNN]: ## PG: Combining CNN encoder with dense decoder  
        net = model(n_channels_x + add_feature_dim ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size,DSC = params['DSC'], sigmoid = sigmoid_activation  )
    elif model in [ RegCNN]: ## PG: Combining CNN encoder with dense decoder  
        net = model(n_channels_x , add_feature_dim ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size,DSC = params['DSC'], sigmoid = sigmoid_activation )
    elif model in [CNNLSTM_monthly]:
        net = model(  n_channels_x= n_channels_x, add_features = add_feature_dim ,hidden_dims = hidden_dims, seq_length = 12, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size, sigmoid = sigmoid_activation, device =  device)
    elif model in [UNetLSTM_monthly]:
        net = model(  n_channels_x= n_channels_x+ add_feature_dim ,seq_length = 12, bilinear=params['bilinear'], sigmoid = sigmoid_activation, device =  device)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
    if params['lr_scheduler']:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)

    ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
    if lead_time is not None:
        mask =  create_mask(full_shape)[:n_train]  #create_mask(ds)[:n_train]
    else:
        mask = train_mask
    time_features  = params['time_features']   
    if model in [UNetLSTM_monthly, CNNLSTM_monthly]:
        train_set = ConvLSTMDataset(ds_train, obs_train, mask=mask, zeros_mask = zeros_mask, in_memory=True, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features,  month_min_max = month_min_max) 
    else:
        train_set = XArrayDataset(ds_train, obs_train, mask=mask, zeros_mask = zeros_mask, in_memory=False, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features, aligned = True, month_min_max = month_min_max, model = model.__name__) 
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if params['version'] == 'IceExtent':

        criterion = nn.BCELoss()
    else:

        if reg_scale is None: ## PG: if no penalizing for negative anomalies
            if low_ress_loss:
                criterion = WeightedMSELowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                criterion_MSE = WeightedMSELowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
            else:
                criterion = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                criterion_MSE = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)

        else:
            if low_ress_loss:
                criterion = WeightedMSEGlobalLossLowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
                criterion_MSE = WeightedMSELowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
            else:
                criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
                criterion_MSE = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)

        if params['combined_prediction']:
                criterion_extent = nn.BCELoss()
    # EVALUATE MODEL
    ##################################################################################################################################

    if lead_time is not None:
        if model in [UNetLSTM_monthly, CNNLSTM_monthly]:
            val_mask = create_mask(full_shape[n_train - 11:n_train + num_val_years*12] )[:, lead_time - 1][..., None] ####create_mask(ds_raw_ensemble_mean)[n_train - 11:n_train + num_val_years*12] 
        else:
            val_mask = create_mask(full_shape[n_train:n_train + num_val_years*12] )[:, lead_time - 1][..., None] ####create_mask(ds_raw_ensemble_mean)[n_train:n_train + num_val_years*12] 
    else:
        val_mask = create_mask(ds_validation)
    if model in [UNetLSTM_monthly, CNNLSTM_monthly]:
        validation_set = ConvLSTMDataset(ds_validation, obs_validation, mask=val_mask, zeros_mask = zeros_mask, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, month_min_max = month_min_max) 
    else:
        validation_set = XArrayDataset(ds_validation, obs_validation, mask=val_mask, zeros_mask = zeros_mask, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max, model = model.__name__) 
    dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=True)   

    if params['version'] == 'IceExtent':
        criterion_eval = nn.BCELoss()
    else:
        if reg_scale is None:
            if low_ress_loss:
                criterion_eval =  WeightedMSELowRess(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
            else:    
                criterion_eval =  WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
        else:
            if low_ress_loss:
                criterion_eval = WeightedMSEGlobalLossLowRess(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
            else:
                criterion_eval = WeightedMSEGlobalLoss(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
    
    criterion_eval_MSE =   WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
    # WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=10, map  =True)
    criterion_eval_area = GlobalLoss( device=device, scale=1, weights=weights_val, loss_area=None, map = True)
    criterion_eval_extent = IceextentlLoss( device=device, scale=1, weights=weights_val, loss_area=None, map = True)
    # IceextentlLoss(device=device, scale=1, weights=weights_val, loss_area=None, map = True)
    ##################################################################################################################################
    del ds_train, obs_train, ds_validation, obs_validation
    gc.collect()
    epoch_loss_train = []
    epoch_loss_train_MSE = []
    epoch_loss_val = []
    epoch_loss_val_extent = []
    epoch_loss_val_area = []
    epoch_loss_val_MSE = []

    for epoch in tqdm.tqdm(range(epochs)):
        net.train()
        batch_loss = 0
        batch_MSE = 0
        for batch, (x, y) in enumerate(dataloader):
            if model == PNet:
                ind = x[-1]  
                x = (x[0],x[1]) if len(x) >2 else x[0]
            else:
                ind = None 
            if (type(x) == list) or (type(x) == tuple):
                x = (x[0].to(device), x[1].to(device)) 
            else:
                x = x.to(device)
            if (type(y) == list) or (type(y) == tuple):
                y, m = (y[0].to(device), y[1].to(device))
            else:
                y = y.to(device)
                m  = None
            optimizer.zero_grad()
            if model in [UNet2, UNet2_NPS]:
                adjusted_forecast = net(x, torch.from_numpy(model_mask.to_numpy()).to(y))
            else:
                adjusted_forecast = net(x, ind = ind)
            if params['combined_prediction']:
                    (y, y_extent) = (y[:,0].unsqueeze(1), y[:,1].unsqueeze(1))
                    (adjusted_forecast, adjusted_forecast_extent) = adjusted_forecast
                    loss_extent = criterion_extent(adjusted_forecast_extent, y_extent)
                    loss = criterion(adjusted_forecast, y, mask = m)
                    MSE = criterion_MSE(adjusted_forecast, y, mask =m)
                    loss = loss + loss_extent
            else:
                if params['version'] == 'IceExtent':
                    loss = criterion(adjusted_forecast, y)
                    MSE = 0
                else:
                    loss = criterion(adjusted_forecast, y, mask = m)
                    MSE = criterion_MSE(adjusted_forecast, y, mask =m)

            batch_loss += loss.item()
            batch_MSE  += MSE.item()
            loss.backward()
            optimizer.step()  

        epoch_loss_train.append(batch_loss / len(dataloader))
        epoch_loss_train_MSE.append(batch_MSE / len(dataloader))

        if params['lr_scheduler']:
            scheduler.step()
        del  adjusted_forecast, y
        gc.collect()
        net.eval()
        val_loss = 0
        val_loss_extent = 0
        val_loss_area = 0
        val_loss_MSE = 0
        
        for batch, (x, target) in enumerate(dataloader_val):         
            with torch.no_grad():            
                if model == PNet:
                    ind = x[-1]  
                    x = (x[0],x[1]) if len(x) >2 else x[0]
                else:
                    ind = None 
                if (type(x) == list) or (type(x) == tuple):
                    test_raw = (x[0].to(device), x[1].to(device))
                else:
                    test_raw = x.to(device)
                if (type(target) == list) or (type(target) == tuple):
                    test_obs, m = (target[0].to(device), target[1].to(device))
                else:
                    test_obs = target.to(device)
                    m = None
                if model in [UNet2, UNet2_NPS]:
                    test_adjusted = net(test_raw, torch.from_numpy(model_mask.to_numpy()).to(device))
                else:
                    test_adjusted = net(test_raw, ind = ind)
                if params['combined_prediction']:
                    (test_obs, test_obs_extent) = (test_obs[:,0].unsqueeze(1), test_obs[:,1].unsqueeze(1))
                    (test_adjusted, test_adjusted_extent) = test_adjusted
                    loss_extent_ = criterion_extent(test_adjusted_extent, test_obs_extent)
                    loss_ = criterion_eval(test_adjusted, test_obs, mask = m)
                    loss_ = loss_ + loss_extent_
                else:
                    if params['version'] == 'IceExtent':
                        loss_ = criterion_eval(test_adjusted, test_obs)
                    else:
                        loss_ = criterion_eval(test_adjusted, test_obs,  mask =m)

                val_loss += loss_.item()
                if m is not None:
                    m[m != 0] = 1
                    test_adjusted = test_adjusted * m
                    m_ = m.sum(dim = (-1,-2)).unsqueeze(-1).unsqueeze(-1).expand_as(m)
                    m_[m_ != 0] = 1
                else:
                    m_ = None
                if params['version'] == 'IceExtent':
                    t_ = torch.ones_like(test_adjusted)
                    t_[test_adjusted<0.5] = 0
                    test_adjusted = t_ * m if m is not None else t_
                    del t_
                loss_extent = criterion_eval_extent(test_adjusted, test_obs, mask =  m_)
                val_loss_extent += loss_extent.item()
                loss_area = criterion_eval_area(test_adjusted, test_obs, mask =  m_)
                val_loss_area += loss_area.item()
                loss_MSE = criterion_eval_MSE(test_adjusted , test_obs, mask =  m_)

                val_loss_MSE += loss_MSE.item()


        epoch_loss_val.append(val_loss / len(dataloader_val))
        epoch_loss_val_extent.append(val_loss_extent / len(dataloader_val))
        epoch_loss_val_area.append(val_loss_area / len(dataloader_val))
        epoch_loss_val_MSE.append(val_loss_MSE / len(dataloader_val))
        # Store results as NetCDF            
    del train_set,validation_set, dataloader, dataloader_val, x , m, test_raw, test_obs,  target, test_adjusted, net
    try:
        del test_obs_extent, test_adjusted_extent, adjusted_forecast_extent, y_extent
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache() 
    torch.cuda.synchronize() 
    epoch_loss_val = smooth_curve(epoch_loss_val)
    epoch_loss_val_extent = smooth_curve(epoch_loss_val_extent)
    epoch_loss_val_area = smooth_curve(epoch_loss_val_area)
    epoch_loss_val_MSE = smooth_curve(epoch_loss_val_MSE)
    if params["version"] in ['IceExtent']:
        epoch_loss_val_area = np.zeros_like(epoch_loss_val_area)
        epoch_loss_val_extent = np.zeros_like(epoch_loss_val_extent)

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(2,epochs+1), epoch_loss_train[1:], color = 'b', label = 'Train Loss Total')
    plt.plot(np.arange(2,epochs+1), epoch_loss_val[1:], color = 'darkorange', label = 'Validation Loss Total')
    plt.plot(np.arange(2,epochs+1), epoch_loss_train_MSE[1:], color = 'b', label = 'Train MSE', linestyle = 'dashed', alpha = 0.5)
    plt.plot(np.arange(2,epochs+1), epoch_loss_val_MSE[1:], color = 'darkorange', label = 'Val MSE', linestyle = 'dashed', alpha = 0.5)
    plt.title(f'{hyperparamater_grid}')
    plt.legend()
    plt.ylabel(params['loss_function'])
    plt.twinx()
    plt.plot(np.arange(2,epochs+1), epoch_loss_val_area[1:], label = 'Validation Area Mean RMSE',  color = 'k', alpha = 0.5)
    plt.ylabel('MSE')
    plt.legend()
    plt.xlabel('Epoch')
    
    plt.grid()
    plt.show()
    plt.savefig(results_dir+f'/val-train_loss_1982-{test_year}-{hyperparamater_grid}.png')
    plt.close()


    with open(Path(results_dir, "Hyperparameter_training.txt"), 'a') as f:
        f.write(
  
            f"{hyperparamater_grid} ---> val_loss at best epoch: {min(epoch_loss_val)} at {np.argmin(epoch_loss_val)+1}  (MSE : {epoch_loss_val_MSE[np.argmin(epoch_loss_val)]})\n" +  ## PG: The scale to be passed to Signloss regularization
            f"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" 
        )
    return epoch_loss_val_MSE[np.argmin(epoch_loss_val)], epoch_loss_val, epoch_loss_train ,epoch_loss_val_area, epoch_loss_val_extent, epoch_loss_val_MSE, epoch_loss_train_MSE

                                 #########         ##########

def run_hp_tunning( ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset, zeros_mask_full :XArrayDataset, land_masks :XArrayDataset,  hyperparameterspace: list, params:dict, y_start: int, y_end:int, out_dir_x , lead_time = None, n_runs=1, numpy_seed=None, torch_seed=None ):

    


    val_losses = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_area = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_extent = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    val_losses_mean = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
    train_losses = np.zeros([y_end - y_start + 1, len(hyperparameterspace), params['epochs']]) ####
    train_losses_MSE = np.zeros([y_end - y_start + 1, len(hyperparameterspace), params['epochs']]) ####

    for ind_, test_year in enumerate(range(y_start,y_end+1)):
    
        # out_dir_xx = f'{out_dir_x}/git_data_20230426'
        # out_dir    = f'{out_dir_xx}/SPNA' 
        out_dir    = f'{out_dir_x}/_{test_year}' 
        
        
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if  params['lr_scheduler']:
            start_factor = params['start_factor']
            end_factor = params['end_factor']
            total_iters = params['total_iters']
        else:
            start_factor = end_factor =total_iters =None
 

        with open(Path(out_dir, "Hyperparameter_training.txt"), 'w') as f:
            f.write(
                f"model\t{params['model']}\n" +
                "default set-up:\n" + 
                f"hidden_dims\t{params['hidden_dims']}\n" +
                f"loss_function\t{params['loss_function']}\n" + 
                f"time_features\t{params['time_features']}\n" +
                f"obs_clim\t{params['obs_clim']}\n" +
                f"ensemble_list\t{params['ensemble_list']}\n" + ## PG: Ensemble list
                f"ensemble_features\t{params['ensemble_features']}\n" + ## PG: Ensemble features
                f"lr\t{params['lr']}\n" +
                f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n" + 
                f"decoder_kernel_size\t{params['decoder_kernel_size']}\n" +
                f"L2_reg\t{params['L2_reg']}\n" +
                f"skip_conv\t{params['skip_conv']}\n" +
                f"low_ress_loss\t{params['low_ress_loss']}\n" +
                f"equal_weights\t{params['equal_weights']}\n" +
                f"active_grid\t{params['active_grid']}\n\n\n" +
                ' ----------------------------------------------------------------------------------\n'
            )
        

        
        losses = np.zeros(len(hyperparameterspace))
 

        
        for ind, dic in enumerate(hyperparameterspace):
            print(f'Training for {dic}')
            # losses[ind], val_losses[ind_, ind, :], val_losses_global[ind_, ind, :], val_losses_corr[ind_, ind, :],  train_losses[ind_, ind, :] = run_training_hp(dic, params, test_year=test_year, lead_years=lead_years, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)
            losses[ind], val_losses[ind_, ind, :],  train_losses[ind_, ind, :] ,val_losses_area[ind_, ind, :],  val_losses_extent[ind_, ind, :], val_losses_mean[ind_, ind, :],  train_losses_MSE[ind_, ind, :] = training_hp(ds_raw_ensemble_mean =  ds_raw_ensemble_mean,obs_raw = obs_raw ,
                   hyperparamater_grid= dic,zeros_mask_full = zeros_mask_full,land_masks=land_masks, params = params , test_year=test_year, lead_time = lead_time, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed)

        
        with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
            f.write(
    
                f"Best MSE: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]} \n" +  ## PG: The scale to be passed to Signloss regularization
                f"--------------------------------------------------------------------------------------------------------\n" 
            )

        print(f"Best loss: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]}")
        print(f'Output dir: {out_dir}')
        print('Training done.')

    coords = []
    for item in hyperparameterspace:
        coords.append(str(tuple(item.values())))

    ds_val = xr.DataArray(val_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val.to_netcdf(out_dir_x + '/validation_losses.nc')

    
    ds_val_area = xr.DataArray(val_losses_area, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_area.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_area.to_netcdf(out_dir_x + '/validation_losses_area_mean.nc')

    ds_val_extent = xr.DataArray(val_losses_extent, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_extent.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_extent.to_netcdf(out_dir_x + '/validation_losses_ice_extent.nc')

    ds_val_mean = xr.DataArray(val_losses_mean, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
    ds_val_mean.attrs['hyperparameters'] = list(config_dict.keys())
    ds_val_mean.to_netcdf(out_dir_x + '/validation_losses_MSE.nc')

    ds_train = xr.DataArray(train_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
    ds_train.attrs['hyperparameters'] = list(config_dict.keys())
    ds_train.to_netcdf(out_dir_x + '/train_losses.nc')

    ds_train_MSE = xr.DataArray(train_losses_MSE, dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
    ds_train_MSE.attrs['hyperparameters'] = list(config_dict.keys())
    ds_train_MSE.to_netcdf(out_dir_x + '/train_losses_MSE.nc')

            

if __name__ == "__main__":

    params = {
        "model": UNet2,
        "hidden_dims":  [ 64, 128,256, 128, 64],#[16, 64, 128, 64, 32],## only for (Reg)CNN
        "time_features": ['month_sin','month_cos', 'imonth_sin', 'imonth_cos'],
        "obs_clim" : False,
        "ensemble_features": False, ## PG
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 100,
        "batch_size": 25,
        "reg_scale" : None,
        "optimizer": torch.optim.Adam,
        "lr": 0.001,
        "loss_function" :'MSE',
        "loss_region": None,
        "subset_dimensions": 'North' , ##  North or South or Global
        'active_grid' : False,
        'low_ress_loss' : False,
        'equal_weights' : False,
        "kernel_size" : 5,
        "decoder_kernel_size" : 1,
        'DSC' : False,
        "bilinear" : True, ## only for UNet
        "L2_reg": 0,
        'lr_scheduler' : True,
        'skip_conv' : False,
        'combined_prediction' : False
    }



    ################################################################# Set basic config ###########################################################################
    lead_months = 12
    lead_time = None
    n_runs = 1  # number of training runs
    params['version'] = 1 ### 1 , 2 ,3, 'PatternsOnly' , 'IceExtent'
    
    ### load data

    obs_ref = 'NASA'
    NPSProj = False
    y_start = 2018
    y_end = 2018

    # y_end = int(ds_raw_ensemble_mean.time[-1]/100) +1 
    params['num_val_years'] = 3
   ##############################################################  Don't touch the following ######################################################################
    ds_raw_ensemble_mean, obs_raw, params, zeros_mask_full, land_masks = HP_congif(params, obs_ref, lead_months, y_start, y_end, NPSProj=NPSProj)
    ########################################################### Set HP space specifics #########################################################################
    
    config_dict = {'time_features' : [['land_mask'], ['month_sin','month_cos', 'imonth_sin', 'imonth_cos'], ['month_sin','month_cos', 'imonth_sin', 'imonth_cos', 'land_mask']] }
    # config_dict = {  'batch_size': [100, 200], 'reg_scale' : [None, 50, 100], 'L2_reg' : [0, 0.0001,0.001 ] }
    hyperparameterspace = config_grid(config_dict).full_grid()

    ##################################################################  Adjust the path if necessary #############3##############################################
    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/{obs_ref}/{params["model"].__name__}/run_set_3_convnext/Model_tunning/'
    if lead_time is None:
        out_dir = out_dir_x + f'NV{params["num_val_years"]}_M{lead_months}_{params["subset_dimensions"]}_v{params["version"]}'
    else:
        out_dir = out_dir_x + f'NV{params["num_val_years"]}_LT{lead_time}_{params["subset_dimensions"]}_v{params["version"]}'
    
    out_dir = out_dir + '_NPSproj' if NPSProj else out_dir + '_1x1'

    if params['active_grid']:
        out_dir = out_dir + '_active_grid'
    if params['bilinear']:
        out_dir = out_dir + '_bilinear'
    if params['skip_conv']:
        out_dir = out_dir + '_skip_conv'   
    if params['combined_prediction']:
        out_dir = out_dir + '_combined'  
    if params['lr_scheduler']:
        out_dir = out_dir + '_lr_scheduler'
        params['start_factor'] = 1.0
        params['end_factor'] = 0.1
        params['total_iters'] = 100
        
    run_hp_tunning(ds_raw_ensemble_mean = ds_raw_ensemble_mean ,obs_raw = obs_raw, zeros_mask_full = zeros_mask_full,land_masks = land_masks,  
                   lead_time = lead_time, hyperparameterspace = hyperparameterspace, params = params, y_start = y_start ,y_end = y_end, out_dir_x = out_dir, n_runs=1, numpy_seed=1, torch_seed=1 )

