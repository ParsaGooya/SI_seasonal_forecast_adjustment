import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import dask
import xarray as xr
from pathlib import Path
import glob
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.autoencoder import Autoencoder
from models.unet import UNet, UNetLCL,UNet_NPS
from models.unetconvnext import UNet2,UNet2_NPS
from models.cnn import CNN, RegCNN
from losses import WeightedMSE, WeightedMSEGlobalLoss
from losses import WeightedMSELowRess , WeightedMSEGlobalLossLowRess
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


    if model_year is None:
        model_year_ = np.min(test_years) - 1
    else:
        model_year_ = model_year
        
    if params["model"] != Autoencoder:
        params["append_mode"] = None
    else:   
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

    
    if params['version'] == 2:

        params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1_seasonal())]
        params['observations_preprocessing_steps'] = []
        
    else:
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []

    if params['version'] == 'IceExtent':
        params['reg_scale'] = None


    print(f"Start run for test year {test_years}...")

    ############################################## load data ##################################
    ensemble_list = params['ensemble_list']
    ensemble_features = params['ensemble_features']
    time_features = params["time_features"]
    model = params['model']
    hidden_dims = params['hidden_dims']
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]
    try:
        ensemble_mode = params['ensemble_mode']
    except:
        params['ensemble_mode'] = 'Mean'
    try:
        batch_normalization = params["batch_normalization"]
        dropout_rate = params["dropout_rate"]
    except:
        obs_clim = params["obs_clim"]
        kernel_size = params["kernel_size"]
        decoder_kernel_size = params["decoder_kernel_size"]

    print("Load forecasts")
    if params['version'] == 3:
        ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/bias_adjusted_North_1983-2020_1x1.nc')['SICN']
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

    ds_raw, obs_raw = align_data_and_targets(ds_in.where(ds_in.time <= (model_year_ + 1)*100, drop = True), obs_in, lead_months)  # extract valid lead times and usable years ## used to be np.min(test_years)
    del ds_in, obs_in
    gc.collect()

    if not ds_raw.time.equals(obs_raw.time):
            
            ds_raw = ds_raw.sel(time = obs_raw.time)
    
    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time','ensembles',...)
        ds_in_ = ds_in_.transpose('time','lead_time','ensembles',...)
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time',...)
        ds_in_ = ds_in_.transpose('time','lead_time',...)
    

    train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time <= (model_year_ + 1)*100].to_numpy()    
    ds_raw_ensemble_mean = xr.concat([ds_raw_ensemble_mean,ds_in_ ], dim = 'time')
    subset_dimensions = params["subset_dimensions"]
    del ds_in_
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
    if params['model'] not in [UNet2, UNet2_NPS]:
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

    if NPSProj is False:
        if model in [UNet,UNetLCL, CNN,UNet2]:
            ds_raw_ensemble_mean = pole_centric(ds_raw_ensemble_mean, subset_dimensions)
            obs_raw =  pole_centric(obs_raw, subset_dimensions)
            land_mask = pole_centric(land_mask, subset_dimensions)
            model_mask = pole_centric(model_mask, subset_dimensions)
            if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                zeros_mask_full = pole_centric(zeros_mask_full, subset_dimensions)

        if model in [RegCNN]:
            ds_raw_ensemble_mean = segment(ds_raw_ensemble_mean, 9)
            obs_raw =  segment(obs_raw, 9)
            land_mask = segment(land_mask, 9)
            model_mask = segment(model_mask,9)
            if any(['active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                raise RuntimeError('You cannot add mask data in chennels with RegCNN model where you have regions stacked as channels!')
            if params['active_grid']:
                zeros_mask_full = segment(zeros_mask_full, 9)
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

        
    if NPSProj:
        weights = (np.ones_like(ds_train.lon) * (np.ones_like(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon})
        weights = weights * mask_projection
        # weights_ = weights * land_mask
    else:
        weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
        ########################################################################
        # weights_ = weights * land_mask
        if params['equal_weights']:
            weights = xr.ones_like(weights)
        if any(['land_mask' not in time_features, model not in [UNet2]]):
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

    params['bilinear'] = False if 'bilinear' not in params.keys() else ...

    if model in [UNet,UNetLCL,UNet2, UNet_NPS, UNet2_NPS]:
        net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'], sigmoid = sigmoid_activation, skip_conv = params['skip_conv'], combined_prediction = params['combined_prediction'])
    elif model in [ CNN]:
        net = model(n_channels_x + add_feature_dim ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size, sigmoid = sigmoid_activation )
    elif model in [ RegCNN]: 
        net = model(n_channels_x , add_feature_dim ,hidden_dimensions =hidden_dims,  kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size, DSC = DSC, sigmoid = sigmoid_activation )

    print('Loading model ....')
    net.load_state_dict(torch.load(glob.glob(model_dir + f'/*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 
    net.to(device)
    net.eval()
    ##################################################################################################################################

    test_years_list = np.arange(1, ds_test.shape[0] + 1)
    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
    test_set = XArrayDataset(ds_test, xr.ones_like(ds_test), lead_time=lead_time,mask = None,zeros_mask = zeros_mask, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max, model = model.__name__)
                    
    if lead_time is None:
        lead_times = ds_test.lead_time.values
    else:
        lead_times = [lead_time]
    if 'ensembles' in ds_test.dims:
        # if model == UNetLSTM:
        #     test_loss = np.zeros(shape=(ds_test.sel(lead_time = lead_times).transpose('time','ensembles','channels',...).shape[:2]))
        #     test_results = np.zeros_like(ds_test.sel(lead_time = lead_times).transpose('time','ensembles','channels',...).data)
        #     results_shape = xr.full_like(ds_test.sel(lead_time = lead_times).transpose('time','ensembles','channels',...), fill_value = np.nan)
        #     test_time_list =  np.arange(1, results_shape.shape[0] + 1)
        # else:
            test_loss = np.zeros(shape=(ds_test.stack(flattened=('time','lead_time')).sel(lead_time = lead_times).transpose('flattened',...).shape[:2]))
            test_results = np.zeros_like(ds_test.stack(flattened=('time','lead_time')).sel(lead_time = lead_times).transpose('flattened',...).data)
            results_shape = xr.full_like(ds_test.stack(flattened=('time','lead_time')).sel(lead_time = lead_times).transpose('flattened',...), fill_value = np.nan)
    else:
        test_loss = np.zeros(shape=(test_set.target.shape[0]))
        test_results = np.zeros_like(test_set.target.isel(channels = slice(0,1)))
        results_shape = xr.full_like(test_set.target.isel(channels = slice(0,1)), fill_value = np.nan)
   
    test_time_list =  np.arange(1, ds_test.shape[0] + 1) ########?
    if params['combined_prediction']:
            test_results_extent = test_results.copy()
            results_shape_extent = results_shape.copy()

    if params['active_grid']:
        if 'ensembles' in ds_test.dims: 
            zeros_mask_test = results_shape.isel(ensembles = 0).copy()
            zeros_mask_test[:] = test_set.zeros_mask[:len(test_time_list)]
        else:
            zeros_mask_test = results_shape.copy()
            zeros_mask_test[:] = test_set.zeros_mask[:len(test_time_list)]
        # if model == UNetLSTM:
        #     zeros_mask_test = zeros_mask_test.transpose('time','lead_time','channels','lat','lon')
        # else:
        zeros_mask_test = zeros_mask_test.unstack('flattened').transpose('time','lead_time',...)

    dataloader = DataLoader(test_set, batch_size=len(lead_times), shuffle=False)

    for time_id, (x, target) in enumerate(dataloader): 
        if 'ensembles' in ds_test.dims:  ## PG: If we have large ensembles:
                ens_id, time_id = np.divmod(time_id, len(test_time_list))  ## PG: find out ensemble index
        with torch.no_grad():
            if (type(x) == list) or (type(x) == tuple):
                # ind = x[2] if model == PNet else None    
                test_raw = (x[0].to(device), x[1].to(device))
            else:
                test_raw = x.to(device)

            if model in [UNet2, UNet2_NPS]:
                test_adjusted = net(test_raw, torch.from_numpy(model_mask.to_numpy()).to(device))
            else:
                test_adjusted = net(test_raw)

            if params['combined_prediction']:
                (test_adjusted, test_adjusted_extent) = test_adjusted
                if 'ensembles' in ds_test.dims: 
                    test_results_extent[ens_id,time_id * len(lead_times) : (time_id+1) * len(lead_times)] = test_adjusted_extent.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
                else:
                    test_results_extent[time_id * len(lead_times) : (time_id+1) * len(lead_times)] = test_adjusted.to(torch.device('cpu')).numpy()
            
            if 'ensembles' in ds_test.dims:   
                test_results[ens_id, time_id * len(lead_times) : (time_id+1) * len(lead_times)] = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
            else:
                test_results[time_id * len(lead_times) : (time_id+1) * len(lead_times)] = test_adjusted.to(torch.device('cpu')).numpy()

    del  test_raw,  x, target, test_adjusted, test_set, ds_test
    try:
        del  test_adjusted_extent
    except:
        pass
    gc.collect()
    ###################################################### has to be eddited for large ensembles!! #####################################################################
    results_shape[:] = test_results[:]

    # if model in [UNet,UNetLCL,CNN]:    ## PG: if the output is already a map
    # if model == UNetLSTM:
    #     test_results = results_shape.transpose('time','lead_time',...)
    # else:
    test_results = results_shape.unstack('flattened').transpose('time','lead_time',...)
    test_results_untransformed = obs_pipeline.inverse_transform(test_results.values)
    result = xr.DataArray(test_results_untransformed, test_results.coords, test_results.dims, name='nn_adjusted')

    if params['combined_prediction']:
            results_shape_extent[:] = test_results_extent[:]
            result_extent = results_shape_extent.unstack('flattened').transpose('time','lead_time',...)


    if obs_clim:
        result = result.isel(channels = 0).expand_dims('channels', axis=2)

    del  results_shape, test_results, test_results_untransformed
    try:
        results_shape_extent
    except:
        pass
    gc.collect()

    result = (result * land_mask)
    if not NPSProj:
        if model in [UNet,UNetLCL, CNN, UNet2]:
            result = reverse_pole_centric(result, subset_dimensions)
        if model in [RegCNN]:
            result = reverse_segment(result)
    result = result.to_dataset(name = 'nn_adjusted') 


    if params['active_grid']:
        if not NPSProj:
            if model in [UNet , UNetLCL,CNN,UNet2]:
                zeros_mask_test = reverse_pole_centric(zeros_mask_test)
            if model in [RegCNN]:
                zeros_mask_test = reverse_segment(zeros_mask_test)
        else:
            zeros_mask_test = zeros_mask_test.rename({'lon':'x', 'lat':'y'})
        result = xr.combine_by_coords([result * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])

    if params['version'] == 'IceExtent':
            result = result.where(result >= 0.5, 0)
            result = result.where(result ==0, 1)

    if params['combined_prediction']:
            result_extent = (result_extent * land_mask)
            if not NPSProj:
                if model in [UNet,UNetLCL, CNN,UNet2]:
                        result_extent = reverse_pole_centric(result_extent, subset_dimensions)
                if model in [RegCNN]:
                        result_extent = reverse_segment(result_extent)
            result_extent = result_extent.to_dataset(name = 'nn_adjusted_extent')
            result_extent = result_extent.where(result_extent >= 0.5, 0)
            result_extent = result_extent.where(result_extent ==0, 1)
            result = xr.combine_by_coords([result , result_extent])
    if save:
        if model_year is not None:
            Path(model_dir + f'/{model_year}_model_predictions').mkdir(parents=True, exist_ok=True)
            model_dir = model_dir + f'/{model_year}_model_predictions'

        if ensemble_mode != 'Mean':
            if np.min(test_years) != np.max(test_years):
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}_LE.nc', mode='w'))
            else:
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}_LE.nc', mode='w'))

        else:

            if np.min(test_years) != np.max(test_years):
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}.nc', mode='w'))
            else:
                result.to_netcdf(path=Path(model_dir, f'saved_model_nn_adjusted_{np.min(test_years)}.nc', mode='w'))

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

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/UNet2/run_set_2_convnext'
    out_dir    = f'{out_dir_x}/N5_LT1_F12_v1_1x1_North_lr0.001_batch25_e100_LNone_bilinear' 

    lead_months = 12
    bootstrap = False
    test_years = np.arange(2016,2022)

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
    for i in range(1,13):
        out_dir    = f'{out_dir_x}/N5_LT{i}_F12_v3_1x1_North_lr0.001_batch25_e100_LNone_bilinear_combined'  
        params = extract_params(out_dir)
        print(f'loaded configuration: \n')
        for key, values in params.items():
            print(f'{key} : {values} \n')
        

        version = int(out_dir.split('/')[-1].split('_')[3][1])

        
        params["version"] = version
        print( f'Version: {version}')

        
        if bootstrap:

            result_list = []
            ensembles = np.arange(1,11)#[f'r{i}i1p2f1' for i in range(1,11)]

            for it in tqdm(range(200)):

                ensemble_list = [random.choice(ensembles) for _ in range(len(ensembles))]
                result_list.append(predict(fct, observation, params, lead_months, out_dir,  test_years, ensemble_list = ensemble_list,  save=False))
            
            output = xr.concat(result_list, dim = 'iteration')

            if np.min(test_years) != np.max(test_years):
                output.to_netcdf(path=Path(out_dir, f'saved_model_nn_adjusted_{np.min(test_years)}-{np.max(test_years)}_bootstrap.nc', mode='w'))
            else:
                output.to_netcdf(path=Path(out_dir, f'saved_model_nn_adjusted_{np.min(test_years)}_bootstrap.nc', mode='w'))
        
        else:
            predict(fct, observation, params, lead_months, out_dir,  test_years, NPSProj  = NPSProj, model_year = 2016, ensemble_mode='Mean',  save=True)

        print(f'Output dir: {out_dir}')
        print('Saved!')

