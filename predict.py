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
from models.unet import UNet,  UNetLCL
from models.cnn import CNN
from models.convlstm import UNetLSTM, PNet
from losses import WeightedMSE, WeightedMSEOutlierLoss, WeightedMSEGlobalLoss
from preprocessing import align_data_and_targets, create_mask, pole_centric, reverse_pole_centric, zeros_mask_gen
from preprocessing import AnomaliesScaler_v1_seasonal,AnomaliesScaler_v2_seasonal, Standardizer, Normalizer, PreprocessingPipeline, calculate_climatology, bias_adj
from torch_datasets import XArrayDataset
import torch.nn as nn
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
import glob




def predict(fct:xr.DataArray , observation:xr.DataArray , params, lead_months, model_dir,  test_years, model_year = None, ensemble_list = None, ensemble_mode = 'Mean', btstrp_it = 200, save=True):


    if model_year is None:
        model_year_ = np.min(test_years) - 1
    else:
        model_year_ = model_year

    if params['version'] == 'PatternsOnly':
        params["obs_clim"] = False
        forecast_preprocessing_steps = [ ('standardize', Standardizer())]
        observations_preprocessing_steps = []
 
    elif params['version'] == 3:

        forecast_preprocessing_steps = [
            ('anomalies', AnomaliesScaler_v1_seasonal())]
        observations_preprocessing_steps = [
            ('anomalies', AnomaliesScaler_v2_seasonal()) ]
         
    else:
        forecast_preprocessing_steps = []
        observations_preprocessing_steps = []

    if params['version'] == 2:

        fct = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/bias_adjusted_North_1983-2020.nc')['SICN']
  

    print(f"Start run for test year {test_years}...")

    ############################################## load data ##################################
    ensemble_list = params['ensemble_list']
    ensemble_features = params['ensemble_features']
    time_features = params["time_features"]
    model = params['model']
    hidden_dims = params['hidden_dims']

    try:
        batch_normalization = params["batch_normalization"]
        dropout_rate = params["dropout_rate"]
    except:
        obs_clim = params["obs_clim"]
        kernel_size = params["kernel_size"]
        decoder_kernel_size = params["decoder_kernel_size"]

    print("Load forecasts")
    if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
        ds_in = fct.sel(ensembles = ensemble_list)['SIC']
            
    else:    ## Load specified members
        ds_in = fct['SIC']

    if ensemble_mode == 'Mean': ##
        ensemble_features = False
        ds_in = ds_in.mean('ensembles').load() ##
    else:
        ds_in = ds_in.load().transpose('time','lead_time','ensembles',...)
        print('Warning: ensemble_mode is None. Predicting for large ensemble ...')
        
    ###### handle nan and inf over land ############
    ds_in = ds_in.where(ds_in<1000,0) ### land is masked in model data with a large number
    observation = observation.clip(0,1)
    ds_in = ds_in.clip(0,1)
    observation = observation.where(~np.isnan(observation) , 0)
    ds_in = ds_in.where(~np.isnan(ds_in) , 0)
    ############################################
    
    obs_in = observation.expand_dims('channels', axis=1)

    if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
        ds_in = ds_in.expand_dims('channels', axis=3).sortby('ensembles')
    else:
        ds_in = ds_in.expand_dims('channels', axis=2) 

    min_year = np.min(test_years)*100
    max_year = (np.min(test_years) + 1 )*100 if len(test_years) <2 else (np.max(test_years) + 1)*100
    ds_in_ = ds_in.where((ds_in.time >= min_year)&(ds_in.time <= max_year) , drop = True).isel(lead_time = np.arange(0,lead_months ))

    ds_raw, obs_raw = align_data_and_targets(ds_in.where(ds_in.time <= (model_year_ + 1)*100, drop = True), obs_in, lead_months)  # extract valid lead times and usable years ## used to be np.min(test_years)

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
    
    if params['subset_dimensions'] is not None:
        if params['subset_dimensions'] == 'North':
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat > 40, drop = True)
            obs_raw = obs_raw.where(obs_raw.lat > 40, drop = True)
        else:
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat < -40, drop = True)
            obs_raw = obs_raw.where(obs_raw.lat < -40, drop = True)
    
    if params['active_grid']:
        zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(lead_time = 0).drop('lead_time').where(obs_raw.time<test_year*100, drop = True ), 5) for test_year in test_years], dim = 'test_year').assign_coords(test_year = test_years)           
        zeros_mask_full = zeros_mask_full.expand_dims('channels', axis=-3)
        zeros_mask_full = zeros_mask_full * (zeros_mask_full.sum(['lat','lon']).max('month')/zeros_mask_full.sum(['lat','lon']))
        if 'ensembles' in ds_raw.dims:
             zeros_mask_full = zeros_mask_full.expand_dims('ensembles', axis=2)

    if params['version'] == 'PatternsOnly':

        weights = np.cos(np.ones_like(ds_raw_ensemble_mean.lon) * (np.deg2rad(ds_raw_ensemble_mean.lat.to_numpy()))[..., None])  # Moved this up
        weights = xr.DataArray(weights, dims = ds_raw_ensemble_mean.dims[-2:], name = 'weights').assign_coords({'lat': ds_raw_ensemble_mean.lat, 'lon' : ds_raw_ensemble_mean.lon}) 
        ds_raw_mean = ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
        obs_raw_mean = ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
        ds_raw_ensemble_mean = ds_raw_ensemble_mean - ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
        obs_raw = obs_raw - ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))



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
######################################################################################### checked till here ##
    if model in [UNet, CNN, UNetLCL, UNetLSTM, PNet]:
        ds_raw_ensemble_mean = pole_centric(ds_raw_ensemble_mean, params['subset_dimensions'])
        obs_raw =  pole_centric(obs_raw, params['subset_dimensions'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params['active_grid']:
        zeros_mask = zeros_mask_full.sel(test_year = model_year_ + 1).drop('test_year')
    else:
        zeros_mask = None
    
    n_train = len(train_years)
    train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...])

    ds_baseline = ds_raw_ensemble_mean[:n_train,...]
    obs_baseline = obs_raw[:n_train,...]
    if params['version'] == 'PatternsOnly':
            ds_baseline_mean = ds_raw_mean[:n_train,...]
            obs_baseline_mean = obs_raw_mean[:n_train,...]

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
    obs = obs_pipeline.transform(obs_raw)

    if params['version']  in ['PatternsOnly',3]:
        sigmoid_activation = False
    else:
        sigmoid_activation = True

    year_max = ds[:n_train].time[-1].values 
    

    # TRAIN MODEL

    lead_time = None
    ds_train = ds[:n_train,...]
    obs_train = obs[:n_train,...]
    
    ds_test = ds[n_train: ,...]

    if params['version'] == 'PatternsOnly':
        if 'ensembles' in ds_baseline_mean.dims:  
            mask_bias_adj = train_mask[...,None, None]
        else:
            mask_bias_adj = train_mask[...,None]

        fct_climatology = calculate_climatology(ds_baseline_mean.where(~np.broadcast_to(mask_bias_adj, ds_baseline_mean.shape)))
        obs_climatology = calculate_climatology(obs_baseline_mean.where(~np.broadcast_to(train_mask[...,None], obs_baseline_mean.shape)))
        fct_bias_adjusted = bias_adj(ds_raw_mean[n_train:,...] , fct_climatology , obs_climatology)


        
    weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 


    ########################################################################

    if model not in [UNet,UNetLCL,CNN, UNetLSTM, PNet] : ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

        ds_train = ds_train.stack(ref = ['lat','lon']) # PG: flatten and sample training data at those locations
        obs_train = obs_train.stack(ref = ['lat','lon']) ## PG: flatten and sample obs data at those locations
        ds_test = ds_test.stack(ref = ['lat','lon'])
        weights = weights.stack(ref = ['lat','lon']) ## PG: flatten and sample weighs at those locations

        img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.


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

    ########################################### load the model ######################################
    try:
        if params['obs_clim']:
            n_channels_x = len(ds_train.channels) + 1
        else:
            n_channels_x = len(ds_train.channels)
        
    except:
        pass

    params['bilinear'] = False if 'bilinear' not in params.keys() else ...

    if model == Autoencoder:
        net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate)
    elif model in [UNet,UNetLCL]:
        net = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params['bilinear'], sigmoid = sigmoid_activation)
    elif model in [ CNN]:
        net = model(n_channels_x + add_feature_dim ,hidden_dims, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size, sigmoid = sigmoid_activation )
    elif model in [UNetLSTM, PNet]:
        net = model(  n_channels_x= n_channels_x+ add_feature_dim ,seq_length = lead_months, bilinear=params['bilinear'], sigmoid = sigmoid_activation, device =  device)
 
    print('Loading model ....')
    net.load_state_dict(torch.load(glob.glob(model_dir + f'/*-{model_year_}*.pth')[0], map_location=torch.device('cpu'))) 
    net.to(device)
    net.eval()
    ##################################################################################################################################

    test_years_list = np.arange(1, ds_test.shape[0] + 1)
    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)
    test_set = XArrayDataset(ds_test, xr.ones_like(ds_test),mask = None,zeros_mask=zeros_mask, lead_time=None, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, year_max = year_max,  model = model.__name__)

    if 'ensembles' in ds_test.dims:
        if model == UNetLSTM:
            test_loss = np.zeros(shape=(ds_test.transpose('time','ensembles','channels',...).shape[:2]))
            test_results = np.zeros_like(ds_test.transpose('time','ensembles','channels',...).data)
            results_shape = xr.full_like(ds_test.transpose('time','ensembles','channels',...), fill_value = np.nan)
            test_time_list =  np.arange(1, results_shape.shape[0] + 1)
        else:
            test_loss = np.zeros(shape=(ds_test.stack(flattened=('time','lead_time')).transpose('flattened',...).shape[:2]))
            test_results = np.zeros_like(ds_test.stack(flattened=('time','lead_time')).transpose('flattened',...).data)
            results_shape = xr.full_like(ds_test.stack(flattened=('time','lead_time')).transpose('flattened',...), fill_value = np.nan)
            test_time_list =  np.arange(1, results_shape.shape[0] + 1)
    else:
        test_loss = np.zeros(shape=(test_set.target.shape[0]))
        test_results = np.zeros_like(test_set.target)
        results_shape = xr.full_like(test_set.target, fill_value = np.nan)

    if params['active_grid']:
        if 'ensembles' in ds_test.dims: 
            zeros_mask_test = results_shape.isel(ensembles = 0).copy()
            zeros_mask_test[:] = test_set.zeros_mask[:len(test_time_list)]
        else:
            zeros_mask_test = results_shape.copy()
            zeros_mask_test[:] = test_set.zeros_mask[:len(test_time_list)]
        if model == UNetLSTM:
            zeros_mask_test = zeros_mask_test.transpose('time','lead_time','channels','lat','lon')
        else:
            zeros_mask_test = zeros_mask_test.unstack('flattened').transpose('time','lead_time',...)


    for i, (x, target) in enumerate(test_set): 
        if 'ensembles' in ds_test.dims:  ## PG: If we have large ensembles:

            ensemble_idx, j = np.divmod(i, len(test_time_list))  ## PG: find out ensemble index

            
            with torch.no_grad():
                if (type(x) == list) or (type(x) == tuple):
                    ind = x[2] if model == PNet else None    
                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                else:
                    test_raw = x.unsqueeze(0).to(device)
                if (type(target) == list) or (type(target) == tuple):
                    test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                else:
                    test_obs = target.unsqueeze(0).to(device)
                    m = None

                test_adjusted = net(test_raw, ind = ind)
                test_results[j,ensemble_idx,]  = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results

        else:

            with torch.no_grad():
                if (type(x) == list) or (type(x) == tuple):
                    ind = x[2] if model == PNet else None 
                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                else:
                    test_raw = x.unsqueeze(0).to(device)
                if (type(target) == list) or (type(target) == tuple):
                    test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                else:
                    test_obs = target.unsqueeze(0).to(device)
                    m = None
                test_adjusted = net(test_raw, ind = ind)
                test_results[i,] = test_adjusted.to(torch.device('cpu')).numpy()

    ##################################################################################################################################
    results_shape[:] = test_results[:]

    if model in [UNet,UNetLCL,CNN,UNetLSTM, PNet]:    ## PG: if the output is already a map
        if model == UNetLSTM:
            test_results = results_shape.transpose('time','lead_time',...)
        else:
            test_results = results_shape.unstack('flattened').transpose('time','lead_time',...)
        if test_results.shape[0] < 12:
            test_results = test_results.combine_first(xr.full_like(ds_test, np.nan))
        test_results_untransformed = obs_pipeline.inverse_transform(test_results.values)
        result = xr.DataArray(test_results_untransformed, ds_test.coords, ds_test.dims, name='nn_adjusted')
    else:  
        test_results = results_shape.unstack('flattened').transpose('time','lead_time',...,'ref').unstack('ref')
        if test_results.shape[0] < 12:
            test_results = test_results.combine_first(xr.full_like(ds_test.unstack('ref'), np.nan))

        test_results_untransformed = obs_pipeline.inverse_transform(test_results.values) ## PG: Check preprocessing.AnomaliesScaler for changes
        result = xr.DataArray(test_results_untransformed, ds_test.unstack('ref').coords, ds_test.unstack('ref').dims, name='nn_adjusted')
    
    if obs_clim:
        result = result.isel(channels = 0).expand_dims('channels', axis=2)

    if model in [UNet,UNetLCL,  CNN, UNetLSTM, PNet]:
        result = reverse_pole_centric(result, params['subset_dimensions'])

    if params['version'] == 'PatternsOnly':                      
        result = result.to_dataset(name = 'nn_adjusted') + fct_bias_adjusted.to_dataset(name = 'nn_adjusted')

    if params['active_grid']:
        if model in [UNet , UNetLCL,CNN, UNetLSTM, PNet]:
            zeros_mask_test = reverse_pole_centric(zeros_mask_test)
        if params['version'] == 'PatternsOnly':
            result = xr.combine_by_coords([result, zeros_mask_test.to_dataset(name = 'active_grid')])
        else:
            result = xr.combine_by_coords([result.to_dataset(name = 'nn_adjusted'), zeros_mask_test.to_dataset(name = 'active_grid')])


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

    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Autoencoder/run_set_2'
    out_dir    = f'{out_dir_x}/N15_M12_vPatternsOnly_North_arch2_batch100_e65_L1' 

    lead_months = 12
    bootstrap = False
    test_years = [2020]


    #################################################################################################
    obs_ref = out_dir_x.split('/')[-3]

    if obs_ref == 'NASA':
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/NASA*.nc')[0]
    else:
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]
    
    observation = xr.open_dataset(data_dir_obs)['SICN']
    ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
    fct = xr.concat(ls, dim = 'time').sortby('time')
    ##################################################################################################
    
    params = extract_params(out_dir)
    print(f'loaded configuration: \n')
    for key, values in params.items():
        print(f'{key} : {values} \n')
    
    try:
        version = int(out_dir.split('/')[-1].split('_')[2][1])
    except:
        version = (out_dir.split('/')[-1].split('_')[2][1:])
      
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
        predict(fct, observation, params, lead_months, out_dir,  test_years, model_year = 2020, ensemble_mode='Mean',  save=True)

    print(f'Output dir: {out_dir}')
    print('Saved!')

