import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.convlstm import  CNNLSTM_monthly, UNetLSTM_monthly
from losses import WeightedMSE, WeightedMSEGlobalLoss
from losses import WeightedMSELowRess , WeightedMSEGlobalLossLowRess
from preprocessing import align_data_and_targets, create_mask, pole_centric, reverse_pole_centric, segment, reverse_segment
from preprocessing import AnomaliesScaler_v1_seasonal, AnomaliesScaler_v2_seasonal, Standardizer, Normalizer, PreprocessingPipeline, calculate_climatology, bias_adj, zeros_mask_gen
from torch_datasets import XArrayDataset
import torch.nn as nn
# from subregions import subregions
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
import glob
# specify data directories
data_dir_forecast = LOC_FORECASTS_SI




def run_training(params, n_years, lead_time, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, save = False):
    


    # if params['model'] in [PNet]:
    #     params['kernel_size'] = None
    #     params['decoder_kernel_size'] = None
    #     params['hidden_dims'] = None
        
    if obs_ref == 'NASA':
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/NASA*.nc')[0]
    else:
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

    assert params['version'] in [1,2,3, 'PatternsOnly', 'IceExtent']

    if params['version'] == 'PatternsOnly':
        params["obs_clim"] = False
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
    
    elif params['version'] == 3:

        params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1_seasonal())]
        params['observations_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v2_seasonal()) ]
        
    else:
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []

    if params['version'] == 'IceExtent':
        params['reg_scale'] = None

    if params['lr_scheduler']:
        start_factor = params['start_factor']
        end_factor = params['end_factor']
        total_iters = params['total_iters']
    else:
        start_factor = end_factor = total_iters = None
    
    if params['low_ress_loss']:
        params['active_grid'] = False
        print('Warning: active_grid turned off because low_ress_loss is on!')

    print("Start training")
    print("Load observations")

    obs_in = xr.open_dataset(data_dir_obs)['SICN']
    
    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##
    ensemble_features = params['ensemble_features']

    if params['version'] == 2:

        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
        ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/bias_adjusted_North_1983-2020.nc')['SICN']
        if ensemble_list is not None:
            raise RuntimeError('With version 2 you are reading the bias adjusted ensemble mean as input. Set ensemble_list to None to proceed.')

    else:

        if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
            print("Load forecasts")
            ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
            ds_in = xr.concat(ls, dim = 'time').sortby('time').sel(ensembles = ensemble_list)
            if ensemble_mode == 'Mean': 
                ds_in = ds_in.mean('ensembles') 
            else:
                ds_in = ds_in.transpose('time','lead_time','ensembles',...)
                print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')

        else:    ## Load specified members
            print("Load forecasts") 
            ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
            ds_in = xr.concat(ls, dim = 'time').mean('ensembles').sortby('time')
   
    ###### handle nan and inf over land ############
     ### land is masked in model data with a large number
    ds_in = ds_in.where(ds_in<1000,np.nan)
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

    ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_time)  # extract valid lead times and usable years

    if not ds_raw.time.equals(obs_raw.time):
            
            ds_raw = ds_raw.sel(time = obs_raw.time)
    
    if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time','ensembles',...)
    else:
        ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time',...)
    
    subset_dimensions = params["subset_dimensions"]

    if subset_dimensions is not None:
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
    # land_mask = land_mask.where(model_mask == 1, 0)
    obs_raw = obs_raw * land_mask
    ds_raw_ensemble_mean = ds_raw_ensemble_mean * land_mask
    ################################### apply the mask #######################

    test_years = np.arange( int(ds_raw_ensemble_mean.time[-1]/100 - n_years + 1), int(ds_raw_ensemble_mean.time[-1]/100) + 2)
    if np.mod(ds_raw_ensemble_mean.time[-1],100) <12:
        test_years = test_years[:-1]

    if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
        zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(lead_time = 0).drop('lead_time').where(obs_raw.time<test_year*100, drop = True ), 3) for test_year in test_years], dim = 'test_year').assign_coords(test_year = test_years)           
        
        for item in ['active_mask', 'full_ice_mask']:
            zeros_mask_full = zeros_mask_full.drop(item) if item not in params["time_features"] else zeros_mask_full
        zeros_mask_full = zeros_mask_full.drop('active_grid') if not params['active_grid'] else zeros_mask_full

        zeros_mask_full = zeros_mask_full.expand_dims('channels', axis=-3)
        if 'ensembles' in ds_raw.dims:
             zeros_mask_full = zeros_mask_full.expand_dims('ensembles', axis=2)

    if params['version'] == 'PatternsOnly':

            weights = np.cos(np.ones_like(ds_raw_ensemble_mean.lon) * (np.deg2rad(ds_raw_ensemble_mean.lat.to_numpy()))[..., None])  # Moved this up
            weights = xr.DataArray(weights, dims = ds_raw_ensemble_mean.dims[-2:], name = 'weights').assign_coords({'lat': ds_raw_ensemble_mean.lat, 'lon' : ds_raw_ensemble_mean.lon}) 
            weights = weights * land_mask
            ds_raw_mean = ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
            obs_raw_mean = ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
            ds_raw_ensemble_mean = ds_raw_ensemble_mean - ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
            obs_raw = obs_raw - ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))

        
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
    l2_reg = params["L2_reg"]
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]

    loss_region = params["loss_region"]
    
    obs_clim = params["obs_clim"]
    active_grid = params['active_grid']
    low_ress_loss = params['low_ress_loss']
    
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
        params['loss_function'] = 'BCELoss'

   
    
    ds_raw_ensemble_mean = pole_centric(ds_raw_ensemble_mean, subset_dimensions)
    obs_raw =  pole_centric(obs_raw, subset_dimensions)
    land_mask = pole_centric(land_mask, subset_dimensions)
    if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
        zeros_mask_full = pole_centric(zeros_mask_full, subset_dimensions)


    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
        f.write(
            f"model\t{model.__name__}\n" +
            f"bilinear\t{params['bilinear']}\n" +
            f"reg_scale\t{reg_scale}\n" +  ## PG: The scale to be passed to Signloss regularization
            f"hidden_dims\t{hidden_dims}\n" +
            f"loss_function\t{params['loss_function']}\n" + 
            f"time_features\t{time_features}\n" +
            f"obs_clim\t{obs_clim}\n" +
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"ensemble_features\t{ensemble_features}\n" + ## PG: Ensemble features
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{0.001}\n" +
            f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n"
            f"kernel_size\t{kernel_size}\n" +
            f"decoder_kernel_size\t{decoder_kernel_size}\n" +
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"loss_region\t{loss_region}\n" +
            f"active_grid\t{active_grid}\n" + 
            f"low_ress_loss\t{low_ress_loss}\n" +
            f"equal_weights\t{params['equal_weights']}\n" + 
            f"subset_dimensions\t{subset_dimensions}\n" + 
            f"L2_reg\t{l2_reg}"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in range(n_runs):
        print(f"Start run {run + 1} of {n_runs}...")
        test_year_results = []
        for y_idx, test_year in enumerate(test_years):
            print(f"Start run for test year {test_year}...")
            monthly_results = []
            for month in range(1,13,params['forecast_range_months']):
                if test_year * 100 + month > ds_raw_ensemble_mean.time[-1]:
                    test_year, month = np.divmod(int(ds_raw_ensemble_mean.time[-1].values),  100)
                    test_year = test_year + np.divmod(month+1,13)[0]
                    month = np.divmod(month+1,13)[1]
                    print(f"\tStart run the final model ...")
                else:
                    print(f"\tStart run month {month} - {month + params['forecast_range_months'] - 1}...")

                if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                    zeros_mask = zeros_mask_full.sel(test_year = test_year).drop('test_year')
                else:
                    zeros_mask = None
                

                train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time < test_year * 100 + month].to_numpy()
                n_train = len(train_years)
                train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...])

                ds_baseline = ds_raw_ensemble_mean[:n_train - month + 1,...]
                obs_baseline = obs_raw[:n_train - month + 1 ,...]
                if params['version'] == 'PatternsOnly':
                        ds_baseline_mean = ds_raw_mean[:n_train - month + 1 ,...]
                        obs_baseline_mean = obs_raw_mean[:n_train - month + 1,...]


                if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
                    preprocessing_mask_fct = np.broadcast_to(train_mask[:n_train - month + 1,...,None,None,None,None], ds_baseline.shape)
                else:
                    preprocessing_mask_fct = np.broadcast_to(train_mask[:n_train - month + 1,...,None,None,None], ds_baseline.shape)
                preprocessing_mask_obs = np.broadcast_to(train_mask[:n_train - month + 1,...,None,None,None], obs_baseline.shape)


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
                obs = obs_pipeline.transform(obs_raw)
                step_arguments = {'anomalies' : [lead_time, month]} if 'anomalies' in obs_pipeline.steps else None

                if params['version']  in [3, 'PatternsOnly']:
                    sigmoid_activation = False
                else:
                    sigmoid_activation = True

                y0 = np.floor(ds[:n_train].time[0].values/100 )
                yr, mn = np.divmod(int(ds[:n_train+params['forecast_range_months']].time[-1].values - y0*100),100)
                month_min_max = [y0, yr * 12 + mn]

                if 'land_mask' in time_features:
                    ds = xr.concat([ds, land_mask.expand_dims('channels', axis = 0)], dim = 'channels')
                
                # TRAIN MODEL

                ds_train = ds[:n_train,...]
                obs_train = obs[:n_train,...]
                if test_year*100 + month <= ds_raw_ensemble_mean.time[-1]:
                        ds_test = ds[n_train:n_train + params['forecast_range_months'],...]
                        obs_test = obs[n_train:n_train + params['forecast_range_months'],...]

                if params['version'] == 'PatternsOnly':
                        if 'ensembles' in ds_baseline_mean.dims:  
                            mask_bias_adj = train_mask[:n_train - month + 1,...,None, None]
                        else:
                            mask_bias_adj = train_mask[:n_train - month + 1,...,None]

                        fct_climatology = calculate_climatology(ds_baseline_mean.where(~np.broadcast_to(mask_bias_adj, ds_baseline_mean.shape)))
                        obs_climatology = calculate_climatology(obs_baseline_mean.where(~np.broadcast_to(train_mask[:n_train - month + 1,...,None], obs_baseline_mean.shape)))
                        target_forecasts = ds_raw_mean[n_train:n_train + params['forecast_range_months'],...]
                        fct_bias_adjusted = xr.concat([target_forecasts[i] - fct_climatology.sel(init_month = month + i) + obs_climatology.sel(init_month = month + i) for i in range(target_forecasts.shape[0])], dim = 'time')
                         


                weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
                weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove()  
                ####################################################################
                weights_ = weights.copy() * land_mask
                if params['equal_weights']:
                    weights = xr.ones_like(weights)
                if 'land_mask' not in time_features:
                    weights = weights * land_mask

                if loss_region is not None:
                    loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region, flat = False)  ### the function has to be editted for flat opeion!!!!! 
                else:
                    loss_region_indices = None


                weights = weights.values
                weights_ = weights_.values

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


                # if model in [PNet]:
                #     net = model(  n_channels_x= n_channels_x+ add_feature_dim ,seq_length = lead_time, bilinear=params['bilinear'], sigmoid = sigmoid_activation, device =  device)
                if model in [CNNLSTM_monthly]:
                    net = model(  n_channels_x= n_channels_x, add_features = add_feature_dim ,hidden_dims = hidden_dims, seq_length = lead_time, kernel_size = kernel_size, decoder_kernel_size = decoder_kernel_size, sigmoid = sigmoid_activation, device =  device)
                if model in [UNetLSTM_monthly]:
                    net = model(  n_channels_x= n_channels_x+ add_feature_dim ,seq_length = lead_time, bilinear=params['bilinear'], sigmoid = sigmoid_activation, device =  device)


                net.to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
                if params['lr_scheduler']:
                    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=params['start_factor'], end_factor=params['end_factor'], total_iters=params['total_iters'])

                ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
                train_set = XArrayDataset(ds_train, obs_train, mask=create_mask(ds)[:n_train] ,zeros_mask = zeros_mask, in_memory=True, lead_time=None, time_features=time_features,ensemble_features =ensemble_features, aligned = True, month_min_max = month_min_max, model = model.__name__) 
                dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

                if params['version'] == 'IceExtent':
                        
                        criterion = nn.BCELoss()
                
                else:

                    if reg_scale is None: ## PG: if no penalizing for negative anomalies
                        if low_ress_loss:
                            criterion = WeightedMSELowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                        else:
                            criterion = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                    else:
                        if low_ress_loss:
                            criterion = WeightedMSEGlobalLossLowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
                        else:
                            criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)

                epoch_loss = []
                net.train()
                # torch.autograd.set_detect_anomaly(True) 
                num_batches = len(dataloader)
                for epoch in tqdm.tqdm(range(epochs)):
                    batch_loss = 0
                    for batch, (x, y) in enumerate(dataloader):
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
                        # with torch.cuda.amp.autocast(enabled=True):   
                        #     adjusted_forecast = net(x, ind = ind)
                        #     loss = criterion(adjusted_forecast, y, mask = m) 
                        #     if params['loss_function'] == 'RMSE': 
                        #         loss = torch.sqrt(loss)

                        # batch_loss += loss.item()
                        # grad_scaler.scale(loss).backward()
                        # grad_scaler.step(optimizer)
                        # grad_scaler.update()
                        adjusted_forecast = net(x)
                        if params['version'] == 'IceExtent':
                            loss = criterion(adjusted_forecast, y)
                        else:
                            loss = criterion(adjusted_forecast, y, mask = m)
                        if params['loss_function'] == 'RMSE': 
                            loss = torch.sqrt(loss)
                        batch_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        # for name, param in net.named_parameters():
                        #     if param.grad is not None:
                        #             print(f'Gradient - {name}: {param.grad.mean()}')
                        # print(epoch, batch,torch.isnan(adjusted_forecast).sum(),batch_loss )
                    epoch_loss.append(batch_loss / num_batches)

                    if params['lr_scheduler']:
                        scheduler.step()

                # EVALUATE MODEL
                
                ##################################################################################################################################
                if test_year*100 + month <= ds_raw_ensemble_mean.time[-1]:

                    ##################################################################################################################################
                    test_years_list = np.arange(1, ds_test.shape[0] + 1)
                    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)

        
                    ## PG: Extract the number of years as well 
                    test_set = XArrayDataset(ds_test, obs_test, lead_time=None,mask = None,zeros_mask = zeros_mask, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max, model = model.__name__)
                    if params['version'] == 'IceExtent':   
                        criterion_test = nn.BCELoss()
                    else:
                        if low_ress_loss:
                            criterion_test =  WeightedMSELowRess(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                        else:    
                            criterion_test =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)

                    if 'ensembles' in ds_test.dims:
                        test_loss = np.zeros(shape=(ds_test.transpose('time','ensembles','channels',...).shape[:2]))
                        test_results = np.zeros_like(ds_test.transpose('time','ensembles','channels',...).isel(lead_time = lead_time).data)
                        results_shape = xr.full_like(ds_test.transpose('time','ensembles','channels',...).isel(lead_time = lead_time), fill_value = np.nan)
                        test_time_list =  np.arange(1, results_shape.shape[0] + 1)
                    else:
                        test_loss = np.zeros(shape=(test_set.target.shape[0]))
                        test_results = np.zeros_like(test_set.target)
                        results_shape = xr.full_like(test_set.target, fill_value = np.nan)
                    

                    if active_grid:
                        if 'ensembles' in ds_test.dims: 
                            zeros_mask_test = results_shape.isel(ensembles = 0).copy()
                            zeros_mask_test[:] = test_set.zeros_mask[:len(test_time_list)]
                        else:
                            zeros_mask_test = results_shape.copy()
                            zeros_mask_test[:] = test_set.zeros_mask

                        zeros_mask_test = zeros_mask_test.transpose('time','lead_time','channels','lat','lon')


                    for i, (x, target) in enumerate(test_set): 
                        if 'ensembles' in ds_test.dims:  ## PG: If we have large ensembles:
        
                            ensemble_idx, j = np.divmod(i, len(test_time_list))  ## PG: find out ensemble index
                            net.eval()
                            with torch.no_grad():
                                if (type(x) == list) or (type(x) == tuple):     
                                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                                else:
                                    test_raw = x.unsqueeze(0).to(device)
                                if (type(target) == list) or (type(target) == tuple):
                                    test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                                else:
                                    test_obs = target.unsqueeze(0).to(device)
                                    m = None
                                test_adjusted = net(test_raw)
                                if m is not None:
                                    m[m != 0] = 1
                                if params['version'] == 'IceExtent':
                                    loss = criterion_test(test_adjusted, test_obs)
                                else:
                                    loss = criterion_test(test_adjusted, test_obs, mask = m)
                                if params['loss_function'] == 'RMSE': 
                                    loss = torch.sqrt(loss)
                                test_results[j,ensemble_idx,] = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
                                test_loss[j,ensemble_idx] = loss.item()
                        else:

                            net.eval()
                            with torch.no_grad():
                                if (type(x) == list) or (type(x) == tuple):
                                    test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
                                else:
                                    test_raw = x.unsqueeze(0).to(device)
                                if (type(target) == list) or (type(target) == tuple):
                                    test_obs, m = (target[0].unsqueeze(0).to(device), target[1].unsqueeze(0).to(device))
                                else:
                                    test_obs = target.unsqueeze(0).to(device)
                                    m = None
                                test_adjusted = net(test_raw)
                                if m is not None:
                                    m[m != 0] = 1
                                if params['version'] == 'IceExtent':
                                    loss = criterion_test(test_adjusted, test_obs)
                                else:
                                    loss = criterion_test(test_adjusted, test_obs, mask = m)
                                if params['loss_function'] == 'RMSE': 
                                    loss = torch.sqrt(loss)
                                test_results[i,] = test_adjusted.to(torch.device('cpu')).numpy()
                                test_loss[i] = loss.item()

                    ###################################################### has to be eddited for large ensembles!! #####################################################################
                    results_shape[:] = test_results[:]
                    
                    if model in [ CNNLSTM_monthly, UNetLSTM_monthly]:  ## PG: if the output is already a map

                        test_results = results_shape.transpose('time',...).expand_dims('lead_time', axis = 2)

                        test_results_untransformed = obs_pipeline.inverse_transform(test_results.values, step_arguments)
                        result = xr.DataArray(test_results_untransformed, test_results.coords, test_results.dims, name='nn_adjusted')


                    if obs_clim:
                        result = result.isel(channels = 0).expand_dims('channels', axis=2)

                    result = result * land_mask  
                    result = reverse_pole_centric(result, subset_dimensions)
                    ##############################################################################################################################################################
                    # Store results as NetCDF           
                    result = result.to_dataset(name = 'nn_adjusted') 
                    if params['version'] == 'PatternsOnly':
                        result = result + fct_bias_adjusted.to_dataset(name = 'nn_adjusted')
                        result = result * reverse_pole_centric(land_mask, subset_dimensions)
                        result = xr.combine_by_coords([result, fct_bias_adjusted.to_dataset(name = 'bias_adjusted')])
                        
                    if params['version'] == 'IceExtent':
                        
                        result = result.where(result >= 0.5, 0)
                        result = result.where(result ==0, 1)

                    if active_grid:

                        zeros_mask_test = reverse_pole_centric(zeros_mask_test)
                        result = xr.combine_by_coords([result * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])
                    
                    monthly_results.append(result)

                    fig, ax = plt.subplots(1,1, figsize=(8,5))
                    ax.plot(np.arange(1,epochs+1), epoch_loss)
                    ax.set_title(f'Train Loss \n test loss: {np.mean(test_loss)}') ###
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    plt.show()
                    plt.savefig(results_dir+f'/Figures/train_loss_198101-{test_year * 100 + month -1}.png')
                    plt.close()

                    if save:
                        nameSave = f"MODEL_V{params['version']}_198101-{test_year * 100 + month -1}.pth"
                        torch.save( net.state_dict(),results_dir + '/' + nameSave)
                else:
                    
                    nameSave = f"MODEL_final_V{params['version']}_198101-{int(ds_raw_ensemble_mean.time[-lead_time])}.pth"
                    # Save locally
                    torch.save( net.state_dict(),results_dir + '/' + nameSave)
                    break
            if len(monthly_results) >0 :
                result = xr.concat(monthly_results, dim = 'time')
                test_year_results.append(result )
        
        xr.concat(test_year_results, dim = 'time').to_netcdf(path=Path(results_dir, f'nn_adjusted_lead_time_{lead_time}_{int(ds_raw_ensemble_mean.time[0])}-{int(ds_raw_ensemble_mean.time[-1])}_{run+1}.nc', mode='w'))
        

if __name__ == "__main__":

  
    n_years =  15 # last n years to test consecutively
    lead_time = 6
    n_runs = 1  # number of training runs

    params = {
        "model": UNetLSTM_monthly,
        "hidden_dims": [64,128,128,64],# [16, 64, 128, 64, 32], ## only for SCNN and CNN
        "time_features": ['month_sin','month_cos', 'imonth_sin', 'imonth_cos'],
        "obs_clim" : False,
        "ensemble_features": False, ## PG
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 60,
        "batch_size": 10,
        "reg_scale" : None,
        "optimizer": torch.optim.Adam,
        "lr": 0.001 ,
        "loss_function" :'RMSE',
        "loss_region": None,
        "subset_dims": 'North',   ## North or South or Global
        'active_grid' : False,
        'low_ress_loss' : False,
        'equal_weights' : False,
        "DSC" : False,  ## only for SCNN 
        "kernel_size" : 5, ## only for SCNN and CNN
        "decoder_kernel_size" : 1, ## only for SCNN and CNN
        "bilinear" : False, ## only for UNet
        "L2_reg": 0,
        'lr_scheduler' : True
    }



    params['version'] = 2   ### 1 , 2 ,3, 'PatternsOnly' , 'IceExtent'
    params['reg_scale'] = None
    params["loss_function"] = 'MSE'
    params['forecast_range_months'] = 1

    obs_ref = 'NASA'

    
    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/{obs_ref}/{params["model"].__name__}/run_set_1'
    out_dir    = f'{out_dir_x}/N{n_years}_LT{lead_time}_F{params["forecast_range_months"]}_v{params["version"]}'

    if params['lr_scheduler']:
        out_dir = out_dir + '_lr_scheduler'
        params['start_factor'] = 1.0
        params['end_factor'] = 0.5
        params['total_iters'] = 100

    if params['model'] in  [CNNLSTM_monthly]:
        out_dir = out_dir + f'_{params["kernel_size"]}{params["decoder_kernel_size"]}'

            
    out_dir  = out_dir + f'_{params["subset_dims"]}_lr{params["lr"]}_batch{params["batch_size"]}_e{params["epochs"]}_L{params["reg_scale"]}'

    if params['subset_dims'] == 'Global':
        params['subset_dimensions'] = None
    else:
        params['subset_dimensions'] = params['subset_dims']

    if params['active_grid']:
        out_dir = out_dir + '_active_grid'
    if params['bilinear']:
        out_dir = out_dir + '_bilinear'


    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)

    run_training(params, n_years=n_years, lead_time=lead_time, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)

    print(f'Output dir: {out_dir}')
    print('Training done.')



