import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import dask
import xarray as xr
from pathlib import Path
from torch.distributions import Normal
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.cvae import cVAE
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




def run_training(params, n_years, lead_months, lead_time = None, NPSProj = False,test_years = None,  n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, save = False):
    if lead_time is not None:
        assert lead_time <=lead_months, f"{lead_time} can not be greater than {lead_months}"

    
    if NPSProj:
        crs = 'NPS'  
    else: 
        crs = '1x1'

    if obs_ref == 'NASA':
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ f'/NASA*{crs}*.nc')[0] 
    else:
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

    assert params['version'] in [1,2,3, 'IceExtent']

  
    if params['version'] == 2:

        params['forecast_preprocessing_steps'] = [
            ('anomalies', AnomaliesScaler_v1_seasonal())]
        params['observations_preprocessing_steps'] = []
        
    else:
        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
    #####################################################################################################
    ### if you used Standardizer make sure to pass VAE = True as an argument to the initializer below ###
    if params['version'] == 3:
        print(' Warning!!! If you used Standardizer as a preprocessing step make sure to pass "VAE = True" as an argument to the initializer!!!')
    #####################################################################################################

    # if params['version'] == 'IceExtent':
    #     params['reg_scale'] = None

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

    if params['version'] == 3:

        params['forecast_preprocessing_steps'] = []
        params['observations_preprocessing_steps'] = []
        ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/bias_adjusted_North_1983-2020_1x1.nc')['SICN'].clip(0,1)
        if ensemble_list is not None:
            raise RuntimeError('With version 3 you are reading the bias adjusted ensemble mean as input. Set ensemble_list to None to proceed.')

    else:

        if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
            print("Load forecasts")
            ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*{crs}*.nc')[0])['SICN'] for intial_month in range(1,13) ]
            ds_in = xr.concat(ls, dim = 'time').sortby('time').sel(ensembles = ensemble_list)
            if ensemble_mode == 'Mean': 
                ds_in = ds_in.mean('ensembles') 
            else:
                ds_in = ds_in.transpose('time','lead_time','ensembles',...)
                print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')

        else:    ## Load specified members
            print("Load forecasts") 
            ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*{crs}*.nc')[0])['SICN'].mean('ensembles').load() for intial_month in range(1,13) ]
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
    
    subset_dimensions = params["subset_dimensions"]

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
    ################################### apply the mask #######################
    if test_years is None:
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
   
    reg_scale = params["reg_scale"]
    model = params["model"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
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

    # if params['version'] == 'IceExtent':
    #     obs_raw = obs_raw.where(obs_raw>=0.15,0)
    #     obs_raw = obs_raw.where(obs_raw ==0 , 1)
    #     # ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean>=0.15,0)
    #     # ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean ==0 , 1)
    #     params['loss_function'] = 'BCELoss'
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
    ###################################################################################
    if n_runs > 1:
        numpy_seed = None
        torch_seed = None

    with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
        f.write(
            f"model\t{model.__name__}\n" +
            f"reg_scale\t{reg_scale}\n" +  ## PG: The scale to be passed to Signloss regularization
            f"loss_function\t{params['loss_function']}\n" + 
            f"time_features\t{time_features}\n" +
            f"obs_clim\t{obs_clim}\n" +
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"ensemble_mode\t{ensemble_mode}\n" + ## PG: Ensemble list
            f"ensemble_features\t{ensemble_features}\n" + ## PG: Ensemble features
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{0.001}\n" +
            f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n" + 
            f"decoder_kernel_size\t{decoder_kernel_size}\n" +
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"loss_region\t{loss_region}\n" +
            f"active_grid\t{active_grid}\n" + 
            f"low_ress_loss\t{low_ress_loss}\n" +
            f"equal_weights\t{params['equal_weights']}\n" + 
            f"subset_dimensions\t{subset_dimensions}\n" + 
            f"L2_reg\t{l2_reg}\n" + 
            f"loss_reduction\t{params['loss_reduction']}\n" + 
            f"VAE_latent_size\t{params['VAE_latent_size']}\n"  + 
            f"scale_factor_channels\t{params['scale_factor_channels']}\n"  + 
            f"VAE_MLP_encoder\t{params['VAE_MLP_encoder']}\n"  + 
            f"training_sample_size\t{params['training_sample_size']}\n" +
            f"hybrid_weight\t{params['hybrid_weight']}\n"
        )
    del ds_raw
    gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in range(n_runs):
        print(f"Start run {run + 1} of {n_runs}...")
        # if lead_time is not None:
            # yearly_results = []
        for y_idx, test_year in enumerate(test_years):
            print(f"Start run for test year {test_year}...")
            monthly_results = []
            monthly_results_deterministic = []
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
                train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...]) if lead_time is None else create_mask(full_shape[:n_train,...])[:, lead_time - 1][..., None] ############

                ds_baseline = ds_raw_ensemble_mean[:n_train - month + 1,...]
                obs_baseline = obs_raw[:n_train - month + 1 ,...].isel(channels = slice(0,1))

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
                obs = obs_pipeline.transform(obs_raw.isel(channels = slice(0,1)))

                if params['combined_prediction']:
                    obs = xr.concat([obs, obs_raw.isel(channels = slice(1,2))], dim  = 'channels')  

                step_arguments = {'anomalies' : [lead_time, month]} if 'anomalies' in obs_pipeline.steps else None
                del ds_baseline, obs_baseline, preprocessing_mask_obs, preprocessing_mask_fct
                gc.collect()

                if params['version']  in [3]:
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
                
                if NPSProj:
                    weights = (np.ones_like(ds_train.lon) * (np.ones_like(ds_train.lat.to_numpy()))[..., None])  # Moved this up
                    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon})
                    weights = weights * land_mask
                    weights_ = weights * land_mask
                else:
                    weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
                    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove()  
                    ####################################################################
                    weights_ = weights * land_mask
                    if params['equal_weights']:
                        weights = xr.ones_like(weights)
                    # if any(['land_mask' not in time_features, model not in [cVAE]]):
                    weights = weights * land_mask

                if loss_region is not None:
                    loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region, flat = False)  ### the function has to be editted for flat opeion!!!!! 
                
                else:
                    loss_region_indices = None


                del ds, obs
                gc.collect()
                torch.cuda.empty_cache() 
                torch.cuda.synchronize() 
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


                net = cVAE(VAE_latent_size = params['VAE_latent_size'], n_channels_x= n_channels_x+ add_feature_dim , sigmoid = sigmoid_activation, NPS_proj = NPSProj, device=device, combined_prediction = params['combined_prediction'], VAE_MLP_encoder = params['VAE_MLP_encoder'], scale_factor_channels = params['scale_factor_channels'])

                net.to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
                if params['lr_scheduler']:
                    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=params['start_factor'], end_factor=params['end_factor'], total_iters=params['total_iters'])

                ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
                if lead_time is not None:
                    mask =  create_mask(full_shape)[:n_train]  #create_mask(ds)[:n_train]
                else:
                    mask = train_mask
                
                train_set = XArrayDataset(ds_train, obs_train, mask=mask, zeros_mask = zeros_mask, in_memory=False, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features, aligned = True, month_min_max = month_min_max, model = 'UNet2') 
                dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


                # if reg_scale is None: ## PG: if no penalizing for negative anomalies
                if low_ress_loss:
                    criterion = WeightedMSEKLDLowRess(weights=weights, device=device, hyperparam=1, reduction=params['loss_reduction'], loss_area=loss_region_indices, scale=reg_scale)
                else:
                    criterion = WeightedMSEKLD(weights=weights, device=device, hyperparam=1, reduction=params['loss_reduction'], loss_area=loss_region_indices, scale=reg_scale)
                # else:
                #     if low_ress_loss:
                #         criterion = WeightedMSEGlobalLossLowRess(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
                #     else:
                #         criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = True)
                if params['combined_prediction']:
                    criterion_extent = nn.BCELoss()

                epoch_loss = []
                epoch_MSE = []
                epoch_KLD = []
                net.train()
                num_batches = len(dataloader)
                step = 0

                model_mask_ = torch.from_numpy(model_mask.to_numpy()).unsqueeze(0)#.expand(n_channels_x + add_feature_dim,*model_mask.shape)  ## Uncomment if multi_channel is True
                obs_mask = torch.from_numpy(land_mask.to_numpy()).unsqueeze(0) ## Uncomment if multi_channel is True

                for epoch in tqdm.tqdm(range(epochs)):
                    batch_loss = 0
                    batch_loss_MSE = 0
                    batch_loss_KLD = 0
                    for batch, (x, y) in enumerate(dataloader):

                        if type(params['beta']) == dict:
                            if epoch + 1< params['beta']['start_epoch']:
                                beta = params['beta']['start']
                            else:
                                range_epochs = (params['beta']['end_epoch'] - params['beta']['start_epoch'] + 1)*num_batches
                                step_beta = np.clip((step - (params['beta']['start_epoch'] - 1)* num_batches) /(range_epochs),a_min = 0, a_max = None)
                                beta = params['beta']['start'] + (params['beta']['end'] - params['beta']['start']) * min(step_beta, 1)

                        else:
                            beta = params['beta']
                        step = step +1

                        if (type(x) == list) or (type(x) == tuple):
                            x = (x[0].to(device), x[1].to(device))
                            model_mask_ = model_mask_.to(x[0])
                        else:
                            x = x.to(device)
                            model_mask_ = model_mask_.to(x)
                        
                        if (type(y) == list) or (type(y) == tuple):
                            y, m = (y[0].to(device), y[1].to(device))
                        else:
                            y = y.to(device)
                            m  = None

                        optimizer.zero_grad()
                        obs_mask = obs_mask.to(y)#.expand_as(y[0])   ## Uncomment if multi_channel is True
                        generated_output, _, mu, log_var , cond_mu, cond_log_var = net(y, obs_mask, x, model_mask_, sample_size = params['training_sample_size'] )
                        if params['hybrid_weight'] is not None:
                            generated_output_GCGN, _, _, _ , _, _ = net(y, obs_mask, x, model_mask_, sample_size = params['training_sample_size'], mode = 'GCGN' )   

                        if params['combined_prediction']:
                            (y, y_extent) = (y[:,0].unsqueeze(1), y[:,1].unsqueeze(1))
                            (generated_output, generated_output_extent) = generated_output
                            loss_extent = criterion_extent(generated_output_extent, y_extent.unsqueeze(0).expand_as(generated_output_extent))
                            if params['hybrid_weight'] is not None:
                                (generated_output_GCGN, generated_output_GCGN_extent) = generated_output_GCGN
                                loss_extent_GCGN = criterion_extent(generated_output_GCGN_extent, y_extent.unsqueeze(0).expand_as(generated_output_extent))
                                loss_extent = loss_extent * params['hybrid_weight'] + ( 1- params['hybrid_weight']) * loss_extent_GCGN
                                del generated_output_GCGN_extent
                            del generated_output_extent

                        loss, MSE, KLD = criterion(generated_output, y.unsqueeze(0).expand_as(generated_output) ,mu, log_var, cond_mu = cond_mu, cond_log_var = cond_log_var ,beta = beta, mask = m, return_ind_loss=True , print_loss = True)
                        if params['hybrid_weight'] is not None:
                            loss_GCGN = criterion(generated_output_GCGN, y.unsqueeze(0).expand_as(generated_output) , mask = m , return_ind_loss=False , print_loss = False)
                            print(f'GCGN : {loss_GCGN}')
                            loss = loss * params['hybrid_weight'] + ( 1- params['hybrid_weight']) * loss_GCGN
                            del generated_output_GCGN
                         
                        if params['combined_prediction']:
                            loss = loss + loss_extent

                        batch_loss += loss.item()
                        batch_loss_MSE += MSE.item()
                        batch_loss_KLD += KLD.item()
                        loss.backward()
                        optimizer.step()
                    epoch_loss.append(batch_loss / num_batches)
                    epoch_MSE.append(batch_loss_MSE / num_batches)
                    epoch_KLD.append(batch_loss_KLD / num_batches)

                    if params['lr_scheduler']:
                        scheduler.step()
                del train_set, dataloader, ds_train, obs_train, generated_output, x, y , m, mu, log_var , cond_mu, cond_log_var 

                gc.collect()
                # EVALUATE MODEL
                ##################################################################################################################################
                if test_year*100 + month <= ds_raw_ensemble_mean.time[-1]:
                    
                    if save:
                        nameSave = f"MODEL_V{params['version']}_198101-{test_year * 100 + month -1}.pth"
                        torch.save( net.state_dict(),results_dir + '/' + nameSave)
                    test_years_list = np.arange(1, ds_test.shape[0] + 1)
                    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)

        
                    ## PG: Extract the number of years as well 
                    test_set = XArrayDataset(ds_test, obs_test, lead_time=lead_time,mask = None,zeros_mask = zeros_mask, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max, model = 'UNet2')
                    # if params['version'] == 'IceExtent':   
                    #     criterion_test = nn.BCELoss()
                    # else:
                    if low_ress_loss:
                        criterion_test =  WeightedMSELowRess(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
                    else:    
                        criterion_test =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)


                    test_loss = np.zeros(shape=(test_set.target.shape[0]))
                    target_ens = xr.concat([test_set.target.expand_dims('ensembles', axis = 0).isel(channels = 0) for _ in range(params['BVAE'])], dim = 'ensembles')

                    test_results = np.zeros_like(target_ens.values)
                    test_results_deterministic = np.zeros_like(test_set.target.values)

                    results_shape = xr.full_like(target_ens, fill_value = np.nan)
                    results_shape_deterministic = xr.full_like(test_set.target, fill_value = np.nan)
                    del target_ens

                    if params['combined_prediction']:
                        test_results_extent = test_results.copy()
                        results_shape_extent = results_shape.copy()
                        test_results_deterministic_extent = test_results_deterministic.copy()
                        results_shape_deterministic_extent = results_shape_deterministic.copy()


                    if active_grid:
                        zeros_mask_test = results_shape.copy()
                        zeros_mask_test[:] = test_set.zeros_mask
                        zeros_mask_test = zeros_mask_test.unstack('flattened').transpose('time','lead_time',...)

                    for i, (x, target) in enumerate(test_set): 
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
                            del x, target
                            _,deterministic_output, _, _, cond_mu, cond_log_var = net(test_obs, obs_mask, test_raw, model_mask_, sample_size = 1 )
                            basic_unet = net.unet(test_raw, model_mask_)
                            cond_var = torch.exp(cond_log_var) + 1e-4
                            cond_std = torch.sqrt(cond_var)

                            z =  Normal(cond_mu, cond_std).rsample(sample_shape=(params['BVAE'],)).squeeze().to(device)
                            # z = torch.unflatten(z , dim = -1, sizes = cond_std.shape[-3:])
                            out = net.generation(z) + basic_unet.squeeze() 
                            generated_output = net.last_conv(out)
                            
                            if params['combined_prediction']:
                                generated_output_extent = net.last_conv2(out)
                                (deterministic_output, deterministic_output_extent) = deterministic_output
                                (test_obs, test_obs_extent) = (test_obs[:,0].unsqueeze(1), test_obs[:,1].unsqueeze(1))
                                test_results_extent[:,i,] = generated_output_extent.to(torch.device('cpu')).numpy()
                                test_results_deterministic_extent[i,] = deterministic_output_extent.squeeze().to(torch.device('cpu')).numpy()
                            del z, out
                            if m is not None:
                                m[m != 0] = 1
                            # if params['version'] == 'IceExtent':
                            #     loss = criterion_test(generated_output, test_obs)
                            # else:
                            loss = criterion_test(torch.mean(generated_output, 0), test_obs, mask = m)

                            test_results[:,i,] = generated_output.to(torch.device('cpu')).numpy()
                            test_results_deterministic[i,] = deterministic_output.squeeze().to(torch.device('cpu')).numpy()
                            test_loss[i] = loss.item()
                    del  test_set , test_raw, test_obs, m, deterministic_output,  generated_output , ds_test, obs_test, cond_mu, cond_var, cond_std, basic_unet
                    gc.collect()

                    ###################################################### has to be eddited for large ensembles!! #####################################################################
                    results_shape[:] = test_results[:]
                    results_shape_deterministic[:] = test_results_deterministic[:]

                    test_results = results_shape.unstack('flattened').transpose('time','lead_time',...)
                    test_results_deterministic = results_shape_deterministic.unstack('flattened').transpose('time','lead_time',...)

                    test_results_untransformed = obs_pipeline.inverse_transform(test_results.values, step_arguments)
                    test_results_untransformed_deterministic = obs_pipeline.inverse_transform(test_results_deterministic.values, step_arguments)

                    result = xr.DataArray(test_results_untransformed, test_results.coords, test_results.dims, name='nn_adjusted')
                    result_deterministic = xr.DataArray(test_results_untransformed_deterministic, test_results_deterministic.coords, test_results_deterministic.dims, name='nn_adjusted')

                    if params['combined_prediction']:
                        results_shape_extent[:] = test_results_extent[:]
                        result_extent = results_shape_extent.unstack('flattened').transpose('time','lead_time',...)

                        results_shape_deterministic_extent[:] = test_results_deterministic_extent[:]
                        result_extent_deterministic = results_shape_deterministic_extent.unstack('flattened').transpose('time','lead_time',...)
                        del results_shape_extent, results_shape_deterministic_extent


                    if obs_clim:
                        result = result.isel(channels = 0).expand_dims('channels', axis=2)

                    del  results_shape, test_results, test_results_untransformed,  results_shape_deterministic, test_results_deterministic, test_results_untransformed_deterministic

                    gc.collect()
                    result = (result * land_mask)
                    result_deterministic = (result_deterministic * land_mask)

                    if not NPSProj:
                            result = reverse_pole_centric(result, subset_dimensions)
                            result_deterministic = reverse_pole_centric(result_deterministic, subset_dimensions)

                    ##############################################################################################################################################################
                    result = result.to_dataset(name = 'nn_adjusted') 
                    result_deterministic = result_deterministic.to_dataset(name = 'nn_adjusted')  

                    # if params['version'] == 'IceExtent':
                        
                    #     result = result.where(result >= 0.5, 0)
                    #     result = result.where(result ==0, 1)

                    #     result_deterministic = result_deterministic.where(result_deterministic >= 0.5, 0)
                    #     result_deterministic = result_deterministic.where(result_deterministic ==0, 1)

                    if active_grid:
                        if not NPSProj:
                            zeros_mask_test = reverse_pole_centric(zeros_mask_test)
                        else:
                            zeros_mask_test = zeros_mask_test.rename({'lon':'x', 'lat':'y'})
                        result = xr.combine_by_coords([result * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])
                        result_deterministic = xr.combine_by_coords([result_deterministic * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])

                    if params['combined_prediction']:
                        result_extent = (result_extent * land_mask)
                        result_extent_deterministic = (result_extent_deterministic * land_mask)
                        if not NPSProj:
                            result_extent = reverse_pole_centric(result_extent, subset_dimensions)
                            result_extent_deterministic = reverse_pole_centric(result_extent_deterministic, subset_dimensions)

                        result_extent = result_extent.to_dataset(name = 'nn_adjusted_extent')
                        result_extent = result_extent.where(result_extent >= 0.5, 0)
                        result_extent = result_extent.where(result_extent ==0, 1)
                        result = xr.combine_by_coords([result , result_extent])

                        result_extent_deterministic = result_extent_deterministic.to_dataset(name = 'nn_adjusted_extent')
                        result_extent_deterministic = result_extent_deterministic.where(result_extent_deterministic >= 0.5, 0)
                        result_extent_deterministic = result_extent_deterministic.where(result_extent_deterministic ==0, 1)
                        result_deterministic = xr.combine_by_coords([result_deterministic , result_extent_deterministic])


                    monthly_results.append(result)
                    monthly_results_deterministic.append(result_deterministic)
                    
                    fig, ax = plt.subplots(1,1, figsize=(8,5))
                    ax.plot(np.arange(1,epochs+1), epoch_loss, label = 'Epoch loss total')
                    ax.plot(np.arange(1,epochs+1), epoch_MSE, linestyle = 'dashed', label = 'Epoch MSE only')
                    ax.plot(np.arange(1,epochs+1), epoch_KLD, linestyle = 'dotted', label = 'Epoch KLD')

                    ax.set_title(f'Train Loss \n test loss (SIC only): {np.mean(test_loss)}') ###
                    ax.legend()
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    plt.show()
                    plt.savefig(results_dir+f'/Figures/train_loss_198101-{test_year * 100 + month -1}.png')
                    plt.close()


                    del result,result_deterministic, net, optimizer
                    gc.collect()
                    torch.cuda.empty_cache() 
                    torch.cuda.synchronize() 
                else:
                    if lead_time is not None:
                        nameSave = f"MODEL_final_V{params['version']}_198101-{int(ds_raw_ensemble_mean.time[-lead_time])}.pth"
                    else:
                        nameSave = f"MODEL_final_V{params['version']}_198101-{int(ds_raw_ensemble_mean.time[-1])}.pth"
                    # Save locally
                    torch.save( net.state_dict(),results_dir + '/' + nameSave)
                    break
            if len(monthly_results) >0 :
                # if lead_time is None:
                    xr.concat(monthly_results, dim = 'time').to_netcdf(path=Path(results_dir, f'nn_adjusted_ENS_{test_year}_{run+1}.nc', mode='w'))
                    xr.concat(monthly_results_deterministic, dim = 'time').to_netcdf(path=Path(results_dir, f'nn_adjusted_deterministic_{test_year}_{run+1}.nc', mode='w'))
                # else:
                #     yearly_results.append(xr.concat(monthly_results, dim = 'time'))
            del monthly_results, monthly_results_deterministic
            gc.collect() 
            torch.cuda.empty_cache() 
            torch.cuda.synchronize() 
            
        # if lead_time is not None:
        #     xr.concat(yearly_results, dim = 'time').to_netcdf(path=Path(results_dir, f'nn_adjusted_lead_time_{lead_time}_{int(ds_raw_ensemble_mean.time[0])}-{int(ds_raw_ensemble_mean.time[-1])}_{run+1}.nc', mode='w'))
    

if __name__ == "__main__":

  
    n_years =  4 # last n years to test consecutively
    lead_months = 12
    lead_time = None ## None for training using all available lead_times as indicated ny lead_months
    n_runs = 1  # number of training runs

    params = {
        "model": cVAE,
        "time_features": ['month_sin','month_cos', 'imonth_sin', 'imonth_cos'],
        "obs_clim" : False,
        "ensemble_features": False, ## PG
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 50,
        "batch_size": 10,
        "reg_scale" : None,
        "beta" : 1,
        "optimizer": torch.optim.Adam,
        "lr": 0.001 ,
        "loss_function" :'MSE',
        "loss_region": None,
        "subset_dims": 'North',   ## North or South or Global
        'active_grid' : False,
        'low_ress_loss' : False,
        'equal_weights' : False,
        "decoder_kernel_size" : 1, ## only for (Reg)CNN
        "L2_reg": 0,
        'lr_scheduler' : False,
        'VAE_latent_size' : 25,
        'VAE_MLP_encoder' : False,
        'scale_factor_channels' : None,
        'BVAE' : 50,
        'training_sample_size' : 1, 
        'loss_reduction' : 'mean' , # mean or sum
        'combined_prediction' : False,
        'hybrid_weight' : None, 
    }



    params['version'] =  1  ### 1 , 2 ,3 , 'IceExtent'
    params['forecast_range_months'] = 12
    params['beta'] =   dict(start = 0, end =0.5, start_epoch = 10 , end_epoch = params['epochs'])  

    obs_ref = 'NASA'
    NPSProj = True
    
    out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/{obs_ref}/{params["model"].__name__}/run_set_2_convnext'
    if type(params['beta']) == dict:
        beta_arg = 'Banealing'
    else:
        beta_arg = f'B{params["beta"]}'
        
    # for lead_time in np.arange(1,13):
    print( f'Training lead_time {lead_time} ...')

    if params['hybrid_weight'] is not None:
        model_type = 'CGNhybrid'
    else:
        model_type = 'CVAE'

    if lead_time is None:
        out_dir    = f'{out_dir_x}/N{n_years}_M{lead_months}_F{params["forecast_range_months"]}_v{params["version"]}_{beta_arg}_batch{params["batch_size"]}_e{params["epochs"]}_Cscale{params["scale_factor_channels"]}_{model_type}_{params["BVAE"]}-{params["training_sample_size"]}_LS{params["VAE_latent_size"]}'
    else:
        out_dir    = f'{out_dir_x}/N{n_years}_LT{lead_time}_F{params["forecast_range_months"]}_v{params["version"]}_{beta_arg}_batch{params["batch_size"]}_e{params["epochs"]}_Cscale{params["scale_factor_channels"]}_{model_type}_{params["BVAE"]}-{params["training_sample_size"]}_LS{params["VAE_latent_size"]}'
    
    if params['VAE_MLP_encoder']:
        out_dir = out_dir + '_Linear'

    out_dir = out_dir + '_NPSproj' if NPSProj else out_dir + '_1x1'

    if params['lr_scheduler']:
        out_dir = out_dir + '_lr_scheduler'
        params['start_factor'] = 1.0
        params['end_factor'] = 0.1
        params['total_iters'] = params['epochs']


    out_dir  = out_dir + f'_{params["subset_dims"]}_lr{params["lr"]}_batch{params["batch_size"]}_e{params["epochs"]}_L{params["reg_scale"]}'

    if params['subset_dims'] == 'Global':
        params['subset_dimensions'] = None
    else:
        params['subset_dimensions'] = params['subset_dims']

    if params['active_grid']:
        out_dir = out_dir + '_active_grid'
    if params['combined_prediction']:
        out_dir = out_dir + '_combined'  

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)
    
    run_training(params, n_years=n_years, lead_months=lead_months,lead_time = lead_time, NPSProj  = NPSProj, test_years = None, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1, save = True)

    print(f'Output dir: {out_dir}')
    print('Training done.')



