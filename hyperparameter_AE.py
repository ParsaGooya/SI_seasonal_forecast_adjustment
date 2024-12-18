# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tqdm

# import dask
# import xarray as xr
# from pathlib import Path
# import random

# import torch
# from torch.utils.data import DataLoader
# from torch.optim import lr_scheduler
# from models.autoencoder import Autoencoder
# from models.unet import UNet
# from losses import WeightedMSE, WeightedMSEOutlierLoss, IceextentlLoss, WeightedMSEGlobalLoss, GlobalLoss
# from preprocessing import align_data_and_targets, create_mask, config_grid
# from preprocessing import AnomaliesScaler_v1_seasonal,AnomaliesScaler_v2_seasonal, Standardizer, PreprocessingPipeline
# from torch_datasets import XArrayDataset
# # from subregions import subregions
# from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
# import glob
# # specify data directories
# data_dir_forecast = LOC_FORECASTS_SI

# def HP_congif(params, obs_ref, lead_months):

#     if params["model"] != Autoencoder:
#         params["append_mode"] = None
#     else:   
#         params["obs_clim"] = False

#     if params['model'] == UNet:
#         params['kernel_size'] = None
#         params['decoder_kernel_size'] = None
#         params['hidden_dims'] = None
#         params['arch'] = None

#     if params['model'] == UNet:
#             params["arch"] = '_default'

#     if obs_ref == 'NASA':
#         data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/NASA*.nc')[0]
#     else:
#         data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

#     print("Start training")
#     print("Load observations")
#     obs_in = xr.open_dataset(data_dir_obs)['SICN']
    
#     if params['version'] == 2:

#         params['forecast_preprocessing_steps'] = []
#         params['observations_preprocessing_steps'] = []
#         ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/global_mean_bias_adjusted_1983-2020.nc')['SICN']

#     else:
#         if params['ensemble_list'] is not None: ## PG: calculate the mean if ensemble mean is none
#             print("Load forecasts")
#             ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
#             ds_in = xr.concat(ls, dim = 'time').sortby('time').sel(ensembles = params['ensemble_list'])
#             if params['ensemble_mode'] == 'Mean': 
#                 ds_in = ds_in.mean('ensembles') 
#             else:
#                 ds_in = ds_in.transpose('time','lead_time','ensembles',...)
#                 print(f'Warning: ensemble_mode is {params["ensemble_mode"]}. Training for large ensemble ...')

#         else:    ## Load specified members
#             print("Load forecasts") 
#             ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
#             ds_in = xr.concat(ls, dim = 'time').mean('ensembles').sortby('time')
    
    # ###### handle nan and inf over land ############
    # ds_in = ds_in.where(ds_in<1000,np.nan)
    # land_mask = obs_in.mean('time').where(np.isnan(obs_in.mean('time')),1).fillna(0)
    # model_mask = ds_in.mean('time')[0].where(np.isnan(ds_in.mean('time')[0]),1).fillna(0).drop('lead_time')
    # obs_in = obs_in.clip(0,1)
    # ds_in = ds_in.clip(0,1)
    # obs_in = obs_in.fillna(0)
    # ds_in = ds_in.fillna(0)
    # ############################################

#     obs_in = obs_in.expand_dims('channels', axis=1)

#     if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
#         ds_in = ds_in.expand_dims('channels', axis=3)
#     else:
#         ds_in = ds_in.expand_dims('channels', axis=2) 


#     ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_months)  # extract valid lead times and usable years

#     if not ds_raw.time.equals(obs_raw.time):
            
#             ds_raw = ds_raw.sel(time = obs_raw.time)
    
#     if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
#         ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time','ensembles',...)
#     else:
#         ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time',...)

#     if params['subset_dimensions'] is not None:
#         if params['subset_dimensions'] == 'North':
#             ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat > 40, drop = True)
#             obs_raw = obs_raw.where(obs_raw.lat > 40, drop = True)
#             land_mask = land_mask.where(land_mask.lat > 40, drop = True)
#         else:
#             ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat < -40, drop = True)
#             obs_raw = obs_raw.where(obs_raw.lat < -40, drop = True)
#             land_mask = land_mask.where(land_mask.lat < -40, drop = True)

    # ################################### apply the mask #######################
    ## land_mask = land_mask.where(model_mask == 1, 0)
    # obs_raw = obs_raw * land_mask
    # ds_raw_ensemble_mean = ds_raw_ensemble_mean * land_mask
    # ################################### apply the mask #######################

#     if params['version'] == 'PatternsOnly':

#         weights = np.cos(np.ones_like(ds_raw_ensemble_mean.lon) * (np.deg2rad(ds_raw_ensemble_mean.lat.to_numpy()))[..., None])  # Moved this up
#         weights = xr.DataArray(weights, dims = ds_raw_ensemble_mean.dims[-2:], name = 'weights').assign_coords({'lat': ds_raw_ensemble_mean.lat, 'lon' : ds_raw_ensemble_mean.lon}) 
#         weights = weights * land_mask
#         ds_raw_ensemble_mean = ds_raw_ensemble_mean - ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
#         obs_raw = obs_raw - ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
#         params["obs_clim"] = None

#     if params['version'] in [3 , 'PatternsOnly']:    
#         params['sigmoid_activation'] = False

#     return ds_raw_ensemble_mean, obs_raw, params, land_mask


# def smooth_curve(list, factor = 0.9):
#     smoothed_list = []
#     for point in list:
#         if smoothed_list:
#             previous = smoothed_list[-1]
#             smoothed_list.append(previous* factor + point * (1- factor))
#         else:
#             smoothed_list.append(point)
#     return smoothed_list



# def training_hp(hyperparamater_grid: dict, params:dict, ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset , land_mask:XArrayDataset  ,test_year, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None):
    
#     assert params['version'] in [1,2, 3,'PatternsOnly']

    
#     if params['version'] == 'PatternsOnly':
        
#         params['forecast_preprocessing_steps'] = [ ('normalizer', Normalizer())]
#         params['observations_preprocessing_steps'] = [('normalizer', Normalizer())]

#     elif params['version'] == 3:

#         params['forecast_preprocessing_steps'] = [
#             ('anomalies', AnomaliesScaler_v1_seasonal())]
#         params['observations_preprocessing_steps'] = [
#             ('anomalies', AnomaliesScaler_v2_seasonal()) ]
        
#     else:

#         params['forecast_preprocessing_steps'] = []
#         params['observations_preprocessing_steps'] = []

#     for key, value in hyperparamater_grid.items():
#             params[key] = value 

#     if params['lr_scheduler']:
#         start_factor = params['start_factor']
#         end_factor = params['end_factor']
#         total_iters = params['total_iters']
#     else:
#         start_factor = end_factor = total_iters  =params['start_factor'] = params['end_factor'] =  params['total_iters'] = None
        
#     ##### PG: Ensemble members to load 
#     ensemble_list = params['ensemble_list']
#     ###### PG: Add ensemble features to training features
#     ensemble_mode = params['ensemble_mode'] ##
#     ensemble_features = params['ensemble_features']

#     reg_scale = params["reg_scale"]
#     model = params["model"]
#     hidden_dims = params["hidden_dims"]
#     time_features = params["time_features"].copy()
#     epochs = params["epochs"]
#     batch_size = params["batch_size"]
#     batch_normalization = params["batch_normalization"]
#     dropout_rate = params["dropout_rate"]
#     batch_shuffle = params["batch_shuffle"]
#     optimizer = params["optimizer"]
#     lr = params["lr"]
#     sigmoid_activation = params['sigmoid_activation']
#     l2_reg = params['L2_reg']

#     forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
#     observations_preprocessing_steps = params["observations_preprocessing_steps"]

#     loss_region = params["loss_region"]
#     subset_dimensions = params["subset_dimensions"]
#     num_val_years = params['num_val_years']

#     if params["arch"] == 2:
#         hidden_dims = [[360, 180, 90], [180, 360]]

#     test_years = test_year

#     if n_runs > 1:
#         numpy_seed = None
#         torch_seed = None



#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     print(f"Start run for test year {test_year}...")

#     train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time < (test_year-num_val_years) * 100].to_numpy()
#     # validation_years = ds_raw_ensemble_mean.year[(ds_raw_ensemble_mean.year >= test_year-3)&(ds_raw_ensemble_mean.year < test_year)].to_numpy()
    
#     n_train = len(train_years)
#     train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...])

#     ds_baseline = ds_raw_ensemble_mean[:n_train,...]
#     obs_baseline = obs_raw[:n_train,...]

#     if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
#         preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
#     else:
#         preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)
#     preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)


#     if numpy_seed is not None:
#         np.random.seed(numpy_seed)
#     if torch_seed is not None:
#         torch.manual_seed(torch_seed)

#     # Data preprocessing
    
#     ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
#     ds = ds_pipeline.transform(ds_raw_ensemble_mean)
    

#     obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
#     # if 'standardize' in ds_pipeline.steps:
#     #     obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
#     obs = obs_pipeline.transform(obs_raw) 

    # y0 = np.floor(ds[:n_train].time[0].values/100 )
    # yr, mn = np.divmod(int(ds[:n_train+12].time[-1].values - y0*100),100)
    # month_min_max = [y0, yr * 12 + mn]
#     # TRAIN MODEL

#     lead_time = None
#     ds_train = ds[:n_train,...]
#     obs_train = obs[:n_train,...]
#     ds_validation = ds[n_train:n_train + num_val_years*12,...]
#     obs_validation = obs[n_train:n_train + num_val_years*12,...]

        
#     weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
#     weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove() 
#     weights = weights * land_mask
#     ####################################################################
#     weights_val = weights.copy()
#     if params['equal_weights']:
#           weights = xr.ones_like(weights) * land_mask

#     if model in [UNet] : ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

#         if loss_region is not None:
#             loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region)
        
#         else:
#             loss_region_indices = None
    
#     else: ## PG: If you have a dense first layer keep the data flattened.

#         ds_train = ds_train.stack(ref = ['lat','lon']) # PG: flatten and sample training data at those locations
#         obs_train = obs_train.stack(ref = ['lat','lon']) ## PG: flatten and sample obs data at those locations
#         weights = weights.stack(ref = ['lat','lon']) ## PG: flatten and sample weighs at those locations
#         weights_val = weights_val.stack(ref = ['lat','lon'])
#         img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.
#         if loss_region is not None:
    
#                     loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region, flat = True) ### the function has to be editted for flat opeion!!!!!

#         else:
#             loss_region_indices = None


#     weights = weights.values
#     weights_val = weights_val.values

#     if time_features is None:
#         if ensemble_features: ## PG: We can choose to add an ensemble feature.
#             add_feature_dim = 1
#         else:
#             add_feature_dim = 0
#     else:
#         if ensemble_features:
#             add_feature_dim = len(time_features) + 1
#         else:
#             add_feature_dim = len(time_features)




#     if model == Autoencoder:
#         net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, sigmoid = sigmoid_activation)
#     elif model == UNet:
#         net = model()


#     net.to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
#     if params['lr_scheduler']:
#         scheduler = lr_scheduler.LinearLR(optimizer, start_factor=params['start_factor'], end_factor=params['end_factor'], total_iters=params['total_iters'])

#     ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
#     train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, in_memory=True, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features, aligned = True, month_min_max = month_min_max) 
#     dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=batch_shuffle)

#     if reg_scale is None: ## PG: if no penalizing for negative anomalies
#         criterion = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)

#     else:
#         criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = False)

#     # EVALUATE MODEL
#     ##################################################################################################################################

#     if model == UNet:
#             pass
#     else:
#         ds_validation = ds_validation.stack(ref = ['lat','lon'])
#         obs_validation = obs_validation.stack(ref = ['lat','lon'])

#     val_mask = create_mask(ds_validation)
#     validation_set = XArrayDataset(ds_validation, obs_validation, mask=val_mask, lead_time=None, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max) 
#     dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=True)   

#     if reg_scale is None:
#         criterion_eval =  WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
#     else:
#         criterion_eval = WeightedMSEGlobalLoss(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale,map  =False)
#     criterion_eval_mean =   WeightedMSE(weights=weights_val, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
#     criterion_eval_extent = GlobalLoss( device=device, scale=1, weights=weights_val, loss_area=None, map = False)
#     ##################################################################################################################################
    
#     epoch_loss_train = []
#     epoch_loss_val = []
#     epoch_loss_val_extent = []
#     epoch_loss_val_mean = []

#     for epoch in tqdm.tqdm(range(epochs)):
#         net.train()
#         batch_loss = 0
        
#         for batch, (x, y) in enumerate(dataloader):
            
#             if (type(x) == list) or (type(x) == tuple):
#                 x = (x[0].to(device), x[1].to(device))
#             else:
#                 x = x.to(device)
#             y = y.to(device)
#             optimizer.zero_grad()
#             adjusted_forecast = net(x)
#             loss = criterion(adjusted_forecast, y)
#             if params['loss_function'] == 'RMSE': 
#                         loss = torch.sqrt(loss)
#             batch_loss += loss.item()
#             loss.backward()
#             optimizer.step()  
#         epoch_loss_train.append(batch_loss / len(dataloader))

#         if params['lr_scheduler']:
#             scheduler.step()

#         net.eval()
#         val_loss = 0
#         val_loss_extent = 0
#         val_loss_mean = 0
        
#         for batch, (x, target) in enumerate(dataloader_val):         
#             with torch.no_grad():            
#                 if (type(x) == list) or (type(x) == tuple):
#                     # test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
#                     test_raw = (x[0].to(device), x[1].to(device))
#                 else:
#                     # test_raw = x.unsqueeze(0).to(device)
#                     test_raw = x.to(device)
#                 # test_obs = target.unsqueeze(0).to(device)
#                 test_obs = target.to(device)
#                 test_adjusted = net(test_raw)
#                 loss_ = criterion_eval(test_adjusted, test_obs)
#                 if params['loss_function'] == 'RMSE': 
#                         loss_ = torch.sqrt(loss_)
#                 val_loss += loss_.item()
#                 loss_extent = criterion_eval_extent(test_adjusted, test_obs)
#                 val_loss_extent += loss_extent.item()
#                 loss_mean = criterion_eval_mean(test_adjusted, test_obs)
#                 if params['loss_function'] == 'RMSE': 
#                     loss_mean = torch.sqrt(loss_mean)
#                 val_loss_mean += loss_mean.item()


#         epoch_loss_val.append(val_loss / len(dataloader_val))
#         epoch_loss_val_extent.append(val_loss_extent / len(dataloader_val))
#         epoch_loss_val_mean.append(val_loss_mean / len(dataloader_val))
#         # Store results as NetCDF            

#     epoch_loss_val = smooth_curve(epoch_loss_val)
#     epoch_loss_val_extent = smooth_curve(epoch_loss_val_extent)
#     epoch_loss_val_mean = smooth_curve(epoch_loss_val_mean)
#     # if params["version"] == 'PatternsOnly':
#     #     epoch_loss_val_extent = np.zeros_like(epoch_loss_val_extent)

#     plt.figure(figsize = (8,5))
#     plt.plot(np.arange(2,epochs+1), epoch_loss_train[1:], label = 'Train')
#     plt.plot(np.arange(2,epochs+1), epoch_loss_val[1:], label = 'Validation')
#     plt.plot(np.arange(2,epochs+1), epoch_loss_val_mean[1:], label = 'MSE Loss', linestyle = 'dashed', alpha = 0.5)
#     plt.title(f'{hyperparamater_grid}')
#     plt.legend()
#     plt.ylabel(params['loss_function'])
#     plt.twinx()
#     plt.plot(np.arange(2,epochs+1), epoch_loss_val_extent[1:], label = 'Validation Area Mean RMSE',  color = 'k', alpha = 0.5)
#     plt.ylabel('Km^2/R')
#     plt.legend()
#     plt.xlabel('Epoch')
    
#     plt.grid()
#     plt.show()
#     plt.savefig(results_dir+f'/val-train_loss_1982-{test_year}-{hyperparamater_grid}.png')
#     plt.close()


#     with open(Path(results_dir, "Hyperparameter_training.txt"), 'a') as f:
#         f.write(
  
#             f"{hyperparamater_grid} ---> val_loss at best epoch: {min(epoch_loss_val)} at {np.argmin(epoch_loss_val)+1}  (MSE : {epoch_loss_val_mean[np.argmin(epoch_loss_val)]})\n" +  ## PG: The scale to be passed to Signloss regularization
#             f"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" 
#         )
#     return epoch_loss_val_mean[np.argmin(epoch_loss_val)], epoch_loss_val, epoch_loss_train ,epoch_loss_val_extent, epoch_loss_val_mean#, epoch_loss_train

#                                  #########         ##########

# def run_hp_tunning( ds_raw_ensemble_mean: XArrayDataset ,obs_raw: XArrayDataset, land_mask :XArrayDataset,   hyperparameterspace: list, params:dict, y_start: int, out_dir_x , n_runs=1, numpy_seed=None, torch_seed=None ):

#     y_end = int(ds_raw_ensemble_mean.time[-1]/100) +1 


#     val_losses = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
#     val_losses_extent = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
#     val_losses_mean = np.zeros([y_end - y_start + 1 , len(hyperparameterspace), params['epochs']]) ####
#     train_losses = np.zeros([y_end - y_start + 1, len(hyperparameterspace), params['epochs']]) ####

#     for ind_, test_year in enumerate(range(y_start,y_end+1)):
    
#         # out_dir_xx = f'{out_dir_x}/git_data_20230426'
#         # out_dir    = f'{out_dir_xx}/SPNA' 
#         out_dir    = f'{out_dir_x}/_{test_year}' 
        
        
#         Path(out_dir).mkdir(parents=True, exist_ok=True)

#         with open(Path(out_dir, "Hyperparameter_training.txt"), 'w') as f:
#             f.write(
#                 f"model\t{params['model']}\n" +
#                 "default set-up:\n" + 
#                 f"hidden_dims\t{params['hidden_dims']}\n" +
#                 f"loss_function\t{params['loss_function']}\n" + 
#                 f"time_features\t{params['time_features']}\n" +
#                 f"ensemble_list\t{params['ensemble_list']}\n" + ## PG: Ensemble list
#                 f"ensemble_features\t{params['ensemble_features']}\n" + ## PG: Ensemble features
#                 f"batch_normalization\t{params['batch_normalization']}\n" +
#                 f"dropout_rate\t{params['dropout_rate']}\n" +
#                 f"sigmoid_activation\t{params['sigmoid_activation']}\n" +
#                 f"lr\t{params['lr']}\n" +
#                 f"lr_scheduler\t{params['lr_scheduler']}: {params['start_factor']} --> {params['end_factor']} in {params['total_iters']} epochs\n"

#                 f"L2_reg\t{params['L2_reg']}\n\n\n" +
#                 ' ----------------------------------------------------------------------------------\n'
#             )
        

        
#         losses = np.zeros(len(hyperparameterspace))
 

        
#         for ind, dic in enumerate(hyperparameterspace):
#             print(f'Training for {dic}')
#             # losses[ind], val_losses[ind_, ind, :], val_losses_global[ind_, ind, :], val_losses_corr[ind_, ind, :],  train_losses[ind_, ind, :] = run_training_hp(dic, params, test_year=test_year, lead_years=lead_years, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)
#             losses[ind], val_losses[ind_, ind, :],  train_losses[ind_, ind, :] , val_losses_extent[ind_, ind, :], val_losses_mean[ind_, ind, :] = training_hp(ds_raw_ensemble_mean =  ds_raw_ensemble_mean,obs_raw = obs_raw , hyperparamater_grid= dic,land_mask=land_mask, params = params , test_year=test_year, n_runs=n_runs, results_dir=out_dir, numpy_seed=numpy_seed, torch_seed=torch_seed)

        
#         with open(Path(out_dir, "Hyperparameter_training.txt"), 'a') as f:
#             f.write(
    
#                 f"Best MSE: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]} \n" +  ## PG: The scale to be passed to Signloss regularization
#                 f"--------------------------------------------------------------------------------------------------------\n" 
#             )

#         print(f"Best loss: {min(losses)} --> {hyperparameterspace[np.argmin(losses)]}")
#         print(f'Output dir: {out_dir}')
#         print('Training done.')

#     coords = []
#     for item in hyperparameterspace:
#         coords.append(str(tuple(item.values())))

#     ds_val = xr.DataArray(val_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
#     ds_val.attrs['hyperparameters'] = list(config_dict.keys())
#     ds_val.to_netcdf(out_dir_x + '/validation_losses.nc')

#     if params["version"] != 'PatternsOnly':

#         ds_val_extent = xr.DataArray(val_losses_extent, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
#         ds_val_extent.attrs['hyperparameters'] = list(config_dict.keys())
#         ds_val_extent.to_netcdf(out_dir_x + '/validation_losses_area_mean.nc')

#     ds_val_mean = xr.DataArray(val_losses_mean, dims = ['test_years', 'hyperparameters','epochs'], name = 'Validation_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1 ), 'hyperparameters': coords})
#     ds_val_mean.attrs['hyperparameters'] = list(config_dict.keys())
#     ds_val_mean.to_netcdf(out_dir_x + '/validation_losses_MSE.nc')

#     ds_train = xr.DataArray(train_losses, dims = ['test_years', 'hyperparameters','epochs'], name = 'Train_loss').assign_coords({'test_years' : np.arange(y_start,y_end +1), 'hyperparameters': coords})
#     ds_train.attrs['hyperparameters'] = list(config_dict.keys())
#     ds_train.to_netcdf(out_dir_x + '/train_losses.nc')

            

# if __name__ == "__main__":

#     dir_name = 'results'
#     run_name = f'training_Autoencoder'
#     #out_dir = Path(dir_name, run_name)
#     #out_dir.mkdir(parents=True, exist_ok=True)
    
#     lead_months = 12
#     n_runs = 1  # number of training runs

#     params = {
#         "model": Autoencoder,
#         "hidden_dims":  [[3600, 1800, 900], [1800, 3600]],# [3600, 1800, 900, 1800, 3600],
#         "time_features":  ['month_sin','month_cos', 'imonth_sin', 'imonth_cos'],
#         "ensemble_features": False, ## PG
#         'ensemble_list' : None, ## PG
#         'ensemble_mode' : 'Mean',
#         "epochs": 100,
#         "batch_size": 8,
#         "reg_scale" : None,
#         "batch_normalization": False,
#         "dropout_rate": 0,
#         "append_mode": 1,
#         "sigmoid_activation" : False,
#         "optimizer": torch.optim.Adam,
#         "lr": 0.001,
#         "batch_shuffle" : True,
#         "loss_region": None,
#         "subset_dimensions": 'North' , ##  North or South or Global
#         "loss_function" : 'MSE',
#         "equal_weights" : False,
#         "L2_reg": 0,
#         'lr_scheduler' : False
#     }


#     ################################################################# Set basic config ###########################################################################
#     lead_months = 12
#     n_runs = 1  # number of training runs
#     params['version'] = 'PatternsOnly'   ### 1 , 2 ,3, 'PatternsOnly' 
#     params["loss_function"] = 'MSE'

#     ### load data

#     obs_ref = 'NASA'

#     ##############################################################  Don't touch the following ######################################################################

#     ds_raw_ensemble_mean, obs_raw, params, land_mask = HP_congif(params, obs_ref, lead_months)

#     ########################################################### Set HP space specifics #########################################################################

#     y_start = 2011
    
#     params['num_val_years'] = 5

#     config_dict = { 'batch_size' : [100], 'reg_scale' : [None,1,3] }
#     hyperparameterspace = config_grid(config_dict).full_grid()


#     params['arch'] = 2


#     ##################################################################  Adjust the path if necessary #############3##############################################

#     out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/{obs_ref}/{params["model"].__name__}/run_set_1/Model_tunning/arch{params["arch"]}/batch_reg_tunning_adj_lr_{params["subset_dimensions"]}_v{params["version"]}_2'

#     if params['lr_scheduler']:
#         out_dir_x = out_dir_x + '_lr_scheduler'
#         params['start_factor'] = 1.0
#         params['end_factor'] = 0.5
#         params['total_iters'] = 100
#     run_hp_tunning(ds_raw_ensemble_mean = ds_raw_ensemble_mean ,obs_raw = obs_raw,land_mask=land_mask,  hyperparameterspace = hyperparameterspace, params = params, y_start = y_start , out_dir_x = out_dir_x, n_runs=1, numpy_seed=1, torch_seed=1 )

