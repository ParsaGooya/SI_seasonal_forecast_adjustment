# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tqdm

# import dask
# import xarray as xr
# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader
# from torch.optim import lr_scheduler
# from models.autoencoder import Autoencoder
# from models.unet import UNet
# from models.cnn import SCNN
# from losses import WeightedMSE, WeightedMSEOutlierLoss, WeightedMSEGlobalLoss
# from preprocessing import align_data_and_targets, create_mask
# from preprocessing import AnomaliesScaler_v1_seasonal,AnomaliesScaler_v2_seasonal, Standardizer, PreprocessingPipeline, calculate_climatology, bias_adj
# from torch_datasets import XArrayDataset
# # from subregions import subregions
# from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
# import glob
# # specify data directories
# data_dir_forecast = LOC_FORECASTS_SI




# def run_training(params, n_years, lead_months, n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, save = False):
    
    
#     if params["model"] != Autoencoder:
#         params["append_mode"] = None

#     if obs_ref == 'NASA':
#         data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/NASA*.nc')[0]
#     else:
#         data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

#     assert params['version'] in [1,2,3, 'PatternsOnly']

#     if params['version'] == 'PatternsOnly':
#         params['forecast_preprocessing_steps'] = [
#         ('standardize', Standardizer())]

#         params['observations_preprocessing_steps'] = []
#     elif params['version'] == 3:

#         params['forecast_preprocessing_steps'] = [
#             ('anomalies', AnomaliesScaler_v1_seasonal())]
#         params['observations_preprocessing_steps'] = [
#             ('anomalies', AnomaliesScaler_v2_seasonal()) ]
         
#     else:
#         params['forecast_preprocessing_steps'] = []
#         params['observations_preprocessing_steps'] = []
    
#     if params['lr_scheduler']:
#         start_factor = params['start_factor']
#         end_factor = params['end_factor']
#         total_iters = params['total_iters']
#     else:
#         start_factor = end_factor = total_iters = None

#     print("Start training")
#     print("Load observations")

#     obs_in = xr.open_dataset(data_dir_obs)['SICN']
    
#     ##### PG: Ensemble members to load 
#     ensemble_list = params['ensemble_list']
#     ###### PG: Add ensemble features to training features
#     ensemble_mode = params['ensemble_mode'] ##
#     ensemble_features = params['ensemble_features']

#     if params['version'] == 2:

#         params['forecast_preprocessing_steps'] = []
#         params['observations_preprocessing_steps'] = []
#         ds_in = xr.open_dataset('/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/Bias_Adjusted/global_mean_bias_adjusted_1983-2020.nc')['SICN']
#         if ensemble_list is not None:
#             raise RuntimeError('With version 2 you are reading the bias adjusted ensemble mean as input. Set ensemble_list to None to proceed.')

#     else:

#         if ensemble_list is not None: ## PG: calculate the mean if ensemble mean is none
#             print("Load forecasts")
#             ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
#             ds_in = xr.concat(ls, dim = 'time').sortby('time').sel(ensembles = ensemble_list)
#             if ensemble_mode == 'Mean': 
#                 ds_in = ds_in.mean('ensembles') 
#             else:
#                 ds_in = ds_in.transpose('time','lead_time','ensembles',...)
#                 print(f'Warning: ensemble_mode is {ensemble_mode}. Training for large ensemble ...')

#         else:    ## Load specified members
#             print("Load forecasts") 
#             ls = [xr.open_dataset(glob.glob(LOC_FORECASTS_SI + f'/*_initial_month_{intial_month}_*.nc')[0])['SICN'] for intial_month in range(1,13) ]
#             ds_in = xr.concat(ls, dim = 'time').mean('ensembles').sortby('time')
    
    # ###### handle nan and inf over land ############
    #  ### land is masked in model data with a large number
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
    
#     subset_dimensions = params["subset_dimensions"]

#     if subset_dimensions is not None:
#         if subset_dimensions == 'North':
#             ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat > 40, drop = True)
#             obs_raw = obs_raw.where(obs_raw.lat > 40, drop = True)
#             land_mask = land_mask.where(land_mask.lat > 40, drop = True)
            # model_mask = model_mask.where(model_mask.lat > 40, drop = True)

#         else:
#             ds_raw_ensemble_mean = ds_raw_ensemble_mean.where(ds_raw_ensemble_mean.lat < -40, drop = True)
#             obs_raw = obs_raw.where(obs_raw.lat < -40, drop = True)
#             land_mask = land_mask.where(land_mask.lat < -40, drop = True)
            # model_mask = model_mask.where(model_mask.lat < -40, drop = True)

    # ################################### apply the mask #######################
    ## land_mask = land_mask.where(model_mask == 1, 0)
    # obs_raw = obs_raw * land_mask
    # ds_raw_ensemble_mean = ds_raw_ensemble_mean * land_mask
    # ################################### apply the mask #######################

#     if params['version'] == 'PatternsOnly':

#             weights = np.cos(np.ones_like(ds_raw_ensemble_mean.lon) * (np.deg2rad(ds_raw_ensemble_mean.lat.to_numpy()))[..., None])  # Moved this up
#             weights = xr.DataArray(weights, dims = ds_raw_ensemble_mean.dims[-2:], name = 'weights').assign_coords({'lat': ds_raw_ensemble_mean.lat, 'lon' : ds_raw_ensemble_mean.lon}) 
#             ds_raw_mean = ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
#             obs_raw_mean = ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
#             ds_raw_ensemble_mean = ds_raw_ensemble_mean - ((ds_raw_ensemble_mean * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
#             obs_raw = obs_raw - ((obs_raw * weights).sum(['lat','lon'])/weights.sum(['lat','lon']))
    
#     if params['version']  in ['PatternsOnly',3]:
          
#             params['sigmoid_activation'] = False
         
#     reg_scale = params["reg_scale"]
#     model = params["model"]
#     hidden_dims = params["hidden_dims"]
#     time_features = params["time_features"]
#     epochs = params["epochs"]
#     batch_size = params["batch_size"]
#     batch_normalization = params["batch_normalization"]
#     dropout_rate = params["dropout_rate"]

#     optimizer = params["optimizer"]
#     lr = params["lr"]
#     sigmoid_activation = params['sigmoid_activation']
#     l2_reg = params["L2_reg"]

#     forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
#     observations_preprocessing_steps = params["observations_preprocessing_steps"]

#     loss_region = params["loss_region"]
    

#     test_years = np.arange( int(ds_raw_ensemble_mean.time[-1]/100 - n_years + 1), int(ds_raw_ensemble_mean.time[-1]/100) + 2)

#     if n_runs > 1:
#         numpy_seed = None
#         torch_seed = None

#     with open(Path(results_dir, "training_parameters.txt"), 'w') as f:
#         f.write(
#             f"model\t{model.__name__}\n" +
#             f"reg_scale\t{reg_scale}\n" +  ## PG: The scale to be passed to Signloss regularization
#             f"hidden_dims\t{hidden_dims}\n" +
#             f"loss_function\t{params['loss_function']}\n" + 
#             f"time_features\t{time_features}\n" +
#             f"extra_predictors\t{params['extra_predictors']}\n" +
#             f"append_mode\t{params['append_mode']}\n" +
#             f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
#             f"ensemble_features\t{ensemble_features}\n" + ## PG: Ensemble features
#             f"epochs\t{epochs}\n" +
#             f"batch_size\t{batch_size}\n" +
#             f"batch_normalization\t{batch_normalization}\n" +
#             f"dropout_rate\t{dropout_rate}\n" +
#             f"optimizer\t{optimizer.__name__}\n" +
#             f"lr\t{0.001}\n" +
#             f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n"
#             f"sigmoid_activation\t{sigmoid_activation}\n" +
#             f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
#             f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            # f"equal_weights\t{params['equal_weights']}\n"  +            
            # f"loss_region\t{loss_region}\n" +
#             f"subset_dimensions\t{subset_dimensions}\n" + 
#             f"L2_reg\t{l2_reg}"
#         )

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     for run in range(n_runs):
#         print(f"Start run {run + 1} of {n_runs}...")
#         for y_idx, test_year in enumerate(test_years):
#             print(f"Start run for test year {test_year}...")


#             train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time < test_year * 100].to_numpy()
#             n_train = len(train_years)
#             train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...])

#             ds_baseline = ds_raw_ensemble_mean[:n_train,...]
#             obs_baseline = obs_raw[:n_train,...]
#             if params['version'] == 'PatternsOnly':
#                     ds_baseline_mean = ds_raw_mean[:n_train,...]
#                     obs_baseline_mean = obs_raw_mean[:n_train,...]


#             if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
#                 preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)
#             else:
#                 preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)
#             preprocessing_mask_obs = np.broadcast_to(train_mask[...,None,None,None], obs_baseline.shape)


#             if numpy_seed is not None:
#                 np.random.seed(numpy_seed)
#             if torch_seed is not None:
#                 torch.manual_seed(torch_seed)

#             # Data preprocessing
            
#             ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
#             ds = ds_pipeline.transform(ds_raw_ensemble_mean)

#             obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
#             # if 'standardize' in ds_pipeline.steps:
#             #     obs_pipeline.add_fitted_preprocessor(ds_pipeline.get_preprocessors('standardize'), 'standardize')
#             obs = obs_pipeline.transform(obs_raw)

#             y0 = np.floor(ds[:n_train].time[0].values/100 )
            # yr, mn = np.divmod(int(ds[:n_train+12].time[-1].values - y0*100),100)
            # month_min_max = [y0, yr * 12 + mn]
#             # TRAIN MODEL

#             lead_time = None
#             ds_train = ds[:n_train,...]
#             obs_train = obs[:n_train,...]
#             if test_year < int(ds_raw_ensemble_mean.time[-1]/100) + 1:
#                     ds_test = ds[n_train:n_train + 12,...]
#                     obs_test = obs[n_train:n_train + 12,...]

#             if params['version'] == 'PatternsOnly':
#                     if 'ensembles' in ds_baseline_mean.dims:  
#                         mask_bias_adj = train_mask[...,None, None]
#                     else:
#                         mask_bias_adj = train_mask[...,None]

#                     fct_climatology = calculate_climatology(ds_baseline_mean.where(~np.broadcast_to(mask_bias_adj, ds_baseline_mean.shape)))
#                     obs_climatology = calculate_climatology(obs_baseline_mean.where(~np.broadcast_to(train_mask[...,None], obs_baseline_mean.shape)))
#                     fct_bias_adjusted = bias_adj(ds_raw_mean[n_train:n_train + 12,...] , fct_climatology , obs_climatology)


#             weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
#             weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove()  
#             ####################################################################
#             weights_ = weights.copy()
            # if params['equal_weights']:
            #     weights = xr.ones_like(weights) * land_mask


#             if model in [UNet,SCNN] : ## PG: If the model starts with a nn.Conv2d write back the flattened data to maps.

#                 img_dim = ds_train.shape[-2] * ds_train.shape[-1] 
#                 if loss_region is not None:
#                     loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region, flat = False)  ### the function has to be editted for flat opeion!!!!! 
                
#                 else:
#                     loss_region_indices = None

            
#             else: ## PG: If you have a dense first layer keep the data flattened.
                
#                 ds_train = ds_train.stack(ref = ['lat','lon']) # PG: flatten and sample training data at those locations
#                 obs_train = obs_train.stack(ref = ['lat','lon']) ## PG: flatten and sample obs data at those locations
#                 weights = weights.stack(ref = ['lat','lon']) ## PG: flatten and sample weighs at those locations
#                 weights_ = weights_.stack(ref = ['lat','lon'])

#                 img_dim = ds_train.shape[-1] ## PG: The input dim is now the length of the flattened dimention.

#                 if loss_region is not None:
#                     loss_region_indices, loss_area = get_coordinate_indices(ds_raw_ensemble_mean, loss_region, flat = True) ### the function has to be editted for flat opeion!!!!!
                
#                 else:
#                     loss_region_indices = None

#             weights = weights.values
#             weights_ = weights_.values

#             if time_features is None:
#                 if ensemble_features: ## PG: We can choose to add an ensemble feature.
#                     add_feature_dim = 1
#                 else:
#                     add_feature_dim = 0
#             else:
#                 if ensemble_features:
#                     add_feature_dim = len(time_features) + 1
#                 else:
#                     add_feature_dim = len(time_features)




#             if model == Autoencoder:
#                 net = model(img_dim, hidden_dims[0], hidden_dims[1], added_features_dim=add_feature_dim, append_mode=params['append_mode'], batch_normalization=batch_normalization, dropout_rate=dropout_rate, sigmoid = sigmoid_activation)
#             elif model == UNet:
#                 net = model()
#             elif model == SCNN: ## PG: Combining CNN encoder with dense decoder
#                 net = model(img_dim,hidden_dims, batch_normalization=batch_normalization, dropout_rate=dropout_rate )

#             net.to(device)
#             optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
#             if params['lr_scheduler']:
#                 scheduler = lr_scheduler.LinearLR(optimizer, start_factor=params['start_factor'], end_factor=params['end_factor'], total_iters=params['total_iters'])

#             ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.
#             train_set = XArrayDataset(ds_train, obs_train, mask=train_mask, in_memory=True, lead_time=lead_time, time_features=time_features,ensemble_features =ensemble_features, aligned = True, month_min_max = month_min_max) 
#             dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#             if reg_scale is None: ## PG: if no penalizing for negative anomalies

#                     criterion = WeightedMSE(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)
#             else:
#                 if params['version'] != 'PatternsOnly':

#                         if type(reg_scale) == dict:
#                             criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale[test_year], map = False)
#                         else:
#                             criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = False)
#                 else:
#                         if type(reg_scale) == dict:
#                             criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale[test_year], map = False)
#                         else:
#                             criterion = WeightedMSEGlobalLoss(weights=weights, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices, scale=reg_scale, map = False)
               

#             epoch_loss = []
#             net.train()
#             num_batches = len(dataloader)
#             for epoch in tqdm.tqdm(range(epochs)):
#                 batch_loss = 0
#                 for batch, (x, y) in enumerate(dataloader):
#                     if (type(x) == list) or (type(x) == tuple):
#                         x = (x[0].to(device), x[1].to(device))
#                     else:
#                         x = x.to(device)
#                     y = y.to(device)
#                     optimizer.zero_grad()
#                     adjusted_forecast = net(x)
#                     loss = criterion(adjusted_forecast, y)
#                     if params['loss_function'] == 'RMSE': 
#                         loss = torch.sqrt(loss)
#                     batch_loss += loss.item()
#                     loss.backward()
#                     optimizer.step()
#                 epoch_loss.append(batch_loss / num_batches)

#                 if params['lr_scheduler']:
#                     scheduler.step()

#             # EVALUATE MODEL
#             ##################################################################################################################################
#             if test_year < int(ds_raw_ensemble_mean.time[-1]/100) + 1:
#                 if model == Autoencoder:
#                     ds_test = ds_test.stack(ref = ['lat','lon'])  ## PG: Sample the test data at the common locations
#                     obs_test = obs_test.stack(ref = ['lat','lon'])
#                 ##################################################################################################################################
#                 test_years_list = np.arange(1, ds_test.shape[0] + 1)
#                 test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)

    
#                 ## PG: Extract the number of years as well 
#                 test_set = XArrayDataset(ds_test, obs_test, lead_time=None,mask = None, time_features=time_features,ensemble_features =ensemble_features,  in_memory=False, aligned = True, month_min_max = month_min_max)
#                 criterion_test =  WeightedMSE(weights=weights_, device=device, hyperparam=1, reduction='mean', loss_area=loss_region_indices)

#                 if 'ensembles' in ds_test.dims:
#                     test_loss = np.zeros(shape=(ds_test.stack(flattened=('time','lead_time')).transpose('flattened',...).shape[:2]))
#                     test_results = np.zeros_like(ds_test.stack(flattened=('time','lead_time')).transpose('flattened',...).data)
#                     results_shape = xr.full_like(ds_test.stack(flattened=('time','lead_time')).transpose('flattened',...), fill_value = np.nan)
#                     test_time_list =  np.arange(1, results_shape.shape[0] + 1)
#                 else:
#                     test_loss = np.zeros(shape=(test_set.target.shape[0]))
#                     test_results = np.zeros_like(test_set.target)
#                     results_shape = xr.full_like(test_set.target, fill_value = np.nan)

                 

#                 for i, (x, target) in enumerate(test_set): 
#                     if 'ensembles' in ds_test.dims:  ## PG: If we have large ensembles:
    
#                         ensemble_idx, j = np.divmod(i, len(test_time_list))  ## PG: find out ensemble index
#                         net.eval()
#                         with torch.no_grad():
#                             if (type(x) == list) or (type(x) == tuple):
#                                 test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
#                             else:
#                                 test_raw = x.unsqueeze(0).to(device)
#                             test_obs = target.unsqueeze(0).to(device)
#                             test_adjusted = net(test_raw)
#                             loss = criterion_test(test_adjusted, test_obs)
#                             test_results[j,ensemble_idx,] = test_adjusted.to(torch.device('cpu')).numpy()  ## PG: write back to test_results
#                             test_loss[j,ensemble_idx] = loss.item()
#                     else:

#                         net.eval()
#                         with torch.no_grad():
#                             if (type(x) == list) or (type(x) == tuple):
#                                 test_raw = (x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device))
#                             else:
#                                 test_raw = x.unsqueeze(0).to(device)
#                             test_obs = target.unsqueeze(0).to(device)
#                             test_adjusted = net(test_raw)
#                             loss = criterion_test(test_adjusted, test_obs)
#                             if params['loss_function'] == 'RMSE': 
#                                 loss = torch.sqrt(loss)
#                             test_results[i,] = test_adjusted.to(torch.device('cpu')).numpy()
#                             test_loss[i] = loss.item()

#                 ###################################################### has to be eddited for large ensembles!! #####################################################################
#                 results_shape[:] = test_results[:]
                
#                 if model in [UNet,SCNN]:  ## PG: if the output is already a map

#                     test_results = results_shape.unstack('flattened').transpose('time','lead_time',...)
#                     if test_results.shape[0] < 12:
#                         test_results = test_results.combine_first(xr.full_like(ds_test, np.nan))

#                     test_results_untransformed = obs_pipeline.inverse_transform(test_results.values)
#                     result = xr.DataArray(test_results_untransformed, ds_test.coords, ds_test.dims, name='nn_adjusted')
#                 else:  
#                     test_results = results_shape.unstack('flattened').transpose('time','lead_time',...,'ref').unstack('ref')
#                     if test_results.shape[0] < 12:
#                         test_results = test_results.combine_first(xr.full_like(ds_test.unstack('ref'), np.nan))

#                     test_results_untransformed = obs_pipeline.inverse_transform(test_results.values) ## PG: Check preprocessing.AnomaliesScaler for changes
#                     result = xr.DataArray(test_results_untransformed, ds_test.unstack('ref').coords, ds_test.unstack('ref').dims, name='nn_adjusted')

#                 # print(test_results)
#                 ##############################################################################################################################################################
#                 # Store results as NetCDF           
#                 if params['version'] == 'PatternsOnly':
                      
#                     result = result.to_dataset(name = 'nn_adjusted') + fct_bias_adjusted.to_dataset(name = 'nn_adjusted')
                    
#                 result.to_netcdf(path=Path(results_dir, f'nn_adjusted_{test_year}_{run+1}.nc', mode='w'))

#                 fig, ax = plt.subplots(1,1, figsize=(8,5))
#                 ax.plot(np.arange(1,epochs+1), epoch_loss)
#                 ax.set_title(f'Train Loss \n test loss: {np.mean(test_loss)}') ###
                
#                 ax.set_xlabel('Epoch')
#                 ax.set_ylabel('Loss')
#                 plt.show()
#                 plt.savefig(results_dir+f'/Figures/train_loss_1982-{test_year-1}.png')
#                 plt.close()

#                 if save:
#                     nameSave = f"MODEL_V{params['version']}_1982-{test_year-1}.pth"
#                     torch.save( net.state_dict(),results_dir + '/' + nameSave)
#             else:
                
#                 nameSave = f"MODEL_final_V{params['version']}_1980-{int(ds_raw_ensemble_mean.time[-1]/100)}.pth"
#                 # Save locally
#                 torch.save( net.state_dict(),results_dir + '/' + nameSave)

# if __name__ == "__main__":

  
#     n_years =  15 # last n years to test consecutively
#     lead_months = 12
#     n_runs = 1  # number of training runs

#     params = {
#         "model": Autoencoder,
#         "arch" : 1, 
#         "hidden_dims": [[3600, 1800, 900], [1800, 3600]],
#         "time_features": ['month_sin','month_cos', 'imonth_sin', 'imonth_cos'],
#         "ensemble_features": False, ## PG
#         'ensemble_list' : None, ## PG
#         'ensemble_mode' : 'Mean',
#         "epochs": 65,
#         "batch_size": 100,
#         "batch_normalization": False,
#         "dropout_rate": 0,
#         "append_mode": 1,
#         "sigmoid_activation" : False,
#         "reg_scale" : None,
#         "optimizer": torch.optim.Adam,
#         "lr": 0.001 ,
#         "equal_weights" : False,
#         "loss_function" :'RMSE',
#         "loss_region": None,
#         "subset_dims": 'North',   ## North or South or Global
#         "L2_reg": 0,
#         'lr_scheduler' : False
#     }


 
#     params['version'] = 'PatternsOnly'   ### 1 , 2 ,3, 'PatternsOnly' 
#     params["arch"] = 2
#     params['reg_scale'] = 1
#     params["loss_function"] = 'MSE'

#     obs_ref = 'NASA'

    
#     out_dir_x  = f'/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/{obs_ref}/{params["model"].__name__}/run_set_1'
#     if params['lr_scheduler']:
#         params['start_factor'] = 1.0
#         params['end_factor'] = 0.5
#         params['total_iters'] = 100

#     if type(params['subset_dims']) == list:
#         for element in params['subset_dims']:

#             params['subset_dimensions'] = element
#             out_dir    = f'{out_dir_x}/N{n_years}_M{lead_months}_v{params["version"]}_{element}_arch{params["arch"]}_lr{params["lr"]}_batch{params["batch_size"]}_e{params["epochs"]}' 
#             if params['lr_scheduler']:
#                 out_dir = out_dir + '_lr_scheduler'
#             Path(out_dir).mkdir(parents=True, exist_ok=True)
#             Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)

#             run_training(params, n_years=n_years, lead_months=lead_months, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)

#     else:
            
            
#             out_dir    = f'{out_dir_x}/N{n_years}_M{lead_months}_v{params["version"]}_{params["subset_dims"]}_arch{params["arch"]}_lr{params["lr"]}_batch{params["batch_size"]}_e{params["epochs"]}_L{params["reg_scale"]}' 
#             if params['lr_scheduler']:
#                 out_dir = out_dir + '_lr_scheduler'
#             if params['subset_dims'] == 'Global':
#                 params['subset_dimensions'] = None
#             else:
#                 params['subset_dimensions'] = params['subset_dims']

#             Path(out_dir).mkdir(parents=True, exist_ok=True)
#             Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)

#             run_training(params, n_years=n_years, lead_months=lead_months, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1)
#     print(f'Output dir: {out_dir}')
#     print('Training done.')
