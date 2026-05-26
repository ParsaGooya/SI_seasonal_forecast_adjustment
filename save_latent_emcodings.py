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
from torch.distributions.multivariate_normal import MultivariateNormal
from losses import WeightedMSEKLD, WeightedMSE, BCElossKLD
from preprocessing import align_data_and_targets, create_mask, pole_centric, reverse_pole_centric, segment, reverse_segment, pad_xarray, smoother, align_data_and_targets
from preprocessing import AnomaliesScaler_v1_seasonal, AnomaliesScaler_v2_seasonal, Standardizer, Normalizer, PreprocessingPipeline, calculate_climatology, bias_adj, zeros_mask_gen
from torch_datasets import XArrayDataset
import torch.nn as nn
# from subregions import subregions
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
import glob
import gc
import umap
# specify data directories
data_dir_forecast = LOC_FORECASTS_SI


class trained_net():
    def __init__( self, out_dir,   model_year, ds_raw_ensemble_mean, obs_raw, land_mask, model_mask, NPSProj, subset_dimensions, device = 'cpu' ):
        params = self.extract_params(out_dir)
        print(f'loaded configuration: \n')
        for key, values in params.items():
            print(f'{key} : {values} \n')

        version = eval(out_dir.split('/')[-1].split('_')[3][1:]) if 'VAL' not in out_dir else eval(out_dir.split('/')[-1].split('_')[4][1:])
        params["version"] = version
        print( f'Version: {version}')

        
        if 'Linear' in out_dir:
                params['VAE_MLP_encoder'] = True

        try:
            scale_factor_channels = params['scale_factor_channels']
        except:
            scale_factor_channels = None

        params["obs_clim"] = False
        params['forecast_range_months'] = eval(out_dir.split('_F')[1].split('_')[0])
        if 'LT' in out_dir:
            lead_time = eval(out_dir.split('_LT')[1].split('_')[0])
        else:
            lead_time = None
        if 'combined' in out_dir:
            params['combined_prediction'] = True
        else:
            params['combined_prediction'] = False

        params['VAE_latent_size'] = eval(out_dir.split('_LS')[1].split('_')[0])
        if params['version'] == 2:

            params['forecast_preprocessing_steps'] = [
                ('anomalies', AnomaliesScaler_v1_seasonal())]
            params['observations_preprocessing_steps'] = []
            
        else:
            params['forecast_preprocessing_steps'] = []
            params['observations_preprocessing_steps'] = []


        if params['version'] == 'IceExtent':
            params['reg_scale'] = None
            params['combined_prediction'] = False

        if '-DecoderVar' in out_dir:
            params['output_sampling'] = 'learn_decoder_variance'
            learn_decoder_variance =  True
            print('Decoder sampling from learned variance matrix available!')
            

        elif '-DecoderRes' in out_dir:
            params['output_sampling'] = None
            params['learn_decoder_residual'] = True
            print('Decoder residual network available!')
            
        else:
            params['learn_decoder_residual'] = False
            learn_decoder_variance = False

        if 'learn_decoder_sampler' not in params.keys():
            params['learn_decoder_sampler'] = {'status' : False, 'noise_std' : None}
        else:
            params['output_sampling'] = None
        
        if 'noise_injection_level' not in params['learn_decoder_sampler'].keys():
            params['learn_decoder_sampler']['noise_injection_level'] = 'medium'

        try:
            Final_kernel_2 = params['Final_kernel_2']
        except:
            Final_kernel_2 = False     
        ############################################## load data ##################################
        time_features = params["time_features"]

        model = (params['model'])
        forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
        observations_preprocessing_steps = params["observations_preprocessing_steps"]
        if 'skip_VAE_added_dim' not in params.keys():
            params['skip_VAE_added_dim'] = False
        try:
            ensemble_mode = params['ensemble_mode']
        except:
            params['ensemble_mode'] = 'Mean'

        try:
            obs_clim = params["obs_clim"]
        except: 
            pass
        try:
            LocallyConnected = params['LocallyConnected']
        except:
            LocallyConnected = False

        if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
            if 'ensembles'  in obs_raw.dims:
                zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(ensembles = 0).isel(lead_time = 0).drop('lead_time').where(obs_raw.time<model_year *100, drop = True ), 3) ], dim = 'test_year').assign_coords(test_year = model_year)
            else:
                zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(lead_time = 0).drop('lead_time').where(obs_raw.time<model_year *100, drop = True ), 3) ], dim = 'test_year').assign_coords(test_year = model_year)           
                            
            for item in ['active_mask', 'full_ice_mask']:
                zeros_mask_full = zeros_mask_full.drop(item) if item not in params["time_features"] else zeros_mask_full
            zeros_mask_full = zeros_mask_full.drop('active_grid') if not params['active_grid'] else zeros_mask_full

            zeros_mask_full = zeros_mask_full.expand_dims('channels', axis=-3)
            if 'ensembles' in ds_raw_ensemble_mean.dims:
                    zeros_mask_full = zeros_mask_full.expand_dims('ensembles', axis=2)

        if params['version'] == 'IceExtent':
            obs_raw = obs_raw.where(obs_raw>=0.15,0)
            obs_raw = obs_raw.where(obs_raw ==0 , 1)
            learn_decoder_variance = False
            params['loss_function'] = 'BCELoss'
        if params['combined_prediction']:
            obs_raw_ = obs_raw.where(obs_raw>=0.15,0)
            obs_raw_ = obs_raw_.where(obs_raw_ ==0 , 1)
            obs_raw = xr.concat([obs_raw, obs_raw_], dim = 'channels')
            params['loss_function'] = 'combined'
            params['learn_decoder_residual'] = False
            del obs_raw_


        if NPSProj is False:
            
                ds_raw_ensemble_mean = pole_centric(ds_raw_ensemble_mean, subset_dimensions)
                obs_raw =  pole_centric(obs_raw, subset_dimensions)
                land_mask = pole_centric(land_mask, subset_dimensions)
                model_mask = pole_centric(model_mask, subset_dimensions)
                if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                    zeros_mask_full = pole_centric(zeros_mask_full, subset_dimensions)

        self.train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time <= (model_year)*100].to_numpy() 

        if lead_time is not None:
            full_shape = xr.full_like(ds_raw_ensemble_mean, np.nan).isel(lat = slice(1,2), lon = slice(1,2))
            ds_raw_ensemble_mean = ds_raw_ensemble_mean.sel(lead_time = slice(lead_time,lead_time))
            obs_raw = obs_raw.sel(lead_time = slice(lead_time,lead_time))



        if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
            self.zeros_mask = zeros_mask_full.sel(test_year = model_year).drop('test_year')
        else:
            self.zeros_mask = None

        n_train = len(self.train_years)
        train_mask = create_mask(ds_raw_ensemble_mean[:n_train,...]) if lead_time is None else create_mask(full_shape[:n_train,...])[:, lead_time - 1][..., None] ############

        ds_baseline = ds_raw_ensemble_mean[:n_train,...]
        obs_baseline = obs_raw[:n_train,...].isel(channels = slice(0,1))


        if 'ensembles' in ds_raw_ensemble_mean.dims: ## PG: Broadcast the mask to the correct shape if you have an ensembles dim.
            preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None,None], ds_baseline.shape)

        else:
            preprocessing_mask_fct = np.broadcast_to(train_mask[...,None,None,None], ds_baseline.shape)

        if 'ensembles' in obs_raw.dims: 
            preprocessing_mask_obs = np.broadcast_to(train_mask[:n_train ,...,None, None,None,None], obs_baseline.shape)
        else:
            preprocessing_mask_obs = np.broadcast_to(train_mask[:n_train ,..., None,None,None], obs_baseline.shape)

        if params['version']  in [3,1.1]:
            sigmoid_activation = False
        else:
            sigmoid_activation = True

        # Data preprocessing

        ds_pipeline = PreprocessingPipeline(forecast_preprocessing_steps).fit(ds_baseline, mask=preprocessing_mask_fct)
        ds = ds_pipeline.transform(ds_raw_ensemble_mean)

        obs_pipeline = PreprocessingPipeline(observations_preprocessing_steps).fit(obs_baseline, mask=preprocessing_mask_obs)
        obs = obs_pipeline.transform(obs_raw.isel(channels = slice(0,1)))

        if params['combined_prediction']:
                obs = xr.concat([obs, obs_raw.isel(channels = slice(1,2))], dim  = 'channels')  

        del ds_baseline, obs_baseline, preprocessing_mask_obs, preprocessing_mask_fct
        gc.collect()


        if 'land_mask' in time_features:
                ds = xr.concat([ds, land_mask.expand_dims('channels', axis = 0)], dim = 'channels')


        if time_features is None:
            self.add_feature_dim = 0
        else:
            self.add_feature_dim = len(time_features)
        if 'land_mask' in time_features:
            self.add_feature_dim -= 1
        if 'ensemble_error' in time_features:
            add_feature_dim -= 1

        self.n_channels_x = len(ds.channels)

        if 'path_to_deterministic' not in params.keys():
            params['path_to_deterministic'] = None

        if params['path_to_deterministic'] is not None:
            from preprocessing import extract_params
            
            params_saved = extract_params(params['path_to_deterministic'])
            NPSProj_saved = False if '1x1' in params['path_to_deterministic'] else True          
            params_saved['combined_prediction'] = True if 'combined' in params['path_to_deterministic'] else False
            lead_time_saved = eval(params['path_to_deterministic'].split('_LT')[1].split('_')[0]) if 'LT' in params['path_to_deterministic'] else None
            n_val_year_saved = int(params['path_to_deterministic'].split('_VAL')[1][:1]) if '_VAL' in params['path_to_deterministic'] else 0
            try:
                LocallyConnected_saved = params_saved['LocallyConnected']
            except:
                LocallyConnected_saved = False
            assert NPSProj_saved == NPSProj, 'The saved deterministic model cannot have a trained for a different projection than your cVAE!'
            version_saved = eval(params['path_to_deterministic'].split('/')[-1].split('_')[3][1:]) if 'VAL' not in params['path_to_deterministic'] else eval(params['path_to_deterministic'].split('/')[-1].split('_')[4][1:])
            if params['version'] == 'IceExtent':
                assert version_saved in  ['IceExtent', 1]
            else:
                assert version_saved == params['version'], 'The saved deterministic model cannot have a different version than your cVAE!'
            assert lead_time_saved == lead_time, 'The saved deterministic model cannot have a trained for a different lead_time than your cVAE!'
            assert params['time_features'] == params_saved['time_features'], 'The saved deterministic model must have the same input as your cVAE for the model data and added features!'
            assert params_saved['combined_prediction'] == params['combined_prediction'], 'The saved deterministic model must have the same output type as your cVAE'
            assert LocallyConnected == LocallyConnected_saved, 'The saved deterministic model must have the same output layer as your cVAE'
            try:
                Final_kernel_2_saved = params_saved['Final_kernel_2']
            except:
                Final_kernel_2_saved = False  
            import_last_conv = False if Final_kernel_2 != Final_kernel_2_saved else True
            from models.unetconvnext import UNet2,UNet2_NPS
            model = eval(params_saved['model'])
            unet = model(n_channels_x= self.n_channels_x+ self.add_feature_dim , bilinear = params_saved['bilinear'], sigmoid = sigmoid_activation, skip_conv = params_saved['skip_conv'], combined_prediction = params_saved['combined_prediction'], LocallyConnected = LocallyConnected_saved)
            try:
                saved_model_dir = glob.glob(params['path_to_deterministic']+ f'/Checkpoints/*-{model_year}*.pth')
                saved_model_dir.sort()
                unet.load_state_dict(torch.load(saved_model_dir[0], map_location=torch.device('cpu'))) 
            except:
                unet.load_state_dict(torch.load(glob.glob(params['path_to_deterministic']+ f'/Checkpoints/*final*-{model_year - 1}*.pth')[0], map_location=torch.device('cpu'))) 

        else:
            unet = None
     

        if 'DetRess' in out_dir:
            self.net = cVAE(VAE_latent_size = params['VAE_latent_size'], n_channels_x= self.n_channels_x+ self.add_feature_dim , sigmoid = sigmoid_activation, NPS_proj = NPSProj, VAE_MLP_encoder = params['VAE_MLP_encoder'],
                            scale_factor_channels = params['scale_factor_channels'], skip_VAE_added_dim = params['skip_VAE_added_dim'], saved_deterministic_model = unet, 
                                learn_decoder_variance = params['learn_decoder_variance']['status'], noise_injection_std = params['learn_decoder_sampler']['noise_std'],  noise_injection_level = params['learn_decoder_sampler']['noise_injection_level'], LocallyConnected= LocallyConnected, device=device)   #temporarily not input to check whether sending the net to device does automatically take care of this

        else:
            self.net = cVAE(VAE_latent_size = params['VAE_latent_size'], n_channels_x= self.n_channels_x+ self.add_feature_dim , sigmoid = sigmoid_activation, NPS_proj = NPSProj, device=device, combined_prediction = params['combined_prediction'],scale_factor_channels = scale_factor_channels,
                    VAE_MLP_encoder = params['VAE_MLP_encoder'], skip_VAE_added_dim = params['skip_VAE_added_dim'], saved_deterministic_model = unet, learn_decoder_variance = learn_decoder_variance, noise_injection_std = params['learn_decoder_sampler']['noise_std'],  noise_injection_level = params['learn_decoder_sampler']['noise_injection_level'], LocallyConnected=LocallyConnected )


        print('Loading model ....')

        model_links = glob.glob(out_dir + f'/Checkpoints/*-{model_year}*.pth') ### Changed 0717 when best model is saved in Checkpoint ####
        if len(model_links) == 0:
            model_links = glob.glob(out_dir + f'/Checkpoints/*final*-{model_year - 1}*.pth')   ### Changed 0717 when best model is saved in Checkpoint ####
            # if len(model_links) == 0:
            #     model_links = glob.glob(model_dir + f'/Checkpoints/*-{model_year_}*.pth') ### Changed 0717 when best model is saved in Checkpoint ####
        model_links.sort()
        if 'parallel' in out_dir:
            net_state_dict = torch.load(model_links[0], map_location=torch.device('cpu'))
            new_state_dict = {}
            for key, value in net_state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value  # Remove "module." (7 chars)
                else:
                    new_state_dict[key] = value
            self.net.load_state_dict(new_state_dict)
        else:
            self.net.load_state_dict(torch.load(model_links[0], map_location=torch.device('cpu'))) 

        try:
            del unet, params_saved
            gc.collect()
        except:
            pass 


        if lead_time is not None:
            mask =  create_mask(full_shape)[:n_train]  #create_mask(ds)[:n_train]
        else:
            mask = train_mask

        self.out_dir = out_dir
        self.net.to(device)
        self.net.eval()

        self.ds = ds
        self.obs = obs
        self.time_features = time_features
        self.params = params
        self.model_year = model_year
        self.mask_dataset = mask
        self.lead_time = lead_time
        self.device = device
        self.n_train = n_train
        self.land_mask = land_mask
        self.model_mask = model_mask
        del ds, obs 
        gc.collect()

    def create_train_set(self, masked = True):
                
        ds_train = self.ds[:self.n_train,...]
        obs_train = self.obs[:self.n_train,...]
        if masked:
            mask  = self.mask_dataset
        else:
            mask = None
        return  XArrayDataset(ds_train, obs_train, mask=mask, zeros_mask = self.zeros_mask, in_memory=False, lead_time=self.lead_time, time_features=self.time_features, aligned = True,  model = 'UNet2') 
    
    def create_test_set(self):
        
        ds_test = self.ds[self.n_train: ,...]
        obs_test = self.obs[self.n_train: ,...]
        if 'ensembles' in obs_test.dims:
            obs_test = obs_test.mean('ensembles')
        return  XArrayDataset(ds_test, obs_test, mask=None, zeros_mask = self.zeros_mask, in_memory=False, lead_time=self.lead_time, time_features=self.time_features, aligned = True,  model = 'UNet2') 
                    
    

    def predict(self, x, y,  latent_sample_size, noise_injection_sample_size, latent_dist, n_stds = 1, z_samples = None, return_preds = True ):
        if not return_preds:
            assert z_samples is None, 'With return_preds = False there is no calculations to do if z_samples are specified'

        model_mask_ = torch.from_numpy(self.model_mask.to_numpy()).unsqueeze(0)
        model_mask_ = model_mask_.expand(self.n_channels_x + self.add_feature_dim,*self.model_mask.shape)
        obs_mask = torch.from_numpy(self.land_mask.to_numpy()).unsqueeze(0)
        model_mask_ = model_mask_.to(y)
        obs_mask = obs_mask.to(y)
        
        # if latent_dist == 'recognition':
        #     if any([ mu is not None, log_var is not None]):
        #         assert all([ mu is not None, log_var is not None])
        # elif latent_dist == 'prior':
        #     if any([ cond_mu is not None, cond_log_var is not None]):
        #         assert all([  cond_mu is not None, cond_log_var is not None])

        with torch.no_grad():
            learn_decoder_variance_training_step = self.params['learn_decoder_variance']['status']
            decoder_inject_noise = self.params['learn_decoder_sampler']['status']
            if not decoder_inject_noise:
                noise_injection_sample_size = 1


            obs_mask = obs_mask.expand_as(y[0])

            # generated_output, _, _, _ , _, _ = net(y, obs_mask, x, model_mask_, sample_size = 1 ) 
            if z_samples is None:
                z_samples_loaded = False
                if self.params['learn_decoder_variance']['status']:
                    _, _, _, mu, log_var , cond_mu, cond_log_var = self.net(y, obs_mask, x, model_mask_, sample_size = self.params['learn_decoder_sampler']["decoder_sample_size"], inject_noise = decoder_inject_noise )
                else:
                    _, _, mu, log_var , cond_mu, cond_log_var = self.net(y, obs_mask, x, model_mask_, sample_size = self.params['learn_decoder_sampler']["decoder_sample_size"], inject_noise = decoder_inject_noise )
            else:
                z_samples_loaded = True
            
            if 'DetRess' not in self.out_dir:
                basic_unet = self.net.unet(x, model_mask_) if not self.net.loaded_unet else self.net.unet(x, model_mask_[0,...])
            else:
                deterministic_output = self.unet(x, model_mask_[0,...])  
            
            decoder_sampler_list = []

            if z_samples is None:
                if latent_dist == 'prior':
                    var = torch.exp(cond_log_var) + 1e-4
                    std = torch.sqrt(var)
                    z_samples =  Normal(cond_mu, std * n_stds).rsample(sample_shape=(latent_sample_size,)).to(self.device)
                elif latent_dist == 'recognition':
                    var = torch.exp(log_var) + 1e-4
                    std = torch.sqrt(var)   
                    z_samples =  Normal(mu, std * n_stds).rsample(sample_shape=(latent_sample_size,)).to(self.device)
            else:
                assert z_samples.shape[1] == y.shape[0]

            if not return_preds:
                return  z_samples, mu, log_var, cond_mu, cond_log_var
            
            else:

                if self.lead_time is None:
                    z = torch.flatten(z_samples, start_dim = 0, end_dim = 1)
                else:
                    z = z_samples.copy()

                for sample in range(noise_injection_sample_size):  
                    if self.params['learn_decoder_sampler']['status']:
                        out = self.net.generation(z, inject_noise = True)
                    else:
                        out = self.net.generation(z)

                    if 'DetRess' in self.out_dir:
                        raise NotImplementedError()
                        # generated_output = torch.unflatten(self.net.last_conv(out), dim = 0, sizes = (latent_sample_size,y.shape[0]))
                        # generated_output = generated_output + deterministic_output 
                        # generated_output = torch.flatten(generated_output, start_dim = 0, end_dim = 1)  

                    else:
                        out = torch.unflatten(out, dim = 0, sizes = (latent_sample_size,y.shape[0]))
                        out = out + basic_unet.squeeze() 
                        out = torch.flatten(out, start_dim = 0, end_dim = 1)     

                        if hasattr(self.net, 'last_conv_trainable'):
                            if self.net.last_conv_trainable is not None:
                                generated_output = self.net.last_conv_trainable(out)
                            else:
                                generated_output = self.net.last_conv(out)
                        else:
                            generated_output = self.net.last_conv(out)

                    generated_output = torch.unflatten(generated_output , dim = 0, sizes = (latent_sample_size,y.shape[0]))
                    decoder_sampler_list.append(generated_output.unsqueeze(0).squeeze(-3).to(torch.device('cpu')).numpy())

                noise_injected_output = np.concatenate(decoder_sampler_list, axis = 0).clip(0,1) * self.land_mask.values


                if z_samples_loaded is False:
                    return  noise_injected_output, z_samples, mu, log_var, cond_mu, cond_log_var
                else:
                    return  noise_injected_output, z_samples
                

    def extract_params(self, model_dir):
        params = {}
        path = glob.glob(model_dir + '/*.txt')[0]
        file = open(path)
        content=file.readlines()
        for line in content:
            if '\t' in line:
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

out_dir_x  = f'/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/output/SI/Full/results/NASA/cVAE/run_set_3_convnext'
out_dir_x_crps  = f'/space/hall7/sitestore/eccc/crd/cccma/users/rpg002/output/SI/Full/results/NASA/cVAE/run_set_3_convnext'

out_dir    = f'{out_dir_x}/N2_M12_VAL3_F12_v1.1_*_B0.01_CscaleNone_CVAE_50_LS1000_Linear_NPSproj_lr_scheduler_North_lr1e-05_batch10x1_e50_equalweights' 
out_dir_crps = f'{out_dir_x_crps}/N2_M12_VAL3_F12_v1.1_*_B0.01_CscaleNone_CVAE-DecoderSamplerV01-10-medium_50_LS1000_Linear_NPSproj_lr_scheduler_North_lr1e-05_batch2x2_e50_equalweights' 
# if '_test' in out_dir_crps:
#     from models.cvae_1001_test import cVAE
if 'DecoderSamplerV01' in out_dir_crps:
    from models.cvae_1001 import cVAE
elif 'DecoderSamplerV17' in out_dir_crps:
    from models.cvae_0717 import cVAE
else:
    from models.cvae_1001 import cVAE

model_list = glob.glob(out_dir + '/Checkpoints/*.pth')
for item in model_list:
    print(item.split('/')[-1])


model_year = 2016

obs_ref = 'NASA'
NPSProj = False if '1x1' in out_dir else True
if NPSProj:
    crs = 'NPS'
else:
    crs = '1x1'

if 'CVAE2' in out_dir:
    ensemble_error = True
else:
    ensemble_error = False

# Fine_Tuned_from_Assim = False
target_ensemble_bootstrap = False
ensemble_list = None
ensemble_mode = 'Mean'

subset_dimensions = 'North'

lead_months = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from preprocessing import load_model_data
# if Fine_Tuned_from_Assim:

if obs_ref == 'NASA':
    data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ f'/NASA*{crs}*.nc')[0] 
elif obs_ref == 'NOAA':
    assert crs == 'NPS'
    data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ f'/NOAA*{crs}*.nc')[0] 
else:
    data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

observation = xr.open_dataset(data_dir_obs)['SICN']
print("Load forecasts")
fct = load_model_data(LOC_FORECASTS_SI, obs_ref, crs, ensemble_list, ensemble_mode)
if ensemble_mode.lower() == 'mean': 
    if ensemble_error:
        fct_std = load_model_data(LOC_FORECASTS_SI, obs_ref, crs, ensemble_list, ensemble_mode = 'std')
else:
    fct = fct.transpose('time','lead_time','ensembles',...)

gc.collect()



if ensemble_error:
     ds_in_std = fct_std
###### handle nan and inf over land ############
if not NPSProj:
    ds_in = fct.where(fct<1000,np.nan) ### land is masked in model data with a large number
    # if not Fine_Tuned_from_Assim:
    #     observation = observation.where(observation<1000,np.nan)
else:
    # mask_projection = (xr.open_dataset(data_dir_obs)['mask'].rename({'x':'lon','y':'lat'}))[...,:,64:-64]
    observation = (observation.rename({'x':'lon','y':'lat'}))
    ds_in = (fct.rename({'x':'lon','y':'lat'}))
    if obs_ref == 'NASA':
        observation = observation[...,:,64:-64]
        ds_in = ds_in[...,:,64:-64]    
        if ensemble_error:
                ds_in_std = ds_in_std[...,:,64:-64] 

if 'ensembles' in observation.dims:
    land_mask = observation.isel(ensembles = 0).mean('time').where(np.isnan(observation.isel(ensembles = 0).mean('time')),1).fillna(0).drop('ensembles')
else:
    land_mask = observation.mean('time').where(np.isnan(observation.mean('time')),1).fillna(0)

model_mask = ds_in.mean('time')[0].where(np.isnan(ds_in.mean('time')[0]),1).fillna(0).drop('lead_time')
observation = observation.clip(0,1)
ds_in = ds_in.clip(0,1)
observation = observation.fillna(0)
ds_in = ds_in.fillna(0)
if ensemble_error:
    ds_in_std = ds_in_std.fillna(0)
############################################
if 'ensembles' in observation.dims:
    obs_in = observation.expand_dims('channels', axis=2)
else:
    obs_in = observation.expand_dims('channels', axis=1)

if 'ensembles' in ds_in.dims: ### PG: add channels dimention to the correct axis based on whether we have ensembles or not
    ds_in = ds_in.expand_dims('channels', axis=3)
else:
    ds_in = ds_in.expand_dims('channels', axis=2) 
    if ensemble_error:
            ds_in_std = ds_in_std.expand_dims('channels', axis=2)
            ds_in = xr.concat([ds_in, ds_in_std], dim = 'channels')




ds_raw, obs_raw = align_data_and_targets(ds_in, obs_in, lead_months, target_ensemble_bootstrap = target_ensemble_bootstrap)  # extract valid lead times and usable years ## used to be np.min(test_years)

del  obs_in
gc.collect()

if not ds_raw.time.equals(obs_raw.time): 
        ds_raw = ds_raw.sel(time = obs_raw.time)


if 'ensembles' in ds_raw.dims: ## PG: reorder dimensions in you have ensembles
    ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time','ensembles',...)
else:
    ds_raw_ensemble_mean = ds_raw.transpose('time','lead_time',...)


del ds_raw
gc.collect()

assert obs_raw.time.equals(ds_raw_ensemble_mean.time)



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

weights_mask = land_mask.copy()
weights_mask[:] = smoother(land_mask, 5)
weights_mask = weights_mask.where(weights_mask == 0 ,1).values

trained_model_mse = trained_net( out_dir,   model_year, ds_raw_ensemble_mean, obs_raw, land_mask, model_mask, NPSProj, subset_dimensions, device )
# trained_model_crps = trained_net( out_dir_crps,   model_year, ds_raw_ensemble_mean, obs_raw, land_mask, model_mask, NPSProj, subset_dimensions, device )

dict_data_mse = {}
# dict_data_crps = {}
ensemble = None

for mode in ['Test', 'Train']:
    print(f'Model = {mode}')


    if mode.lower() == 'train':
        data_set = trained_model_mse.create_train_set(masked = False)
    else:
        data_set = trained_model_mse.create_test_set()

    target_month  = np.mod(data_set.data.time.to_numpy(),100) + data_set.data.lead_time.to_numpy() - 1
    target_month = np.mod(target_month - 0.5, 12) + 0.5

    times = data_set.data.time.to_numpy()
    lead_times = data_set.data.lead_time.to_numpy()


    x_in = torch.from_numpy(data_set.data.to_numpy()).float().to(device)
    target = torch.from_numpy(data_set.target.to_numpy()).float().to(device)
    features = torch.from_numpy(data_set.time_features).float().to(device)

    del data_set
    gc.collect()



    latent_dist = 'recognition' # 'recognition' or 'prior'
    n_stds = 1
    latent_sample_size = 1
    noise_injection_sample_size = 1
    dict_data_mse[mode] = {}
    # dict_data_crps[mode] = {}



    for lead_time in tqdm(np.arange(1,13)):  #[1,6,12]

        if all([ensemble is not None, lead_time is not None]):
            inds = np.where((ensembles == ensemble) & ((lead_times == lead_time)))
        elif type(lead_time) == list:
            inds = np.where( (np.isin(lead_times, lead_time)))
        elif lead_time is not None:
            inds = np.where(lead_times == lead_time)
        else:
            inds = np.full_like(lead_times, True, dtype=bool)

        x_ = x_in[inds]
        y_ = target[inds]
        if trained_model_mse.params['time_features'] is not None:
            x_ = (x_, features[inds])

        init_times = times[inds]
        add_year, target_month = np.divmod(np.mod(init_times,100) + lead_time - 1 - 0.5, 12) 
        target_times = (np.divmod(init_times,100)[0] + add_year ) * 100 + target_month + 0.5

        z_samples = [] 
        mu = [] 
        log_var = [] 
        cond_mu = [] 
        cond_log_var = []
        latent_sampled = []
        for start_ind in tqdm(np.arange(0, y_.shape[0],10)):
            if type(x_) == tuple:
                x = (x_[0][start_ind:start_ind + 10], x_[1][start_ind: start_ind + 10])
            else:
                x = x_[start_ind: start_ind + 10]

            y = y_[start_ind: start_ind + 10]
            

            # latent_sampled_mse, z_samples = trained_model_mse.predict( x, y,  latent_sample_size, noise_injection_sample_size, latent_dist = latent_dist, n_stds = 1)
            # latent_sampled_crps, _ = trained_model_crps.predict( x, y,  latent_sample_size, noise_injection_sample_size, latent_dist = latent_dist, z_samples = z_samples)

            latent_sampled_, z_samples_, mu_, log_var_, cond_mu_, cond_log_var_ = trained_model_mse.predict( x, y,  latent_sample_size, noise_injection_sample_size, latent_dist = latent_dist, n_stds = 1)

            z_samples.append(z_samples_.cpu())
            mu.append(mu_.cpu())
            log_var.append(log_var_.cpu())
            cond_mu.append(cond_mu_.cpu())
            cond_log_var.append(cond_log_var_.cpu())
            # latent_sampled.append(latent_sampled_)
            # latent_sampled = torch.cat(latent_sampled, axis = 0)
        dict_data_mse[mode][lead_time] = dict(latent_sampled = None, z_samples = torch.cat(z_samples, axis = 1), mu = torch.cat(mu, axis = 0), 
                                              log_var = torch.cat(log_var, axis = 0), cond_mu = torch.cat(cond_mu, axis = 0), cond_log_var = torch.cat(cond_log_var, axis = 0), init_times = init_times, target_times = target_times)

        
        # latent_sampled, z_samples, mu, log_var, cond_mu, cond_log_var  = trained_model_crps.predict( x, y,  latent_sample_size, noise_injection_sample_size, latent_dist = latent_dist, n_stds = 1)
        # dict_data_crps[mode][lead_time] = dict(latent_sampled = latent_sampled, z_samples = z_samples, mu = mu, log_var = log_var, cond_mu = cond_mu, cond_log_var = cond_log_var)

import pickle
with open(f"{out_dir}/saved_model_{model_year}_latent_encodings.pkl", "wb") as f:
    pickle.dump(dict_data_mse, f)

# with open(f"{out_dir_crps}/saved_model_{model_year}_latent_encodings.pkl", "wb") as f:
#     pickle.dump(dict_data_crps, f)