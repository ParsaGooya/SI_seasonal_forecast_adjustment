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
from models.cvae_0717 import cVAE
from losses import WeightedMSE, BCElossKLD #, WeightedMSEKLD, WeightedMSELowRessKLD
from losses_new import VAElossLowRess, VAEloss
from preprocessing import align_data_and_targets, create_mask, pole_centric, reverse_pole_centric, segment, reverse_segment, pad_xarray, smoother
from preprocessing import AnomaliesScaler_v1_seasonal, AnomaliesScaler_v2_seasonal, Standardizer, Normalizer, PreprocessingPipeline, calculate_climatology, bias_adj, zeros_mask_gen
from torch_datasets import XArrayDataset
import torch.nn as nn
# from subregions import subregions
import glob
import gc
import os
# specify data directories
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
data_dir_forecast = LOC_FORECASTS_SI


from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, batch_size=32, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers = num_workers)
    return dataloader

def cleanup():
    dist.destroy_process_group()

def train_loop(rank, world_size, train_set, validation_set, training_runs, params, model_mask, obs_mask, unet, test_year, month, lead_time,  results_dir):
    
    setup(rank, world_size)

    n_channels_x = params['n_channels_x']
    add_feature_dim = params['add_feature_dim']
    sigmoid_activation = params['sigmoid_activation']
    NPSProj = params['NPSProj']

    model = params["model"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    decoder_kernel_size = params["decoder_kernel_size"]
    optimizer = params["optimizer"]
    lr = params["lr"]
    l2_reg = params["L2_reg"]
    obs_clim = params["obs_clim"]
    active_grid = params['active_grid']
    multi_ress_loss_kernel_size = params['multi_ress_loss_kernel_size']
    low_ress_loss_kernel_size = params['low_ress_loss_kernel_size']

    weights_mask = params['weights_mask']
    weights = params['weights']

    net = cVAE(VAE_latent_size = params['VAE_latent_size'], n_channels_x= n_channels_x+ add_feature_dim , sigmoid = sigmoid_activation, NPS_proj = NPSProj, combined_prediction = params['combined_prediction'], VAE_MLP_encoder = params['VAE_MLP_encoder'],
                            scale_factor_channels = params['scale_factor_channels'], skip_VAE_added_dim = params['skip_VAE_added_dim'], saved_deterministic_model = unet, freeze_deterministic = params['freeze_deterministic'],
                              learn_decoder_variance = params['learn_decoder_variance']['status'], noise_injection_std = params['learn_decoder_sampler']['noise_std'], LocallyConnected= params['LocallyConnected'], device = rank) 

    if params['saved_checkpoint_dir'] is not None:

            checkpoint_model_dir = params['checkpoint_model_dir']
            checkpoint_restart_epoch = int(checkpoint_model_dir.split('_epoch_')[1].split('.pth')[0])
            print(f'\nTest year {test_year} restart training from epoch {checkpoint_restart_epoch} ...\n')
            with open(Path(results_dir, "training_parameters.txt"), 'a') as f:
                f.write(f"\nTest year {test_year} restarted training from epoch {checkpoint_restart_epoch} ... \n")
                if params['saved_checkpoint_dir'] != 'same':
                            f.write(f"\nTest year {test_year} contintuing {params['saved_checkpoint_dir']} \n")

            net_state_dict = torch.load(checkpoint_model_dir, map_location=torch.device('cpu'))
            new_state_dict = {}
            for key, value in net_state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value  # Remove "module." (7 chars)
                else:
                    new_state_dict[key] = value
            net.load_state_dict(new_state_dict)
    else:
        checkpoint_restart_epoch = 0
    
    net = net.to(rank)  
    # find_unused_parameters = True if any([params['freeze_deterministic'], 'training_decoder_variance' in training_runs]) else False
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    dataloader = prepare(train_set, rank, world_size, batch_size=batch_size)
    num_batches = len(dataloader)

    if rank == 0:
        dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=False) 
        num_batches_val = len(dataloader_val)


    torch.cuda.set_device(rank)
    outputs = [None for _ in range(world_size)]
    data_to_gather = {}
    data_to_gather['num_batches'] = num_batches

    for training in training_runs:
        data_to_gather[training] = {}

        if training == 'training_base_CVAE':

            learn_decoder_variance_training_step = params['learn_decoder_variance']['status']
            decoder_inject_noise = params['learn_decoder_sampler']['status']

            if params['learn_decoder_variance']['offline']:
                print(f'Training CVAE model except decoder variance, rank {rank} ... \n')
                learn_decoder_variance_training_step = False
                for param in net.last_conv_var.parameters():
                        param.requires_grad = False 

            else:
                print(f'Training CVAE model, rank {rank} ... \n')

            epochs = params['epochs']
            lr = params['lr']
            multi_ress_loss_kernel_size = params['multi_ress_loss_kernel_size']
            low_ress_loss_kernel_size = params['low_ress_loss_kernel_size']
            beta_ = params['beta']

        elif training == 'training_decoder_variance':
            print(f'Optimizing decoder variance , rank {rank}...')
            learn_decoder_variance_training_step = True
            
            for param in net.parameters():
                param.requires_grad = True
            for param in net.prior.parameters():
                    param.requires_grad = False         
            for param in net.recognition.parameters():
                    param.requires_grad = False        
            for param in net.unet.parameters():
                    param.requires_grad = False     
            # for param in net.last_conv.parameters():
            #         param.requires_grad = False     
            # if params['combined_prediction']:
            #     for param in net.last_conv2.parameters():
            #             param.requires_grad = False                              

            multi_ress_loss_kernel_size = None
            low_ress_loss_kernel_size = None
            lr = 0.0001 
            beta_ = params['beta']
            if params['learn_decoder_variance']['epochs'] is not None:
                epochs = params['learn_decoder_variance']['epochs']  
                                                    
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = l2_reg)
        if params['lr_scheduler']:
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=params['start_factor'], end_factor=params['end_factor'], total_iters=params['total_iters'])
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['total_iters'], eta_min=params['end_factor'])
        ## PG: XArrayDataset now needs to know if we are adding ensemble features. The outputs are datasets that are maps or flattened in space depending on the model.

        
        if params['version'] == 'IceExtent':
            criterion = BCElossKLD(device=rank,  reduction=params['loss_reduction'])

        else:
            if params['low_ress_loss_kernel_size'] is None:  
                # criterion = WeightedMSEKLD(weights=weights, device=rank, weights_mask = weights_mask, hyperparam=1, reduction=params['loss_reduction'],   multi_ress_loss_kernel_size = multi_ress_loss_kernel_size)
                criterion = VAEloss(weights=weights, device=rank, weights_mask = weights_mask,  reduction=params['loss_reduction'], loss_area=None, 
                                    multi_ress_loss_kernel_size = multi_ress_loss_kernel_size, learn_decoder_variance = learn_decoder_variance_training_step, decoder_inject_noise = decoder_inject_noise)
        
            else:
                # criterion = WeightedMSELowRessKLD(weights=weights, device=rank, weights_mask = weights_mask, hyperparam=1, reduction=params['loss_reduction'],   kernel = params['low_ress_loss_kernel_size'])
                criterion = VAElossLowRess(weights=weights, device=rank, weights_mask = weights_mask, reduction=params['loss_reduction'], loss_area=None, 
                                            kernel = low_ress_loss_kernel_size, learn_decoder_variance = learn_decoder_variance_training_step)
        
        
        if params['combined_prediction']:
            criterion_extent = nn.BCELoss()

        epoch_loss = []
        epoch_MSE = []
        epoch_KLD = []

        epoch_loss_validation = []
        epoch_loss_validation_GCN = []
        epoch_MSE_validation = []
        epoch_KLD_validation = []

        step = 0

        best_valScore = float('inf')
        earlystopping_counter = 0

        for epoch in tqdm.tqdm(range(epochs)):
            ################################# Train ###############################
            net.train()
            dataloader.sampler.set_epoch(epoch) 
            batch_loss = 0
            batch_loss_MSE = 0
            batch_loss_KLD = 0
            
            for batch, (x, y) in enumerate(dataloader):

                if type(beta_) == dict:
                    if epoch + 1< beta_['start_epoch']:
                        beta = beta_['start']
                    else:
                        range_epochs = (beta_['end_epoch'] - beta_['start_epoch'] + 1)*num_batches
                        step_beta = np.clip((step - (beta_['start_epoch'] - 1)* num_batches) /(range_epochs),a_min = 0, a_max = None)
                        beta = beta_['start'] + (beta_['end'] - beta_['start']) * min(step_beta, 1)

                else:
                    beta = beta_
                step = step +1

                if (type(x) == list) or (type(x) == tuple):
                    x = (x[0].to(rank), x[1].to(rank))
                    model_mask = model_mask.to(x[0])
                else:
                    x = x.to(rank)
                    model_mask = model_mask.to(x)
                
                if (type(y) == list) or (type(y) == tuple):
                    y, m = (y[0].to(rank), y[1].to(rank))
                else:
                    y = y.to(rank)
                    m  = None

                optimizer.zero_grad()
                obs_mask = obs_mask.to(y).expand_as(y[0])

                if params['learn_decoder_variance']['status']:
                    generated_output, output_logvar, _, mu, log_var , cond_mu, cond_log_var = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], inject_noise = decoder_inject_noise )
                else:
                    generated_output, _, mu, log_var , cond_mu, cond_log_var = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], inject_noise = decoder_inject_noise )
                    output_logvar = None
                
                if not learn_decoder_variance_training_step:
                    output_logvar = None

                if params['combined_prediction']:
                    (y, y_extent) = (y[:,0].unsqueeze(1), y[:,1].unsqueeze(1))
                    (generated_output, generated_output_extent) = generated_output
                    loss_extent = criterion_extent(generated_output_extent, y_extent.unsqueeze(0).expand_as(generated_output_extent))
                    del generated_output_extent
                
                if not decoder_inject_noise:
                    y_ = y.unsqueeze(0).expand_as(generated_output)
                else:
                    y_ = y
                
                loss, MSE, KLD = criterion(y_, generated_output, output_logvar ,mu, log_var, cond_mu = cond_mu, cond_log_var = cond_log_var ,beta = beta, mask = m, return_ind_loss=True , print_loss = params['print_loss'])
                
                del generated_output
                if learn_decoder_variance_training_step:
                    del output_logvar
                torch.cuda.empty_cache() 

                if params['hybrid_weight'] is not None:
                    if params['learn_decoder_variance']['status']:
                        generated_output_GCGN, output_logvar_GCGN, _, _, _ , _, _ = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], mode = 'GCGN', inject_noise = decoder_inject_noise )   
                    else:
                        generated_output_GCGN, _, _, _ , _, _ = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], mode = 'GCGN', inject_noise = decoder_inject_noise )  
                        output_logvar_GCGN = None

                    if not learn_decoder_variance_training_step:
                        output_logvar_GCGN = None

                    if params['combined_prediction']:
                        (generated_output_GCGN, generated_output_GCGN_extent) = generated_output_GCGN
                        loss_extent_GCGN = criterion_extent(generated_output_GCGN_extent, y_extent.unsqueeze(0).expand_as(generated_output_extent))
                        loss_extent = loss_extent * params['hybrid_weight'] + ( 1- params['hybrid_weight']) * loss_extent_GCGN
                        del generated_output_GCGN_extent, y_extent
                        torch.cuda.empty_cache() 
                

                    loss_GCGN = criterion(y_, generated_output_GCGN,output_logvar_GCGN , mask = m , return_ind_loss=False , print_loss = False)
                    print(f'GCGN : {loss_GCGN}')
                    loss = loss * params['hybrid_weight'] + ( 1- params['hybrid_weight']) * loss_GCGN
                    del generated_output_GCGN
                    if learn_decoder_variance_training_step:
                        output_logvar_GCGN
                    torch.cuda.empty_cache() 

                if params['combined_prediction']:
                    loss = loss + loss_extent

                batch_loss += loss.item()
                batch_loss_MSE += MSE.item()
                batch_loss_KLD += KLD.item()

                loss = loss / params['grad_accumulation_steps']
                loss.backward()

                if (batch + 1) % params['grad_accumulation_steps'] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
            
            if (batch + 1) % params['grad_accumulation_steps'] != 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss.append(batch_loss)
            epoch_MSE.append(batch_loss_MSE)
            epoch_KLD.append(batch_loss_KLD)

            if params['lr_scheduler']:
                scheduler.step()

            del x, y ,y_, m, mu, log_var , cond_mu, cond_log_var, loss, MSE, KLD  
            gc.collect()
            torch.cuda.empty_cache() 
            ################################# Validation ###############################
            if rank == 0:
                net.eval()
                batch_loss_validation = 0
                batch_loss_validation_GCN = 0
                batch_loss_MSE_validation = 0
                batch_loss_KLD_validation = 0
                for batch, (x, y) in enumerate(dataloader_val):
                    with torch.no_grad():   
                            if (type(x) == list) or (type(x) == tuple):
                                x = (x[0].to(0), x[1].to(0))    
                            else:
                                x = x.to(0)                                
                            
                            if (type(y) == list) or (type(y) == tuple):
                                y, m = (y[0].to(0), y[1].to(0))
                            else:
                                y = y.to(0)
                                m  = None

                            if params['learn_decoder_variance']['status']:
                                generated_output, output_logvar, _, mu, log_var , cond_mu, cond_log_var = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], inject_noise = decoder_inject_noise )
                            else:
                                generated_output, _, mu, log_var , cond_mu, cond_log_var = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], inject_noise = decoder_inject_noise )
                                output_logvar = None
                            
                            if not learn_decoder_variance_training_step:
                                output_logvar = None


                            if params['combined_prediction']:
                                (y, y_extent) = (y[:,0].unsqueeze(1), y[:,1].unsqueeze(1))
                                (generated_output, generated_output_extent) = generated_output
                                loss_extent = criterion_extent(generated_output_extent, y_extent.unsqueeze(0).expand_as(generated_output_extent))
                                del generated_output_extent
                            
                            if not decoder_inject_noise:
                                y_ = y.unsqueeze(0).expand_as(generated_output)
                            else:
                                y_ = y
                            
                            loss, MSE, KLD = criterion(y_, generated_output, output_logvar ,mu, log_var, cond_mu = cond_mu, cond_log_var = cond_log_var ,beta = beta, mask = m, return_ind_loss=True , print_loss = False)

                            
                            del generated_output
                            if learn_decoder_variance_training_step:
                                del output_logvar
                            torch.cuda.empty_cache() 

                            
                            if params['learn_decoder_variance']['status']:
                                generated_output_GCGN, output_logvar_GCGN, _, _, _ , _, _ = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], mode = 'GCGN', inject_noise = decoder_inject_noise )   
                            else:
                                generated_output_GCGN, _, _, _ , _, _ = net(y, obs_mask, x, model_mask, sample_size = params['learn_decoder_sampler']["decoder_sample_size"], mode = 'GCGN', inject_noise = decoder_inject_noise )  
                                output_logvar_GCGN = None

                            if not learn_decoder_variance_training_step:
                                output_logvar_GCGN = None

                            if params['combined_prediction']:
                                (generated_output_GCGN, generated_output_GCGN_extent) = generated_output_GCGN
                                loss_extent_GCGN = criterion_extent(generated_output_GCGN_extent, y_extent.unsqueeze(0).expand_as(generated_output_extent))
                                loss_extent = loss_extent * params['hybrid_weight'] + ( 1- params['hybrid_weight']) * loss_extent_GCGN
                                del generated_output_GCGN_extent, y_extent
                                torch.cuda.empty_cache() 
                        

                            loss_GCGN = criterion(y_, generated_output_GCGN,output_logvar_GCGN , mask = m , return_ind_loss=False , print_loss = False)
                            if params['hybrid_weight'] is not None:
                                loss = loss * params['hybrid_weight'] + ( 1- params['hybrid_weight']) * loss_GCGN
                            del generated_output_GCGN
                            if learn_decoder_variance_training_step:
                                output_logvar_GCGN
                            torch.cuda.empty_cache() 

                            if params['combined_prediction']:
                                loss = loss + loss_extent

                            batch_loss_validation += loss.item()
                            batch_loss_validation_GCN += loss_GCGN.item()
                            batch_loss_MSE_validation += MSE.item()
                            batch_loss_KLD_validation += KLD.item()

                epoch_loss_validation.append(batch_loss_validation / num_batches_val)
                epoch_loss_validation_GCN.append(batch_loss_validation_GCN / num_batches_val)
                epoch_MSE_validation.append(batch_loss_MSE_validation / num_batches_val)
                epoch_KLD_validation.append(batch_loss_KLD_validation / num_batches_val)

                if epoch == 0:
                    best_valScore = epoch_MSE_validation[-1]
                    earlystopping_counter = 0

                del x, y ,y_, m, mu, log_var , cond_mu, cond_log_var, loss, MSE, KLD 
                gc.collect()
                torch.cuda.empty_cache() 
            ################################# Save Checkpoints ###############################
            if np.mod(epoch + 1 + checkpoint_restart_epoch , 10) == 0:
                if rank == 0:
                    fig, ax = plt.subplots(1,1, figsize=(8,5))
                    if params['version'] == 'IceExtent':
                        label = 'BCE' 
                    elif learn_decoder_variance_training_step: 
                        label = 'LLH'
                    elif decoder_inject_noise:
                        label = 'CRPS'
                    else:
                        label = 'MSE' 

                    ax.plot(np.arange(1,len(epoch_loss_validation)+1), epoch_loss_validation, color = 'r', linestyle = 'dashed', label = 'Val loss total')
                    ax.plot(np.arange(1,len(epoch_loss_validation)+1), epoch_loss_validation_GCN, color = 'k', linestyle = 'dashed', label = 'Val loss GCN')
                    ax.plot(np.arange(1,len(epoch_loss_validation)+1), epoch_MSE_validation, color = 'b', linestyle = 'dashed', label = f'Val {label} only')
                    ax.plot(np.arange(1,len(epoch_loss_validation)+1), epoch_KLD_validation, color = 'g', linestyle = 'dashed', label = 'Val KLD')

                    ax.set_title(f'Val Loss ') ###
                    ax.legend()
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    plt.show()
                    try:
                        os.remove(glob.glob(results_dir+f'/Figures/Val_loss_198101-{(test_year - n_validation_years )* 100 + month -1}_{training}_epoch_{checkpoint_restart_epoch}_*.png')[0])
                    except:
                        pass

                    plt.savefig(results_dir+f'/Figures/Val_loss_198101-{(test_year - n_validation_years )* 100 + month -1}_{training}_epoch_{checkpoint_restart_epoch}_{len(epoch_loss_validation) + checkpoint_restart_epoch}.png')
                    plt.close()



            ################################# Early Stopping ###############################
            
            if rank == 0:
                if epoch_MSE_validation[-1] < best_valScore - ( 0.02 * best_valScore):  # if new score not 5% better than best val score
                    best_valScore = epoch_MSE_validation[-1]
                    earlystopping_counter = 0
                    if test_year*100 + month <= params['ds_raw_ensemble_mean_times'][-1]:
                        nameSave = f"MODEL_V{params['version']}_198101-{test_year * 100 + month -1}"
                    else:
                        if lead_time is not None:
                                nameSave = f"MODEL_final_V{params['version']}_198101-{int(params['ds_raw_ensemble_mean_times'][-lead_time])}"
                        else:
                                nameSave = f"MODEL_final_V{params['version']}_198101-{int(params['ds_raw_ensemble_mean_times'][-1])}"
                                
                    saved_model = glob.glob(results_dir + '/Checkpoints/' + nameSave + "*.pth")
                    if len(saved_model) > 0:
                        for link in saved_model:
                            os.remove(link)
                    torch.save( net.state_dict(), results_dir + '/Checkpoints/' + nameSave + f"_epoch_{epoch + 1 + checkpoint_restart_epoch}.pth")
                    Early_Stop = False
                else:
                    if params['earlystoppingbuffer'] is not None:
                        earlystopping_counter += 1
                        if (earlystopping_counter >= params['earlystoppingbuffer']) and (epoch >= 15 ):         
                            Early_Stop = True # want to train for at least 15 epochs
                            print(f"epoch val MSE score {epoch_MSE_validation[-1]} has not decreased over {params['earlystoppingbuffer']} epochs compared to best {best_valScore} ")
                            with open(Path(results_dir, "training_parameters.txt"), 'a') as f:
                                f.write(f"\n Test year {test_year}, stopping early at {epoch + 1 + checkpoint_restart_epoch} --> epoch val MSE score {epoch_MSE_validation[-1]} has not decreased over {params['earlystoppingbuffer']} epochs compared to best {best_valScore}")
            else:
                Early_Stop = False # want to train for at least 15 epochs
 
                dist.broadcast(torch.tensor(Early_Stop, dtype=torch.bool), src=0)   
                if Early_Stop.item():           
                    print(
                        f"Stopping early at epoch {epoch}")
                    break
       
            
        data_to_gather[training]['epoch_loss'] = np.array(epoch_loss)
        data_to_gather[training]['epoch_MSE'] = np.array(epoch_MSE)
        data_to_gather[training]['epoch_KLD'] = np.array(epoch_KLD)

        data_to_gather[training]['epoch_loss_validation'] = np.array(epoch_loss_validation)
        data_to_gather[training]['epoch_loss_validation_GCN'] = np.array(epoch_loss_validation_GCN)
        data_to_gather[training]['epoch_MSE_validation'] = np.array(epoch_MSE_validation)
        data_to_gather[training]['epoch_KLD_validation'] = np.array(epoch_KLD_validation)
        data_to_gather[training]['best_valScore'] = best_valScore

        data_to_gather[training]['Early_Stop'] = Early_Stop
        data_to_gather['checkpoint_restart_epoch'] = checkpoint_restart_epoch

        dist.all_gather_object(outputs, data_to_gather)
        cleanup()




def run_training(params, n_years, n_validation_years = 0, lead_months = 12, lead_time = None, NPSProj = False,test_years = None,  n_runs=1, results_dir=None, numpy_seed=None, torch_seed=None, save = False):
    if lead_time is not None:
        assert lead_time <=lead_months, f"{lead_time} can not be greater than {lead_months}"

    if not params['masked_weights']:
        assert 'land_mask' in params['time_features']
    
    if NPSProj:
        crs = 'NPS'  
    else: 
        crs = '1x1'

    if obs_ref == 'NASA':
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ f'/NASA*{crs}*.nc')[0] 
    else:
        data_dir_obs = glob.glob(LOC_OBSERVATIONS_SI+ '/uws*.nc')[0]

    assert params['version'] in [1,2,3,1.1, 'IceExtent']

  
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

    if params['version'] == 'IceExtent':
        params['combined_prediction'] = False
        assert params['learn_decoder_variance']['status'] is False, 'Ice extent version decoder output is itself a probability distribution'
        assert params['learn_decoder_sampler']['status'] is False, 'Ice extent version decoder output is itself a probability distribution'
        assert params['multi_ress_loss_kernel_size']  is None
        assert params['low_ress_loss_kernel_size']  is None
        assert params['loss_reduction'] == 'mean', 'Loss reduction SUM not yet set in the loss ...'

    if not params['learn_decoder_variance']['status']:
        params['learn_decoder_variance']['offline'] = False


    if params['learn_decoder_sampler']['status']:
        assert params['combined_prediction'] is False, 'CRPS does not work on combined output'
        assert params['multi_ress_loss_kernel_size'] is None, 'CRPS works on grid scale'
        assert params['low_ress_loss_kernel_size'] is None, 'CRPS works on grid scale'
    else:
        params['learn_decoder_sampler']['noise_std'] = None
        
        
    if params['lr_scheduler']:
        start_factor = params['start_factor']
        end_factor = params['end_factor']
        total_iters = params['total_iters']
    else:
        start_factor = end_factor = total_iters = None
    
    if params['multi_ress_loss_kernel_size'] is not None:
        params['active_grid'] = False
        print('Warning: active_grid turned off because multi_ress_loss is on!')
        assert params['low_ress_loss_kernel_size'] is None
    if params['low_ress_loss_kernel_size'] is not None:
        assert params['multi_ress_loss_kernel_size'] is None

    print("Start training")
    print("Load observations")

    obs_in = xr.open_dataset(data_dir_obs)['SICN']
    
    ##### PG: Ensemble members to load 
    ensemble_list = params['ensemble_list']
    ###### PG: Add ensemble features to training features
    ensemble_mode = params['ensemble_mode'] ##

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
        mask_projection = (xr.open_dataset(data_dir_obs)['mask'].rename({'x':'lon','y':'lat'}))[...,:,64:-64]
        obs_in = (obs_in.rename({'x':'lon','y':'lat'}))[...,:,64:-64]
        ds_in = (ds_in.rename({'x':'lon','y':'lat'}))[...,:,64:-64]


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
        zeros_mask_full = xr.concat([zeros_mask_gen(obs_raw.isel(lead_time = 0).drop('lead_time').where(obs_raw.time<(test_year - n_validation_years) *100, drop = True ), 3) for test_year in test_years], dim = 'test_year').assign_coords(test_year = test_years)           
        
        for item in ['active_mask', 'full_ice_mask']:
            zeros_mask_full = zeros_mask_full.drop(item) if item not in params["time_features"] else zeros_mask_full
        zeros_mask_full = zeros_mask_full.drop('active_grid') if not params['active_grid'] else zeros_mask_full

        zeros_mask_full = zeros_mask_full.expand_dims('channels', axis=-3)
        if 'ensembles' in ds_raw.dims:
             zeros_mask_full = zeros_mask_full.expand_dims('ensembles', axis=2)
   
    model = params["model"]
    time_features = params["time_features"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    grad_accumulation_steps = params['grad_accumulation_steps']
    decoder_kernel_size = params["decoder_kernel_size"]
    optimizer = params["optimizer"]
    lr = params["lr"]
    l2_reg = params["L2_reg"]
    forecast_preprocessing_steps = params["forecast_preprocessing_steps"]
    observations_preprocessing_steps = params["observations_preprocessing_steps"]
    LocallyConnected = params['LocallyConnected']
    obs_clim = params["obs_clim"]
    active_grid = params['active_grid']
    multi_ress_loss_kernel_size = params['multi_ress_loss_kernel_size']
    low_ress_loss_kernel_size = params['low_ress_loss_kernel_size']

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
        params['loss_function'] = 'BCELoss'

        
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
            f"path_to_deterministic\t{params['path_to_deterministic']}\n" +
            f"learn_decoder_variance\t{params['learn_decoder_variance']}\n" +
            f"learn_decoder_sampler\t{params['learn_decoder_sampler']}\n" +
            f"loss_function\t{params['loss_function']}\n" + 
            f"time_features\t{time_features}\n" +
            f"obs_clim\t{obs_clim}\n" +
            f"ensemble_list\t{ensemble_list}\n" + ## PG: Ensemble list
            f"ensemble_mode\t{ensemble_mode}\n" + ## PG: Ensemble list
            f"epochs\t{epochs}\n" +
            f"batch_size\t{batch_size}\n" +   
            f"grad_accumulation_steps\t{grad_accumulation_steps}\n" + 
            f"optimizer\t{optimizer.__name__}\n" +
            f"lr\t{params['lr']}\n" +
            f"lr_scheduler\t{params['lr_scheduler']}: {start_factor} --> {end_factor} in {total_iters} epochs\n" + 
            f"decoder_kernel_size\t{decoder_kernel_size}\n" +
            f"forecast_preprocessing_steps\t{[s[0] if forecast_preprocessing_steps is not None else None for s in forecast_preprocessing_steps]}\n" +
            f"observations_preprocessing_steps\t{[s[0] if observations_preprocessing_steps is not None else None for s in observations_preprocessing_steps]}\n" +
            f"active_grid\t{active_grid}\n" + 
            f"multi_ress_loss_kernel_size\t{multi_ress_loss_kernel_size}\n" +
            f"low_ress_loss_kernel_size\t{low_ress_loss_kernel_size}\n" +
            f"LocallyConnected\t{LocallyConnected}\n" +
            f"equal_weights\t{params['equal_weights']}\n" + 
            f"subset_dimensions\t{subset_dimensions}\n" + 
            f"L2_reg\t{l2_reg}\n" + 
            f"loss_reduction\t{params['loss_reduction']}\n" + 
            f"VAE_latent_size\t{params['VAE_latent_size']}\n"  + 
            f"scale_factor_channels\t{params['scale_factor_channels']}\n"  + 
            f"VAE_MLP_encoder\t{params['VAE_MLP_encoder']}\n"  + 
            f"skip_VAE_added_dim\t{params['skip_VAE_added_dim']}\n" +
            f"hybrid_weight\t{params['hybrid_weight']}\n"
        )
    del ds_raw
    gc.collect()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                    month = max(np.divmod(month+1,13)[1],1)
                    print(f"\tStart run the final model ...")
                else:
                    print(f"\tStart run month {month} - {month + params['forecast_range_months'] - 1}...")

                if any([params['active_grid'],'active_mask' in params["time_features"], 'full_ice_mask' in params["time_features"]]):
                    zeros_mask = zeros_mask_full.sel(test_year = test_year).drop('test_year')
                else:
                    zeros_mask = None
                

                train_years = ds_raw_ensemble_mean.time[ds_raw_ensemble_mean.time < (test_year - n_validation_years) * 100 + month].to_numpy()
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

                if params['version']  in [3, 1.1]:
                    sigmoid_activation = False
                else:
                    sigmoid_activation = True

                params['ds_raw_ensemble_mean_times'] = ds_raw_ensemble_mean.time

                if 'land_mask' in time_features:
                    ds = xr.concat([ds, land_mask.expand_dims('channels', axis = 0)], dim = 'channels')
                
                ############################################# Prepare data ###########################################

                ds_train = ds[:n_train,...]
                obs_train = obs[:n_train,...]

                ds_validation = ds[n_train:n_train + n_validation_years*12,...]
                obs_validation = obs[n_train:n_train + n_validation_years*12,...]     

                if test_year*100 + month <= ds_raw_ensemble_mean.time[-1]:
                        ds_test = ds[n_train  + n_validation_years*12 :n_train + n_validation_years*12 + params['forecast_range_months'],...]
                        obs_test = obs[n_train  + n_validation_years*12 :n_train + n_validation_years*12 + params['forecast_range_months'],...]

                if params['masked_weights']:
                    weights_mask = land_mask.copy()
                    weights_mask[:] = smoother(land_mask, 5)
                    weights_mask = weights_mask.where(weights_mask == 0 ,1).values
                else:
                    weights_mask = None

                if NPSProj:
                    weights = (np.ones_like(ds_train.lon) * (np.ones_like(ds_train.lat.to_numpy()))[..., None])  # Moved this up
                    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon})
                    # weights = weights * land_mask_smooth   if params['masked_weights'] else weights
                    weights_ = weights * land_mask
                else:
                    weights = np.cos(np.ones_like(ds_train.lon) * (np.deg2rad(ds_train.lat.to_numpy()))[..., None])  # Moved this up
                    weights = xr.DataArray(weights, dims = ds_train.dims[-2:], name = 'weights').assign_coords({'lat': ds_train.lat, 'lon' : ds_train.lon}) # Create an DataArray to pass to Spatialnanremove()  
                    ####################################################################
                    weights_ = weights * land_mask
                    if params['equal_weights']:
                        weights = xr.ones_like(weights)

                del ds, obs
                gc.collect()
                torch.cuda.empty_cache() 
                torch.cuda.synchronize() 
                weights = weights.values
                weights_ = weights_.values
                params['weights'] = weights
                params['weights_mask'] = weights_mask
                if lead_time is not None:
                    mask =  create_mask(full_shape)[:n_train]  #create_mask(ds)[:n_train]
                    val_mask = create_mask(full_shape[n_train:n_train + n_validation_years *12] )[:, lead_time - 1][..., None] ####create_mask(ds_raw_ensemble_mean)[n_train:n_train + num_val_years*12] 
                else:
                    val_mask = create_mask(ds_validation)
                    mask = train_mask

                train_set = XArrayDataset(ds_train, obs_train, mask=mask, zeros_mask = zeros_mask, in_memory=False, lead_time=lead_time, time_features=time_features, aligned = True, model = 'UNet2') 
                # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                
                validation_set = XArrayDataset(ds_validation, obs_validation, mask=val_mask, zeros_mask = zeros_mask, lead_time=lead_time, time_features=time_features,  in_memory=False, aligned = True,  model = 'UNet2') 
                # dataloader_val = DataLoader(validation_set, batch_size=batch_size, shuffle=False)   
                

                ############################################## Prepare model #######################################
                if time_features is None:
                        add_feature_dim = 0
                else:
                        add_feature_dim = len(time_features)
                if 'land_mask' in time_features:
                    add_feature_dim -= 1


                n_channels_x = len(ds_train.channels)

                model_mask_ = torch.from_numpy(model_mask.to_numpy()).unsqueeze(0).expand(n_channels_x + add_feature_dim,*model_mask.shape)
                obs_mask = torch.from_numpy(land_mask.to_numpy()).unsqueeze(0)

                params['n_channels_x'] = n_channels_x
                params['add_feature_dim'] = add_feature_dim
                params['sigmoid_activation'] = sigmoid_activation
                params['NPSProj'] = NPSProj
                
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
                    if test_year*100 + month <= ds_raw_ensemble_mean.time[-1]:
                        saved_model_year_dir = f'*-{(test_year - n_val_year_saved) * 100 + month -1}*.pth'
                    else:
                        saved_model_year_dir = f'*final*.pth'

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
                    from models.unetconvnext import UNet2,UNet2_NPS
                    model = eval(params_saved['model'])
                    unet = model(n_channels_x= n_channels_x+ add_feature_dim , bilinear = params_saved['bilinear'], sigmoid = sigmoid_activation, skip_conv = params_saved['skip_conv'], combined_prediction = params_saved['combined_prediction'], LocallyConnected = LocallyConnected_saved)
                    saved_model_dir = glob.glob(params['path_to_deterministic']+ '/Checkpoints/' +saved_model_year_dir)
                    saved_model_dir.sort()
                    try:
                        unet.load_state_dict(torch.load(saved_model_dir[0], map_location=torch.device('cpu')))
                    except:
                        print(f'No saved deterministic model found for test year {test_year}')
                        break
                else:
                    unet = None

                if params['saved_checkpoint_dir'] is not None:
                    if params['saved_checkpoint_dir'] == 'Same':
                        saved_checkpoint_dir = results_dir
                    else:
                        saved_checkpoint_dir = params['saved_checkpoint_dir'] 

                    if test_year*100 + month <= params['ds_raw_ensemble_mean_times'][-1]:
                        nameSave = f"MODEL_V{params['version']}_198101-{test_year * 100 + month -1}*"
                    else:
                        if lead_time is not None:
                                nameSave = f"MODEL_final_V{params['version']}_198101-{int(params['ds_raw_ensemble_mean_times'][-lead_time])}*"
                        else:
                                nameSave = f"MODEL_final_V{params['version']}_198101-{int(params['ds_raw_ensemble_mean_times'][-1])}*"

                    try:
                        params['checkpoint_model_dir'] = glob.glob(saved_checkpoint_dir + '/Checkpoints/' + nameSave )[0]

                    except:
                        print("No checkpoints found !")
                        break

                if all([params['learn_decoder_variance']['offline'], params['learn_decoder_variance']['status']]):
                    training_runs = ['training_base_CVAE', 'training_decoder_variance']
                else:
                    training_runs = ['training_base_CVAE']


                # world_size = torch.cuda.device_count()
                world_size =  len(os.environ.get("CUDA_VISIBLE_DEVICES", "Not set").split(','))
                print(f'number of available GPUs : {world_size} \n')

                if world_size>1:
                    params['print_loss'] = False
                else:
                    params['print_loss'] = True
                # breakpoint()
                #################################################### Train/Eval parallalization #############################################

                mp.spawn(train_loop, args = (world_size, train_set, validation_set, training_runs, params, model_mask_, obs_mask, unet, test_year, month, lead_time, results_dir) , nprocs = world_size)
                outputs = torch.load(results_dir + "/outputs.pt")
                checkpoint_restart_epoch = outputs[0]['checkpoint_restart_epoch']
                os.remove(results_dir + "/outputs.pt")
                del train_set, ds_train, obs_train

                for training in training_runs:
                        total_batches = np.sum([outputs[gpu_ind]['num_batches'] for  gpu_ind in range(world_size) ])
                        
                        for gpu_ind in range(world_size):
                            if gpu_ind == 0:
                                epoch_loss = (outputs[0][training]['epoch_loss'] )
                                epoch_MSE = (outputs[0][training]['epoch_MSE'])
                                epoch_KLD =  (outputs[0][training]['epoch_KLD'])   

                                epoch_loss_validation = (outputs[0][training]['epoch_loss_validation'] )
                                epoch_loss_validation_GCN = (outputs[0][training]['epoch_loss_validation_GCN'] )
                                epoch_MSE_validation = (outputs[0][training]['epoch_MSE_validation'])
                                epoch_KLD_validation =  (outputs[0][training]['epoch_KLD_validation']) 
                                best_valScore =  (outputs[0][training]['best_valScore'])    

                                Early_Stop =  (outputs[0][training]['Early_Stop'])                 
                            else:
                                epoch_loss = epoch_loss +  (outputs[gpu_ind][training]['epoch_loss'] )
                                epoch_MSE = epoch_MSE + (outputs[gpu_ind][training]['epoch_MSE'])
                                epoch_KLD = epoch_KLD + (outputs[gpu_ind][training]['epoch_KLD'])                                            
                            
                        epoch_loss = epoch_loss / total_batches
                        epoch_MSE = epoch_MSE / total_batches
                        epoch_KLD = epoch_KLD / total_batches


                ################################# Plot ###############################
                        fig, ax = plt.subplots(1,1, figsize=(8,5))
                        if params['version'] == 'IceExtent':
                            label = 'BCE' 
                        elif training == 'training_decoder_variance': 
                            label = 'LLH'
                        elif params['learn_decoder_sampler']['status']:
                            label = 'CRPS'
                        else:
                            label = 'MSE' 
                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_loss, color = 'r', label = 'Epoch loss total')
                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_MSE, color = 'b', linestyle = 'dashed', label = f'Epoch {label} only')
                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_KLD, color = 'g', linestyle = 'dotted', label = 'Epoch KLD')

                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_loss_validation, color = 'r', linestyle = 'dashed', label = 'Val loss total')
                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_loss_validation_GCN, color = 'r', linestyle = 'dashed', label = 'Val loss GCN')
                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_MSE_validation, color = 'b', linestyle = 'dashed', label = f'Val {label} only')
                        ax.plot(np.arange(1,len(epoch_loss)+1), epoch_KLD_validation, color = 'g', linestyle = 'dashed', label = 'Val KLD')

                        ax.set_title(f'Train/Val Loss  - best val loss : {best_valScore}  ') ###
                        ax.legend()
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        plt.show()
                        plt.savefig(results_dir+f'/Figures/train_val_loss_198101-{(test_year - n_validation_years ) * 100 + month -1}_{training}_epoch_{checkpoint_restart_epoch}_{len(epoch_loss) + checkpoint_restart_epoch}.png')
                        plt.close()

                del train_set, dataloader, ds_train, obs_train,
                del validation_set, dataloader_val, ds_validation, obs_validation
                gc.collect()
                torch.cuda.empty_cache() 
              
                # Test years
                ##################################################################################################################################
                if test_year*100 + month <= params['ds_raw_ensemble_mean_times'][-1]:

                    nameSave = f"MODEL_V{params['version']}_198101-{test_year * 100 + month -1}_epoch_*.pth"

                    net = cVAE(VAE_latent_size = params['VAE_latent_size'], n_channels_x= n_channels_x+ add_feature_dim , sigmoid = sigmoid_activation, NPS_proj = NPSProj, combined_prediction = params['combined_prediction'], VAE_MLP_encoder = params['VAE_MLP_encoder'],
                            scale_factor_channels = params['scale_factor_channels'], skip_VAE_added_dim = params['skip_VAE_added_dim'], saved_deterministic_model = unet, freeze_deterministic = params['freeze_deterministic'],
                              learn_decoder_variance = params['learn_decoder_variance']['status'], noise_injection_std = params['learn_decoder_sampler']['noise_std'], device = 0) 

                    net_state_dict = torch.load(results_dir + '/Checkouts/' + nameSave, map_location=torch.device('cpu'))
                    new_state_dict = {}
                    for key, value in net_state_dict.items():
                        if key.startswith('module.'):
                            new_state_dict[key[7:]] = value  # Remove "module." (7 chars)
                        else:
                            new_state_dict[key] = value
                    net.load_state_dict(new_state_dict)
                    net = net.to(0)  
                    
                    test_years_list = np.arange(1, ds_test.shape[0] + 1)
                    test_lead_time_list = np.arange(1, ds_test.shape[1] + 1)

        
                    ## PG: Extract the number of years as well 
                    test_set = XArrayDataset(ds_test, obs_test, lead_time=lead_time,mask = None,zeros_mask = zeros_mask, time_features=time_features, in_memory=False, aligned = True, model = 'UNet2')
                    if params['version'] == 'IceExtent':   
                        criterion_test = nn.BCELoss( reduction='mean')
                    else:
                        criterion_test =  WeightedMSE(weights=weights_, device=0, hyperparam=1, reduction='mean')


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
                                test_raw = (x[0].unsqueeze(0).to(0), x[1].unsqueeze(0).to(0))
                                model_mask_ = model_mask_.to(test_raw[0])
                            else:
                                test_raw = x.unsqueeze(0).to(0)
                                model_mask_ = model_mask_.to(test_raw)
                            if (type(target) == list) or (type(target) == tuple):
                                test_obs, m = (target[0].unsqueeze(0).to(0), target[1].unsqueeze(0).to(0))
                            else:
                                test_obs = target.unsqueeze(0).to(0)
                                m = None
                                
                            obs_mask = obs_mask.to(test_obs).expand_as(test_obs[0])
                            del x, target
                            if params['learn_decoder_variance']['status']:
                                _, _, deterministic_output, _, _, cond_mu, cond_log_var = net(test_obs, obs_mask, test_raw, model_mask_, sample_size = 1, inject_noise = params['learn_decoder_sampler']['status'] )
                            else:
                                _, deterministic_output, _, _, cond_mu, cond_log_var = net(test_obs, obs_mask, test_raw, model_mask_, sample_size = 1, inject_noise = params['learn_decoder_sampler']['status'] )

                            basic_unet = net.unet(test_raw, model_mask_) if not net.loaded_unet else net.unet(test_raw, model_mask_[0,...])
                            cond_var = torch.exp(cond_log_var) + 1e-4
                            cond_std = torch.sqrt(cond_var)

                            z =  Normal(cond_mu, cond_std).rsample(sample_shape=(params['BVAE'],)).squeeze().to(0)
                            # z = torch.unflatten(z , dim = -1, sizes = cond_std.shape[-3:])
                            out = net.generation(z, inject_noise = params['learn_decoder_sampler']['status']) + basic_unet.squeeze() 
                            generated_output = net.last_conv(out)
                            output_logvar = net.last_conv_var(out) if params['learn_decoder_variance']['status'] else None

                            if params['combined_prediction']:
                                generated_output_extent = net.last_conv2(out)
                                (deterministic_output, deterministic_output_extent) = deterministic_output
                                (test_obs, test_obs_extent) = (test_obs[:,0].unsqueeze(1), test_obs[:,1].unsqueeze(1))
                                test_results_extent[:,i,] = generated_output_extent.squeeze().to(torch.device('cpu')).numpy()
                                test_results_deterministic_extent[i,] = deterministic_output_extent.squeeze().to(torch.device('cpu')).numpy()
                            del z, out
                            torch.cuda.empty_cache() 

                            if params['learn_decoder_variance']['status']:
                                epsilon = torch_normal_sampling(torch.zeros(generated_output.shape).to(generated_output), torch.sqrt(torch.exp(output_logvar) + 1e-4), num_samples=  1 ).to(device)[0]
                                epsilon[torch.exp(output_logvar) < 0.0001] = 0 
                                generated_output = generated_output + epsilon

                            if m is not None:
                                m[m != 0] = 1
                            if params['version'] == 'IceExtent':
                                loss = criterion_test(torch.mean(generated_output, 0).unsqueeze(0), test_obs)
                            else:
                                loss = criterion_test(test_obs, torch.mean(generated_output, 0).unsqueeze(0) , mask = m)



                            test_results[:,i,] = generated_output.squeeze().to(torch.device('cpu')).numpy()
                            test_results_deterministic[i,] = deterministic_output.squeeze().to(torch.device('cpu')).numpy()
                            test_loss[i] = loss.item()
                    del  test_set , test_raw, test_obs, m, deterministic_output,  generated_output , ds_test, obs_test, cond_mu, cond_var, cond_std, basic_unet
                    gc.collect()
                    torch.cuda.empty_cache() 
                    ###################################################### reverse preprocessing steps #####################################################################
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
                    # if params['masked_weights']:
                    #     result = (result * land_mask)
                    #     result_deterministic = (result_deterministic * land_mask)
                    
                    if not NPSProj:
                            result = reverse_pole_centric(result, subset_dimensions)
                            result_deterministic = reverse_pole_centric(result_deterministic, subset_dimensions)
                            reverse_pole_centric(land_mask, subset_dimensions).to_dataset(name  = 'land_mask').to_netcdf(path=Path(results_dir, f'obs_land_mask.nc', mode='w'))
                    else:
                            land_mask.to_dataset(name  = 'land_mask').to_netcdf(path=Path(results_dir, f'obs_land_mask.nc', mode='w'))

                    ##############################################################################################################################################################
                    result = result.to_dataset(name = 'nn_adjusted') 
                    result_deterministic = result_deterministic.to_dataset(name = 'nn_adjusted')  

                    if params['version'] == 'IceExtent':
                        
                        result = result.where(result >= 0.5, 0)
                        result = result.where(result ==0, 1)

                        result_deterministic = result_deterministic.where(result_deterministic >= 0.5, 0)
                        result_deterministic = result_deterministic.where(result_deterministic ==0, 1)

                    if active_grid:
                        if not NPSProj:
                            zeros_mask_test = reverse_pole_centric(zeros_mask_test)
                        else:
                            zeros_mask_test = zeros_mask_test.rename({'lon':'x', 'lat':'y'})
                        result = xr.combine_by_coords([result * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])
                        result_deterministic = xr.combine_by_coords([result_deterministic * zeros_mask_test, zeros_mask_test.to_dataset(name = 'active_grid')])

                    if params['combined_prediction']:
                        # if params['masked_weights']:
                        #     result_extent = (result_extent * land_mask)
                        #     result_extent_deterministic = (result_extent_deterministic * land_mask)
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

                    del result,result_deterministic, net
                    gc.collect()
                    torch.cuda.empty_cache()   

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
    
def torch_normal_sampling( mu, std, num_samples = 1, truncated_dist = None, multivariate = False ):
    from torch.distributions.multivariate_normal import MultivariateNormal

    if truncated_dist is not None:
        samples = []
        while len(samples) < num_samples:
            if multivariate:
                sample =  MultivariateNormal(mu, covariance_matrix=std).rsample(sample_shape=(num_samples,))
            else:
                sample =  Normal(mu, std).rsample(sample_shape=(num_samples,))
            # Keep only samples within the bounds
 
            if isinstance(truncated_dist, np.ndarray):
                truncated_dist = torch.from_numpy(truncated_dist)

            sample_dists =  np.sqrt(((sample - sample.mean(axis = 0))**2).sum(-1))
            sample = sample[sample_dists <= truncated_dist]

            samples.append(sample)
        # Concatenate all valid samples and return the required number
        return torch.cat(samples)[:num_samples]
    else:
        if multivariate:
                z =  MultivariateNormal(mu, covariance_matrix=std).rsample(sample_shape=(num_samples,))
        else:
                z =  Normal(mu, std).rsample(sample_shape=(num_samples,))
        return z
    

if __name__ == "__main__":

  
    n_years =  1 # last n years to test consecutively
    lead_months = 12
    lead_time = None ## None for training using all available lead_times as indicated ny lead_months
    n_runs = 1  # number of training runs
    n_validation_years = 3

    params = {
        "model": cVAE,
        "path_to_deterministic" : None, # '/space/hall5/sitestore/eccc/crd/ccrn/users/rpg002/output/SI/Full/results/NASA/UNet2/run_set_3_convnext/N5_M12_F12_v1.1_1x1_North_lr0.001_batch100_e50_LNone_bilinear_low_ress_loss16', 
        "freeze_deterministic" : False,
        "learn_decoder_variance" : dict( status= False,  offline = True, epochs = 100), 
        'learn_decoder_sampler' : dict( status= True,  noise_std =1, decoder_sample_size = 100), 
        "time_features": ['month_sin','month_cos', 'imonth_sin', 'imonth_cos','land_mask'],
        "obs_clim" : False,
        'ensemble_list' : None, ## PG
        'ensemble_mode' : 'Mean',
        "epochs": 75,
        "batch_size": 5,
        "grad_accumulation_steps" : 5,  # default 1
        "beta" : 1,
        "optimizer": torch.optim.Adam,
        "lr": 0.0001 ,
        "loss_function" :'MSE',
        "subset_dims": 'North',   ## North or South or Global
        'active_grid' : False,
        'multi_ress_loss_kernel_size' : None,
        'low_ress_loss_kernel_size' : None,
        'equal_weights' : True,
        'masked_weights' : True,
        "decoder_kernel_size" : 1, ## only for (Reg)CNN
        "L2_reg": 0,
        'lr_scheduler' : False,
        'VAE_latent_size' : 1000,
        'VAE_MLP_encoder' : True,
        'scale_factor_channels' : None,
        'LocallyConnected' : False,
        'BVAE' : 50,
        'loss_reduction' : 'mean' , # mean or sum
        'combined_prediction' : False,
        'hybrid_weight' :   None, ### CVAE weight
        'skip_VAE_added_dim' : True,
        'saved_checkpoint_dir' : None,   # None, 'same', or dir to model
        'earlystoppingbuffer' : None, ## buffer number
    }



    params['version'] =  1.1  ### 1 , 2 ,3 , 'IceExtent'
    params['forecast_range_months'] = 12
    params['beta'] =  0.1 #dict(start = 0, end =0.1, start_epoch = 1 , end_epoch = params['epochs'])  

    obs_ref = 'NASA'
    NPSProj = False
    
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
    if params['skip_VAE_added_dim']:
        model_type = 'skip-' + model_type
    if params['LocallyConnected']:
        model_type = model_type + '-LC2D'

    if params['learn_decoder_variance']['status']:
        assert params['learn_decoder_sampler']['status'] is False
        if not params['learn_decoder_variance']['offline']:
            model_type = model_type + '-DecoderVar'
            params['learn_decoder_variance']['epochs'] = None
        else:
            model_type = model_type + '-DecoderVarOffline'

    if params['learn_decoder_sampler']['status']:
        assert params['learn_decoder_variance']['status'] is False
        model_type = model_type + f'-DecoderSamplerV17-{params["learn_decoder_sampler"]["decoder_sample_size"]}'



    if lead_time is None:
        out_dir    = f'{out_dir_x}/N{n_years}_M{lead_months}_VAL{n_validation_years}'
    else:
        out_dir    = f'{out_dir_x}/N{n_years}_LT{lead_time}_VAL{n_validation_years}'
    
    out_dir = out_dir + f'_F{params["forecast_range_months"]}_v{params["version"]}_parallel_*_{beta_arg}_Cscale{params["scale_factor_channels"]}_{model_type}_{params["BVAE"]}_LS{params["VAE_latent_size"]}'


    if params['VAE_MLP_encoder']:
        out_dir = out_dir + '_Linear'

    out_dir = out_dir + '_NPSproj' if NPSProj else out_dir + '_1x1'

    if params['lr_scheduler']:
        out_dir = out_dir + '_lr_scheduler'
        params['start_factor'] = 1.0
        params['end_factor'] = 0.1
        params['total_iters'] = params['epochs']

    if type(params["grad_accumulation_steps"]) != int:
        params["grad_accumulation_steps"] = 1
    out_dir  = out_dir + f'_{params["subset_dims"]}_lr{params["lr"]}_batch{params["batch_size"]}x{params["grad_accumulation_steps"]}_e{params["epochs"]}_equalweights'

    if params['subset_dims'] == 'Global':
        params['subset_dimensions'] = None
    else:
        params['subset_dimensions'] = params['subset_dims']

    if params['active_grid']:
        out_dir = out_dir + '_active_grid'
    if params['combined_prediction']:
        out_dir = out_dir + '_combined'  
    if params['multi_ress_loss_kernel_size'] is not None:
        out_dir = out_dir + f'_multiressloss{params["multi_ress_loss_kernel_size"]}'
    if params['low_ress_loss_kernel_size'] is not None:
         out_dir = out_dir + f'_low_ress_loss{params["low_ress_loss_kernel_size"]}'
    if not params['masked_weights']:
        out_dir = out_dir + '_weightsnonmaked'  
    if params['path_to_deterministic'] is not None:
        if 'low_ress_loss' in params['path_to_deterministic']:
            out_dir = out_dir + '_pretrainedUNETlowress'
        elif 'multi_ress_loss' in params['path_to_deterministic']:
            out_dir = out_dir + '_pretrainedUNETmultiress'  
        else:
            out_dir = out_dir + '_pretrainedUNET'   
        if not params['freeze_deterministic']:
             out_dir = out_dir + 'notforzen'

    if params['saved_checkpoint_dir'] != 'Same':
        if params['saved_checkpoint_dir'] is not None:
            out_dir = out_dir + '_FromCheckPoint'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir + '/Figures').mkdir(parents=True, exist_ok=True)
        Path(out_dir + '/Checkpoints').mkdir(parents=True, exist_ok=True)
    else:
        out_dir_sub_paths = out_dir.split(f'e{params["epochs"]}')
        out_dir = out_dir_sub_paths[0]
        for sub_path in out_dir_sub_paths[1:]:
          out_dir = out_dir + '*' + sub_path  
        out_dir = glob.glob(out_dir)
        out_dir.sort()
        out_dir = out_dir[-1]    
    
    try:
        run_training(params, n_years=n_years, n_validation_years =n_validation_years, lead_months=lead_months,lead_time = lead_time, NPSProj  = NPSProj, test_years = None, n_runs=n_runs, results_dir=out_dir, numpy_seed=1, torch_seed=1, save = True)
        print(f'Output dir: {out_dir}')
        print('Training done.')

    except Exception as e:
        import shutil
        shutil.rmtree(out_dir)
        print("Terminated due to the follwoing error:\n", e)
        raise  # 





