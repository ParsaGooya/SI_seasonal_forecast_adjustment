import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from models.partialconv2d import PartialConv2d

class cVAE(nn.Module):
	
    def __init__( self, VAE_latent_size, n_channels_x=1 ,  sigmoid = True, NPS_proj = False,scale_factor_channels = None, combined_prediction = False, VAE_MLP_encoder = False,
                 skip_VAE_added_dim = False, saved_deterministic_model = None, freeze_deterministic = True, learn_decoder_variance = False, noise_injection_std = None, device = torch.device('cpu') ): # temporarily not input to check whether sending the net to device does automatically take care of this
        super().__init__()
        self.combined_prediction = combined_prediction
        self.VAE_MLP_encoder = VAE_MLP_encoder
        self.n_channels_x = n_channels_x
        self.loaded_unet = True if saved_deterministic_model is not None else False
        self.learn_decoder_variance = learn_decoder_variance
        self.noise_injection_std = noise_injection_std
        self.device = device

        if self.combined_prediction:
             num_obs_channels = 2
        else:
             num_obs_channels = 1


        if not NPS_proj:
            if saved_deterministic_model is None:
                self.unet = prediction(n_channels_x, sigmoid, scale_factor_channels = scale_factor_channels )
            else:
                self.unet = Trimed_unet(saved_deterministic_model)
                if freeze_deterministic:
                    for param in self.unet.parameters():
                        param.requires_grad = False

            self.recognition = prior_recognition(n_channels_x + num_obs_channels, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.prior = prior_recognition(n_channels_x + num_obs_channels, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.generation = generation(sigmoid = sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels, skip_VAE_added_dim = skip_VAE_added_dim, noise_injection_std = noise_injection_std, device=device)
        else:
            if saved_deterministic_model is None:
                self.unet = prediction_NPS(n_channels_x, sigmoid, scale_factor_channels = scale_factor_channels)
            else:
                self.unet = Trimed_unet_NPS(saved_deterministic_model)
                if freeze_deterministic:
                    for param in self.unet.parameters():
                        param.requires_grad = False

            self.recognition = prior_recognition_NPS(n_channels_x + num_obs_channels, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.prior = prior_recognition_NPS(n_channels_x + num_obs_channels, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.generation = generation_NPS(sigmoid = sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels, skip_VAE_added_dim = skip_VAE_added_dim, noise_injection_std = noise_injection_std, device = device)	
		
        if saved_deterministic_model:
            self.last_conv = saved_deterministic_model.last_conv
            if freeze_deterministic:
                for param in self.last_conv.parameters():
                        param.requires_grad = False
            if combined_prediction:
                self.last_conv2 = saved_deterministic_model.last_conv2
                if freeze_deterministic:
                    for param in self.last_conv2.parameters():
                        param.requires_grad = False
            if learn_decoder_variance:
                try:
                    self.last_conv_var = saved_deterministic_model.last_conv_var
                    if freeze_deterministic:
                        for param in self.last_conv_var.parameters():
                            param.requires_grad = False    
                except:
                    self.last_conv_var = OutConv(16 , 1, sigmoid = False, NPS_proj = True)
                    print('Saved model does not have decoder variance learned')                 
        else:		
            self.last_conv = OutConv(16  , 1, sigmoid = sigmoid, NPS_proj= NPS_proj)
            if combined_prediction:
                self.last_conv2 = OutConv(16 , 1, sigmoid = True, NPS_proj = True)
            if learn_decoder_variance:
                 self.last_conv_var = OutConv(16 , 1, sigmoid = False, NPS_proj = True)
                  
        self.N = torch.distributions.Normal(0, 1)
        # Get sampling working on GPU
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)


    # def to_device(self, device):
    #         self.device = device
    #         self.N.loc = self.N.loc.cuda()
    #         self.N.scale = self.N.scale.cuda() 
    #         if self.noise_injection_std is not None:
    #             self.generation.to_device(device)
            
    def forward(self, obs, obs_mask, model, model_mask, sample_size = 1, seed = None, nstd = 1, inject_noise = False,  mode = 'CVAE'):
            
        basic_unet = self.unet(model, model_mask) if not self.loaded_unet else self.unet(model, model_mask[0,...])
        deterministic_output = self.last_conv(basic_unet)
        if self.loaded_unet:
            with torch.no_grad(): 
                deterministic_output = deterministic_output.clip(0,1) ### PG V0717

        if self.combined_prediction:
            with torch.no_grad(): 
                    deterministic_output_extent = self.last_conv2(basic_unet)
                    deterministic_output_extent = (deterministic_output_extent >= 0.5).float() ### PG V0717
            deterministic_output = torch.cat([deterministic_output, deterministic_output_extent], dim = -3) 

        mask_recognition = torch.cat([obs_mask, model_mask], dim = 0)
        mu, log_var = self.recognition(obs, cond = model, mask = mask_recognition)
        
        mask_prior = torch.cat([model_mask, obs_mask], dim = 0)
        cond_mu, cond_log_var = self.prior(model, cond = deterministic_output, mask = mask_prior)

        if self.combined_prediction:
            deterministic_output = (deterministic_output[...,0,:,:], deterministic_output[...,1,:,:])

        if mode == 'CVAE':
            z = self.sample( mu, log_var, 1, seed, nstd = nstd)
        elif mode == 'GCGN':
            z = self.sample( cond_mu, cond_log_var, 1, seed, nstd = nstd)
    
        if inject_noise:
            z = z.expand(sample_size, *z.shape[1:])

        out_shape = z.shape
        z = torch.flatten(z, start_dim = 0, end_dim = 1)

        out = self.generation(z, inject_noise = inject_noise)
        del z

        out =  torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])  
        out = out + basic_unet.unsqueeze(0).expand_as(out)
        del basic_unet
        out = torch.flatten(out, start_dim = 0, end_dim = 1)


        generated_output = self.last_conv(out)
        generated_output = torch.unflatten(generated_output, dim = 0 , sizes = out_shape[0:2])

        if self.learn_decoder_variance:
            output_log_variance = self.last_conv_var(out)
            output_log_variance = torch.unflatten(output_log_variance, dim = 0 , sizes = out_shape[0:2])

        if self.combined_prediction:
            generated_output_extent = self.last_conv2(out)
            generated_output_extent = torch.unflatten(generated_output_extent, dim = 0 , sizes = out_shape[0:2])
            generated_output = (generated_output, generated_output_extent)

        del out
        if self.learn_decoder_variance:
            return generated_output, output_log_variance, deterministic_output, mu, log_var , cond_mu, cond_log_var
        else:
            return generated_output, deterministic_output, mu, log_var , cond_mu, cond_log_var

    def sample( self, mu, log_var, sample_size = 1, seed = None, nstd = 1):
        if seed is not None:
            current_rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        var = torch.exp(log_var) + 1e-4

        if nstd !=1:
            N = torch.distributions.Normal(0, nstd)
        # Get sampling working on GPU
            N.loc = N.loc.to(self.device)
            N.scale = N.scale.to(self.device)
            out = mu + torch.sqrt(var)*N.sample((sample_size,*mu.shape))
        else:
            out = mu + torch.sqrt(var)*self.N.sample((sample_size,*mu.shape))
        
        if seed is not None:
            torch.random.set_rng_state(current_rng_state)
        
        return out
    



class generation(nn.Module):
		
    def __init__( self, VAE_latent_size,  sigmoid = True, VAE_MLP_encoder = False, scale_factor_channels = 4, skip_VAE_added_dim = False , noise_injection_std = None, device = torch.device('cpu') ):
        
        super().__init__()
        self.VAE_MLP_encoder = VAE_MLP_encoder
        self.added_dim  = 0
        self.decoder_noise_injection = False
        self.skip_VAE_added_dim = skip_VAE_added_dim

        if VAE_MLP_encoder:
            self.combine = nn.Linear(VAE_latent_size, 256 * 6*11)
        else:
            self.combine = nn.Conv2d(VAE_latent_size, 256, kernel_size=1)
            assert skip_VAE_added_dim is False

        if noise_injection_std is not None:
            self.decoder_noise_injection = True
            self.N = torch.distributions.Normal(0, noise_injection_std)
            self.N.loc = self.N.loc.to(device)
            self.N.scale = self.N.scale.to(device)
            self.added_dim  += 1
        else:
            self.N = None
        if skip_VAE_added_dim :
            self.skip1 = nn.Linear(VAE_latent_size, 1 * 12*22)
            self.skip2 = nn.Linear(VAE_latent_size, 1 *25*45)
            self.skip3 = nn.Linear(VAE_latent_size, 1 *50 * 90)
            self.skip4 = nn.Linear(VAE_latent_size, 1 *100 * 180)

            self.added_dim  += 1
        
        if self.added_dim > 0:
            self.skip4conv =  SingleConvNext(16 + self.added_dim, 16,  multi_channel=False, return_mask=False) 


        self.up1 = Up(256 , 128, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
        self.up2 = Up(128 + self.added_dim , 64, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
        self.up3 = Up(64 + self.added_dim , 32, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
        self.up4 = Up(32 + self.added_dim , 16, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
    

    def forward(self, z, inject_noise = False):
        # Upsampling
        x = self.combine(z)
        if self.VAE_MLP_encoder:
            x = torch.unflatten(x, dim = 1, sizes = (256,6,11))

        x = self.up1(x)  # (batch, 128, 12, 22)
        if self.skip_VAE_added_dim:
            z_ = self.skip1(z)
            z_ = torch.unflatten(z_, dim = 1, sizes = (1,12,22))
            x = torch.cat([x,z_], dim = 1)
        if all([inject_noise, self.decoder_noise_injection]):
            N = self.N.sample((x.shape[0],1,12,22 ))
            x = torch.cat([x,N], dim = 1)

        x = self.up2(x, pad = (0,1,0,1))  # (batch, 64, 25, 45)
        if self.skip_VAE_added_dim:
            z_ = self.skip2(z)
            z_ = torch.unflatten(z_, dim = 1, sizes = (1,25,45))
            x = torch.cat([x,z_], dim = 1)
        if all([inject_noise, self.decoder_noise_injection]):
            N = self.N.sample((x.shape[0],1,25,45))
            x = torch.cat([x,N], dim = 1)

        x = self.up3(x)  # (batch, 32, 50, 90)
        if self.skip_VAE_added_dim:
            z_ = self.skip3(z)
            z_ = torch.unflatten(z_, dim = 1, sizes = (1,50,90))
            x = torch.cat([x,z_], dim = 1)
        if all([inject_noise, self.decoder_noise_injection]):
            N = self.N.sample((x.shape[0],1,50,90 ))
            x = torch.cat([x,N], dim = 1)
                        
        x = self.up4(x)  # (batch, 16, 100, 180)      
        if self.added_dim >0:
            if self.skip_VAE_added_dim:
                z_ = self.skip4(z)
                z_ = torch.unflatten(z_, dim = 1, sizes = (1,100,180))
                x = torch.cat([x,z_], dim = 1)
            if all([inject_noise, self.decoder_noise_injection]):
                N = self.N.sample((x.shape[0],1,100,180 ))
                x = torch.cat([x,N], dim = 1)    
            x = self.skip4conv(x)

        return x
	
class generation_NPS(nn.Module):
		
    def __init__( self, VAE_latent_size,   sigmoid = True, VAE_MLP_encoder = False , scale_factor_channels = 4 , skip_VAE_added_dim = False, noise_injection_std = None, device = torch.device('cpu')  ):

        super().__init__()
        self.VAE_MLP_encoder = VAE_MLP_encoder
        self.added_dim = 0
        self.decoder_noise_injection = False
        self.skip_VAE_added_dim = skip_VAE_added_dim

        if VAE_MLP_encoder:
            # self.combine = nn.Linear(VAE_latent_size, 512 * 13*9)  ### deeper model
            self.combine = nn.Linear(VAE_latent_size, 256 * 27*19)
        else:
            # self.combine = nn.Conv2d(VAE_latent_size, 512, kernel_size=1) ### deeper model
            self.combine = nn.Conv2d(VAE_latent_size, 256, kernel_size=1)
            assert skip_VAE_added_dim is False

        if noise_injection_std is not None:
            self.decoder_noise_injection = True
            self.N = torch.distributions.Normal(0, noise_injection_std)
            self.N.loc = self.N.loc.to(device)
            self.N.scale = self.N.scale.to(device)
            self.added_dim  += 1
        else:
            self.N = None

        if skip_VAE_added_dim :
            # self.skip1 = nn.Linear(VAE_latent_size, 1 * 27*19) ### deeper model
            self.skip2 = nn.Linear(VAE_latent_size, 1 *54*38)
            self.skip3 = nn.Linear(VAE_latent_size, 1 *108 * 76)
            self.skip4 = nn.Linear(VAE_latent_size, 1 *216 * 152)
            self.skip5 = nn.Linear(VAE_latent_size, 1 *432 * 304)

            self.added_dim  += 1
        
        if self.added_dim > 0:
            self.skip5conv =   SingleConvNext(16 + self.added_dim, 16,  multi_channel=False, return_mask=False) 

                            
        # self.up1 = Up(512, 256, scale_factor_channels = scale_factor_channels, noise_dist = self.N) ### deeper model
        self.up2 = Up(256 , 128, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
        self.up3 = Up(128+ self.added_dim , 64, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
        self.up4 = Up(64+ self.added_dim , 32, scale_factor_channels = scale_factor_channels, noise_dist = self.N)
        self.up5 = Up(32+ self.added_dim , 16, scale_factor_channels = scale_factor_channels, noise_dist = self.N)

    def forward(self, z, inject_noise = False):
        x = self.combine(z)
        if self.VAE_MLP_encoder:
            # x = torch.unflatten(x, dim = 1, sizes = (512,13,9)) ### deeper model
            x = torch.unflatten(x, dim = 1, sizes = (256,27,19))
        # Upsampling
        #################### deeper model ##################
        # x = self.up1(x, pad = (0,1,0,1))  # (batch, 256, 27, 19)
        # if self.skip_VAE_added_dim:
        #     z_ = self.skip1(z)
        #     z_ = torch.unflatten(z_, dim = 1, sizes = (1,27,19))
        #     x = torch.cat([x,z_], dim = 1)
        # if all([inject_noise, self.decoder_noise_injection]):
        #     N = self.N.sample((x.shape[0],1,27,19 ))
        #     x = torch.cat([x,N], dim = 1)
        #######################################################
        x = self.up2(x)  # (batch, 128, 54, 38)
        if self.skip_VAE_added_dim:
            z_ = self.skip2(z)
            z_ = torch.unflatten(z_, dim = 1, sizes = (1,54,38))
            x = torch.cat([x,z_], dim = 1)
        if all([inject_noise, self.decoder_noise_injection]):
            N = self.N.sample((x.shape[0],1,54,38 ))
            x = torch.cat([x,N], dim = 1)

        x = self.up3(x)  # (batch, 64, 108, 76)
        if self.skip_VAE_added_dim:
            z_ = self.skip3(z)
            z_ = torch.unflatten(z_, dim = 1, sizes = (1,108,76))
            x = torch.cat([x,z_], dim = 1)
        if all([inject_noise, self.decoder_noise_injection]):
            N = self.N.sample((x.shape[0],1,108,76 ))
            x = torch.cat([x,N], dim = 1)

        x = self.up4(x)  # (batch, 32, 216, 152)
        if self.skip_VAE_added_dim:
            z_ = self.skip4(z)
            z_ = torch.unflatten(z_, dim = 1, sizes = (1,216,152))
            x = torch.cat([x,z_], dim = 1)
        if all([inject_noise, self.decoder_noise_injection]):
            N = self.N.sample((x.shape[0],1,216,152 ))
            x = torch.cat([x,N], dim = 1)

        x = self.up5(x)  # (batch, 16, 432, 304) 
        if self.added_dim >0:    
            if self.skip_VAE_added_dim:
                z_ = self.skip5(z)
                z_ = torch.unflatten(z_, dim = 1, sizes = (1,432,304))
                x = torch.cat([x,z_], dim = 1)
            if all([inject_noise, self.decoder_noise_injection]):
                N = self.N.sample((x.shape[0],1,432,304))
                x = torch.cat([x,N], dim = 1)                      
            x = self.skip5conv(x)

        return x
	

class prior_recognition(nn.Module):
 
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, VAE_latent_size = None, VAE_MLP_encoder = False, scale_factor_channels = 4 ):
        
        super().__init__()
        self.n_channels_x = n_channels_x 
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, scale_factor_channels = scale_factor_channels)
        # self.d5 = Down(256, 512)
        # last conv of downsampling
        if VAE_latent_size is None:
              VAE_latent_size = 256
        if VAE_MLP_encoder:
            self.VAE_MLP_input_dim = 6 * 11 * 256
        else:
            self.VAE_MLP_input_dim = None
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels * 256
        self.last_conv_down = DoubleConvNext(256, 256, mid_channels=mid_channels, multi_channel=True, return_mask=False, VAE_latent_size = VAE_latent_size, VAE_MLP_input_dim = self.VAE_MLP_input_dim)


    def forward(self, x, cond, mask):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if cond is not None:
            if (type(cond) == list) or (type(cond) == tuple):    
                cond_in = torch.cat([cond[0], cond[1]], dim=1)
            else:
                cond_in = cond
            x_in = torch.cat([x_in, cond_in], dim=1)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])

        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

    # Downsampling
        x2, mask2 = self.d1(x1, mask1)  # (batch, 32, 50, 90)
        x3, mask3 = self.d2(x2, mask2)  # (batch, 64, 25, 45)
        x4, mask4 = self.d3(x3, mask3)  # (batch, 128, 12, 22)
        x5, mask5 = self.d4(x4, mask4)  # (batch, 256, 6, 11)
        
        mu, log_var = self.last_conv_down(x5, mask5)  # (batch, 256, 6, 11)     
        return mu, log_var
	
class prior_recognition_NPS(nn.Module):
 
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, VAE_latent_size = None, VAE_MLP_encoder = False, scale_factor_channels = 4 ):
        
        super().__init__()
        self.n_channels_x = n_channels_x 
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, scale_factor_channels = scale_factor_channels)
        # self.d5 = Down(256, 512, scale_factor_channels = scale_factor_channels)
        # last conv of downsampling
        if VAE_latent_size is None:
              VAE_latent_size = 256   #512 ### deeper model
        if VAE_MLP_encoder:
            self.VAE_MLP_input_dim = 27* 19 * 256   #  13 * 9 * 512  ### deeper model
        else:
            self.VAE_MLP_input_dim = None
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels * 256 # 512  ### deeper model
        self.last_conv_down = DoubleConvNext(256, 256,mid_channels=mid_channels, multi_channel=True, return_mask=False, VAE_latent_size = VAE_latent_size, VAE_MLP_input_dim = self.VAE_MLP_input_dim)
                                                # 512  ### deeper model
    def forward(self, x, cond, mask):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if cond is not None:
            if (type(cond) == list) or (type(cond) == tuple):    
                cond_in = torch.cat([cond[0], cond[1]], dim=1)
            else:
                cond_in = cond
            x_in = torch.cat([x_in, cond_in], dim=1)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 304)

    # Downsampling
        x2, mask2  = self.d1(x1, mask1)  # (batch, 32, 216, 152)
        x3, mask3  = self.d2(x2, mask2)  # (batch, 64, 108, 76)
        x4, mask4  = self.d3(x3, mask3)  # (batch, 128, 54, 38)
        x5, mask5  = self.d4(x4, mask4)  # (batch, 256, 27, 19)
        # x6, mask6  = self.d5(x5, mask5)  # (batch, 512, 13, 9) ### deeper model
        
        mu, log_var = self.last_conv_down(x5, mask5)  # (batch, 512, 27, 19)     
        return mu, log_var
    
class prediction(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True , scale_factor_channels = 4, output_layer = False):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        self.output_layer = output_layer
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, scale_factor_channels = scale_factor_channels)
        # self.d5 = Down(256, 512)
        # last conv of downsampling
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels *256
        self.last_conv_down = DoubleConvNext(256, 256,mid_channels=mid_channels, multi_channel=True, return_mask=False)
        # upsampling:
        self.up1 = Up(256, 128, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(128, 64, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(64, 32, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(32, 16, scale_factor_channels = scale_factor_channels)

        if self.output_layer:
            self.last_conv = OutConv(16  , 1, sigmoid = sigmoid, NPS_proj= False)
			# self last layer:		

    def forward(self, x, mask):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

    # Downsampling
        x2, mask2 = self.d1(x1, mask1)  # (batch, 32, 50, 90)
        x3, mask3 = self.d2(x2, mask2)  # (batch, 64, 25, 45)
        x4, mask4 = self.d3(x3, mask3)  # (batch, 128, 12, 22)
        x5, mask5 = self.d4(x4, mask4)  # (batch, 256, 6, 11)
        
        x6 = self.last_conv_down(x5, mask5)  # (batch, 256, 6, 11)
        
        # Upsampling
        x = self.up1(x6)  # (batch, 128, 12, 22)
        x = self.up2(x, pad = (0,1,0,1))  # (batch, 64, 25, 45)
        x = self.up3(x)  # (batch, 32, 50, 90)
        x = self.up4(x)  # (batch, 16, 100, 180)
        if self.output_layer:
            x = self.last_conv(x)
        
        return x
    

class prediction_NPS(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, scale_factor_channels = 4, output_layer = False ):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        self.output_layer = output_layer
    
        # input  (batch, n_channels_x, 100, 180)
        
        self.initial_conv = InitialConv(n_channels_x, 16)

        # downsampling:
        self.d1 = Down(16, 32, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, scale_factor_channels = scale_factor_channels) 
        # self.d5 = Down(256, 512, scale_factor_channels = scale_factor_channels) ### deeper model
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels * 256 # 512 ### deeper model
        # last conv of downsampling
        # self.last_conv_down = DoubleConvNext(512, 512,mid_channels=mid_channels, multi_channel=True, return_mask=False)
        self.last_conv_down = DoubleConvNext(256, 256,mid_channels=mid_channels, multi_channel=True, return_mask=False) ### deeper model

        # upsampling:

        # self.up1 = Up(512, 256, scale_factor_channels = scale_factor_channels) ### deeper model
        self.up2 = Up(256, 128, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(128, 64, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(64, 32, scale_factor_channels = scale_factor_channels)
        self.up5 = Up(32, 16, scale_factor_channels = scale_factor_channels)

        if self.output_layer:
            self.last_conv = OutConv(16  , 1, sigmoid = sigmoid, NPS_proj= True)
				
    def forward(self, x, mask, ind = None):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 304)

    # Downsampling
        x2, mask2  = self.d1(x1, mask1)  # (batch, 32, 216, 152)
        x3, mask3  = self.d2(x2, mask2)  # (batch, 64, 108, 76)
        x4, mask4  = self.d3(x3, mask3)  # (batch, 128, 54, 38)
        x5, mask5  = self.d4(x4, mask4)  # (batch, 256, 27, 19) 
        # x6, mask6  = self.d5(x5, mask5)  # (batch, 512, 13, 9) ### deeper model

        # x7 = self.last_conv_down(x6, mask6)  # (batch, 512, 13, 9) ### deeper model
        x6 = self.last_conv_down(x5, mask5)  # (batch, 256, 27, 19)

        # Upsampling
        # x = self.up1(x7, pad = (0,1,0,1))  # (batch, 256, 27, 19) ### deeper model
        x = self.up2(x6)  # (batch, 128, 54, 38)
        x = self.up3(x)  # (batch, 64, 108, 76)
        x = self.up4(x)  # (batch, 32, 216, 152)
        x = self.up5(x)  # (batch, 16, 432, 304)
        if self.output_layer:
            x = self.last_conv(x)
        
        return x
    



class DoubleConvNext(nn.Module):
    r"""Adopted from from https://github.com/m2lines/Samudra/blob/main/samudra/model.py"""
    def __init__(self, in_channels, out_channels, mid_channels=None, multi_channel=False, return_mask=False, VAE_latent_size = None, VAE_MLP_input_dim = None, noise_dist =  None  ):
        super().__init__()
        self.VAE_latent_size = VAE_latent_size
        if self.VAE_latent_size is not None:
              self.return_mask = False
              assert noise_dist is None, 'Encoder should be deterministic!'
        if mid_channels is None:
                mid_channels = out_channels
        self.return_mask = return_mask
        self.multi_channel = multi_channel
        self.VAE_MLP_input_dim = VAE_MLP_input_dim
        self.noise_dist = noise_dist
        if self.noise_dist is not None:
            added_dim = 1
        else:
            added_dim = 0

        if all([in_channels == out_channels, self.noise_dist is None]):
            self.skip_module = lambda x: x  # Identity-function required in forward pass
            self.lambda_skip = True
        else:
            self.lambda_skip = False
            self.skip_module = PartialConv2d(in_channels=in_channels + added_dim,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=False)
                
        self.conv1 = PartialConv2d(in_channels+added_dim, mid_channels, kernel_size=3, padding= 1, multi_channel=multi_channel, return_mask=True)
        # self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn1 = LayerNorm(mid_channels, data_format='channels_first' )
        self.act1 = nn.GELU()
        
        self.conv2 = PartialConv2d(mid_channels+added_dim, mid_channels, kernel_size=3, padding= 1, multi_channel=multi_channel, return_mask=True)
        # self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn2 = LayerNorm(mid_channels, data_format='channels_first' )
        self.act2 = nn.GELU()

        self.mlp = PartialConv2d(in_channels=mid_channels+added_dim,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=True)
        # self.mlp = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        if VAE_latent_size is not None:
            # self.bn_vae = nn.BatchNorm2d(out_channels)
            self.bn_vae = LayerNorm(out_channels, data_format='channels_first' )
            self.acr_vae = nn.ReLU(inplace = True)

            if VAE_MLP_input_dim is not None:
                self.mu = nn.Linear(VAE_MLP_input_dim, VAE_latent_size, bias=False ) 
                self.log_var = nn.Linear(VAE_MLP_input_dim, VAE_latent_size, bias=False ) 
            else:
                self.mu = nn.Conv2d(out_channels, VAE_latent_size, kernel_size=1, bias=False)
                self.log_var = nn.Conv2d(out_channels, VAE_latent_size, kernel_size=1, bias=False) 

    def forward(self, x, mask = None):
            if self.noise_dist is not None:
                N = self.noise_dist.sample((x.shape[0],1,x.shape[-2],x.shape[-1] ))
                x = torch.cat([x,N], dim = 1)

            if self.multi_channel:
                assert mask is not None
            if  self.lambda_skip:
                skip = self.skip_module(x)     
            else:
                skip = self.skip_module(x, mask)
            x, mask = self.conv1(x, mask)
            x = self.bn1(x)
            x = self.act1(x)

            if not self.multi_channel:
                mask = None
            if self.noise_dist is not None:
                N = self.noise_dist.sample((x.shape[0],1,x.shape[-2],x.shape[-1] ))
                x = torch.cat([x,N], dim = 1)
            x, mask = self.conv2(x, mask)
            x = self.bn2(x)
            x = self.act2(x)

            if self.noise_dist is not None:
                N = self.noise_dist.sample((x.shape[0],1,x.shape[-2],x.shape[-1] ))
                x = torch.cat([x,N], dim = 1)
            x, mask= self.mlp(x, mask)
            # x= self.mlp(x)
            x = x + skip

            if self.VAE_latent_size:
                x = self.bn_vae(x)  ## turn on if layernorm is used not batchnorm
                x = self.acr_vae(x)
                if self.VAE_MLP_input_dim is not None:
                    x = torch.flatten(x, start_dim = 1)
                mu = self.mu(x)
                log_var = self.log_var(x)
                return mu, log_var

            else: 
                if self.return_mask:
                    return x, mask
                else:
                    return x


class SingleConvNext(nn.Module):
    r"""Adopted from from https://github.com/m2lines/Samudra/blob/main/samudra/model.py"""
    def __init__(self, in_channels, out_channels,  multi_channel=False, return_mask=False, noise_dist =  None ):
        super().__init__()

        self.return_mask = return_mask
        self.multi_channel = multi_channel
        self.noise_dist = noise_dist
        if self.noise_dist is not None:
            added_dim = 1
        else:
            added_dim = 0
                # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
            self.lambda_skip = True
        else:
            self.lambda_skip = False
            self.skip_module = PartialConv2d(in_channels=in_channels+added_dim,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=False)
                
        self.conv1 = PartialConv2d(in_channels+added_dim, in_channels, kernel_size=3, padding= 1, multi_channel=multi_channel, return_mask=True)
        self.bn1 = LayerNorm(in_channels, data_format='channels_first' )
        self.act1 = nn.GELU()
        
        self.mlp = PartialConv2d(in_channels=in_channels+added_dim,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=True)


    def forward(self, x, mask = None):
            if self.noise_dist is not None:
                N = self.noise_dist.sample((x.shape[0],1,x.shape[-2],x.shape[-1] ))
                x = torch.cat([x,N], dim = 1)

            if self.multi_channel:
                assert mask is not None
            if  self.lambda_skip:
                skip = self.skip_module(x)     
            else:
                skip = self.skip_module(x, mask)
            x, mask = self.conv1(x, mask)
            x = self.bn1(x)
            x = self.act1(x)
            if not self.multi_channel:
                mask = None
            if self.noise_dist is not None:
                N = self.noise_dist.sample((x.shape[0],1,x.shape[-2],x.shape[-1] ))
                x = torch.cat([x,N], dim = 1)
                
            x, mask= self.mlp(x, mask)
            # x= self.mlp(x)
            x = x + skip

            if self.return_mask:
                return x, mask
            else:
                return x


class Down(nn.Module):
        """Downscaling with double conv then maxpool"""

        def __init__(self, in_channels, out_channels, pooling_padding = 0,scale_factor_channels = 4):
                super().__init__()
                self.maxpool = nn.MaxPool2d(2,stride = 2, padding = pooling_padding)
                if scale_factor_channels is None:
                     mid_channels = None
                else:
                     mid_channels = scale_factor_channels * in_channels

                self.doubleconv = DoubleConvNext(in_channels, out_channels,mid_channels= mid_channels, multi_channel=True, return_mask=True)	
        def forward(self, x, mask):
                x1, mask1 = self.doubleconv(x, mask)
                x1 = self.maxpool(x1)
                mask1 = self.maxpool(mask1)
                return x1, mask1


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, up_kernel = 3, scale_factor_channels = 4, noise_dist = None):
            super().__init__()
            if scale_factor_channels is None:
                mid_channels = None
            else:
                mid_channels = scale_factor_channels * in_channels


            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = PartialConv2d(in_channels, in_channels, kernel_size=3, padding=  1)
            self.conv = DoubleConvNext(in_channels, out_channels, mid_channels= mid_channels, multi_channel=False, return_mask=False, noise_dist = noise_dist)
    
    def forward(self, x, pad = None):# input is CHW
        x = self.up(x)   
        x = self.conv_mid(x)
        if pad is not None:
            x = F.pad(x, pad)
        x = self.conv(x)
        return x
		

class InitialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
            super().__init__()
            self.firstconv = PartialConv2d(in_channels, out_channels ,kernel_size=3, padding= [1,0], multi_channel=True, return_mask=True)
            # self.BN = nn.BatchNorm2d(out_channels)
            self.BN = LayerNorm(out_channels, data_format='channels_first' ) 
            self.activation = nn.ReLU(inplace=True)
    def forward(self, x, mask):
            x1 = pad_ice(x, [0,1])
            mask1 = pad_ice(mask, [0,1])
            x1, mask1 = self.firstconv(x1, mask1)
            x1 = self.BN(x1)
            x1 = self.activation(x1)
            return x1, mask1

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid = True, NPS_proj = False):
            super().__init__()
            self.NPS_proj = NPS_proj
            if NPS_proj:
                padding = 1
            else:
                padding= [1,0]
            # self.conv1 = PartialConv2d(in_channels, in_channels, kernel_size=3, padding= padding)
            if sigmoid:
                self.conv2 = nn.Sequential(
                            # nn.BatchNorm2d(in_channels),
                            LayerNorm(in_channels, data_format='channels_first' ),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid())
                    
            else:
                self.conv2 = nn.Sequential(
                            # nn.BatchNorm2d(in_channels),
                            LayerNorm(in_channels, data_format='channels_first' ),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels, out_channels, kernel_size=1))
        
    def forward(self, x):
            # if not self.NPS_proj:
            # 	x = pad_ice(x, [0,1])
            # x1 = self.conv1(x)
            return self.conv2(x)


class OutConv_logvar(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid = True, NPS_proj = False):
            super().__init__()
            self.NPS_proj = NPS_proj
            if NPS_proj:
                padding = 1
            else:
                padding= [1,0]
            # self.conv1 = PartialConv2d(in_channels, in_channels, kernel_size=3, padding= padding)
            if sigmoid:
                self.conv2 = nn.Sequential(
                            # nn.BatchNorm2d(in_channels),
                            LayerNorm(in_channels, data_format='channels_first' ),
                            nn.ReLU(inplace=True),
                            PartialConv2d(in_channels, out_channels, kernel_size=3, padding= 1), nn.Sigmoid())
                    
            else:
                self.conv2 = nn.Sequential(
                            # nn.BatchNorm2d(in_channels),
                            LayerNorm(in_channels, data_format='channels_first' ),
                            nn.ReLU(inplace=True),
                            PartialConv2d(in_channels, out_channels, kernel_size=3, padding= 1))
        
    def forward(self, x):
            # if not self.NPS_proj:
            # 	x = pad_ice(x, [0,1])
            # x1 = self.conv1(x)
            return self.conv2(x)
    


class LayerNorm(nn.Module):
    r""" Adopted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
	LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def pad_ice(x,   size): # NxCxHxW

		if type(size) in [list, tuple]:
			size_v = size[0]
			size_h = size[1]
		else:
			size_h = size_v = size
		
		if size_v >0:
			north_pad = torch.flip(x[...,-1*size_v:,:], dims=[-2])
			south_pad = torch.flip(x[...,:size_v,:], dims=[-2])
			north_pad = torch.roll(north_pad, shifts = 180, dims = [-1])  
			south_pad = torch.roll(south_pad, shifts = 180, dims = [-1])
			x = torch.cat([south_pad, x, north_pad], dim = -2 )
		if size_h > 0:
			west_pad = torch.flip(x[...,:size_h] ,dims = [-2])
			east_pad = torch.flip(x[...,-1*size_h:], dims = [-2])
			x = torch.cat([torch.flip(west_pad,dims = [-1]) , x, torch.flip(east_pad,dims = [-1])], dim = -1 )
		
		return x




class Trimed_unet(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        self.n_channels_x = original_model.n_channels_x
        self.bilinear = original_model.bilinear
        self.skip_connection = original_model.skip_connection
    
        self.initial_conv = original_model.initial_conv

        self.d1 = original_model.d1
        self.d2 = original_model.d2
        self.d3 = original_model.d3
        self.d4 = original_model.d4

        self.last_conv_down = original_model.last_conv_down

        self.up1 = original_model.up1
        self.up2 = original_model.up2 
        self.up3 = original_model.up3
        self.up4 = original_model.up4

                    
    def forward(self, x, mask, ind = None):

        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        mask = mask.unsqueeze(0)#.expand_as(x_in[0])      # uncomment if multichannel is True
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 32, 100, 180)

    # Downsampling
        x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 64, 50, 90) (batch, 32, 100, 180)
        x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 128, 25, 45) (batch, 64, 50, 90)
        x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 256, 12, 22) (batch, 128, 25, 45)
        x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 512, 6, 11) (batch, 256, 12, 22)
        
        x6 = self.last_conv_down(x5, mask5)  # (batch, 512, 6, 11)
        
        # Upsampling
        if self.skip_connection:
            x = self.up1(x6, x5_bm, mask5_bm)  # (batch, 256, 12, 22)
            x = self.up2(x, x4_bm, mask4_bm)  # (batch, 128, 25, 45)
            x = self.up3(x, x3_bm, mask3_bm)  # (batch, 64, 50, 90)
            x = self.up4(x, x2_bm, mask2_bm)  # (batch, 32, 100, 180)
        else:
            x = self.up1(x6)  # (batch, 256, 12, 22)
            x = self.up2(x)  # (batch, 128, 25, 45)
            x = self.up3(x)  # (batch, 64, 50, 90)
            x = self.up4(x)  # (batch, 32, 100, 180)			
        
        return x
    

class Trimed_unet_NPS(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        self.n_channels_x = original_model.n_channels_x
        self.bilinear = original_model.bilinear
        self.skip_connection = original_model.skip_connection
    
        self.initial_conv = original_model.initial_conv

        self.d1 = original_model.d1
        self.d2 = original_model.d2
        self.d3 = original_model.d3
        self.d4 = original_model.d4
        self.d5 = original_model.d5

        self.last_conv_down = original_model.last_conv_down

        self.up1 = original_model.up1
        self.up2 = original_model.up2 
        self.up3 = original_model.up3
        self.up4 = original_model.up4
        self.up5 = original_model.up5
                    
    def forward(self, x, mask, ind = None):

        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        mask = mask.unsqueeze(0)#.expand_as(x_in[0])   # uncomment if multichannel is True
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 32, 432, 432)

    # Downsampling
        x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 64, 216, 216) (batch, 32, 432, 432)
        x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 128, 108, 108) (batch, 64, 216, 216)
        x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 256, 54, 54)  (batch, 128, 108, 108)
        x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 512, 27, 27) (batch, 256, 54, 54)
        x6, x6_bm, mask6, mask6_bm  = self.d5(x5, mask5)  # (batch, 1024, 13, 13) (batch, 512, 27, 27)

        x7 = self.last_conv_down(x6, mask6)  # (batch, 1024, 13, 13)

        # Upsampling
        if self.skip_connection:
            x = self.up1(x7, x6_bm, mask6_bm)  # (batch, 512, 27, 27)
            x = self.up2(x, x5_bm, mask5_bm)  # (batch, 256, 54, 54)
            x = self.up3(x, x4_bm, mask4_bm)  # (batch, 128, 108, 108)
            x = self.up4(x, x3_bm, mask3_bm)  # (batch, 64, 216, 216)
            x = self.up5(x, x2_bm, mask2_bm)  # (batch, 32, 432, 432)
        else:
            x = self.up1(x7)  # (batch, 512, 27, 27)
            x = self.up2(x)  # (batch, 256, 54, 54)
            x = self.up3(x)  # (batch, 128, 108, 108)
            x = self.up4(x)  # (batch, 64, 216, 216)
            x = self.up5(x)  # (batch, 32, 432, 432)					
        
        return x
    


class cVAE_res(nn.Module):
	
    def __init__( self,   sigmoid = True, NPS_proj = False,scale_factor_channels = None, saved_CVAE_model = None, learn_decoder_variance = False, add_feature_dim = None, device = torch.device('cpu') ):
        super().__init__()

        
        self.combined_prediction = saved_CVAE_model.combined_prediction
        self.VAE_MLP_encoder = saved_CVAE_model.VAE_MLP_encoder
        self.loaded_unet = saved_CVAE_model.loaded_unet
        self.learn_decoder_variance = saved_CVAE_model.learn_decoder_variance
        self.device = device

        if learn_decoder_variance:
            n_channels_x = 3
        else:
            n_channels_x = 2

        self.unet = saved_CVAE_model.unet
        self.recognition = saved_CVAE_model.recognition
        self.prior = saved_CVAE_model.prior
        self.generation = saved_CVAE_model.generation
        self.last_conv = saved_CVAE_model.last_conv
        self.last_conv_var = saved_CVAE_model.last_conv_var
            
        if NPS_proj:
            self.residual = prediction_NPS_small(n_channels_x , sigmoid = False, scale_factor_channels = scale_factor_channels, output_layer=True )
        else:
            self.residual = prediction_small(n_channels_x , sigmoid = False, scale_factor_channels = scale_factor_channels, output_layer=True )
                  
        self.N = torch.distributions.Normal(0, 1)
        # Get sampling working on GPU
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)


    # def to_device(self, device):

    #     self.device = device
    #     self.N.loc = self.N.loc.cuda()
    #     self.N.scale = self.N.scale.cuda()        

    def forward(self, obs, obs_mask, model, model_mask, sample_size = 1, seed = None, nstd = 1, mode = 'CVAE'):
            
        basic_unet = self.unet(model, model_mask) if not self.loaded_unet else self.unet(model, model_mask[0,...])
        deterministic_output = self.last_conv(basic_unet)
        if self.combined_prediction:
            deterministic_output_extent = self.last_conv2(basic_unet)
            deterministic_output = (deterministic_output, deterministic_output_extent)

        mask_recognition = torch.cat([obs_mask, model_mask], dim = 0)
        mu, log_var = self.recognition(obs, cond = model, mask = mask_recognition)
        
        mask_prior = torch.cat([model_mask, obs_mask], dim = 0)
        cond_mu, cond_log_var = self.prior(model, cond = deterministic_output, mask = mask_prior)
        
        if mode == 'CVAE':
            z = self.sample( mu, log_var, sample_size, seed, nstd = nstd)
        elif mode == 'GCGN':
            z = self.sample( cond_mu, cond_log_var, sample_size, seed, nstd = nstd) 
        out_shape = z.shape
        z = torch.flatten(z, start_dim = 0, end_dim = 1)
        out = self.generation(z)
        del z
        out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
        out = out + basic_unet.unsqueeze(0).expand_as(out)
        out = torch.flatten(out, start_dim = 0, end_dim = 1)
        generated_output = self.last_conv(out)
            
        # feetures = torch.cat([model[0], model[1]], dim=1)[:,1:] if (type(model) == list) or (type(model) == tuple) else None
        # if feetures is not None:
        #     feetures = feetures.unsqueeze(0).expand(sample_size, *feetures.shape)
        #     inp = torch.cat([generated_output.clip(0,1), feetures], dim = -3)  ############ text with cliping
        #     del feetures
        # else:
        inp = torch.cat([generated_output.clip(0,1), model_mask[0,...].unsqueeze(0).unsqueeze(0).expand_as(generated_output)], dim = -3) ############ text with cliping
        if self.learn_decoder_variance:
            output_logvar =  self.last_conv_var(out)
            output_var = torch.exp(output_logvar)
            output_var[output_var < 0.0001] = 0
            inp = torch.cat([inp, output_var], dim = -3)
            del output_var, output_logvar

        # inp = torch.flatten(inp, start_dim = 0, end_dim = 1)
        output_residual = self.residual(inp, obs_mask.expand(inp.shape[-3:]))
        
        del inp
        output_residual = torch.unflatten(output_residual, dim = 0 , sizes = out_shape[0:2])
        generated_output = torch.unflatten(generated_output, dim = 0 , sizes = out_shape[0:2])
        if self.combined_prediction:
            generated_output_extent = self.last_conv2(out)
            generated_output_extent = torch.unflatten(generated_output_extent, dim = 0 , sizes = out_shape[0:2])
            generated_output = (generated_output, generated_output_extent)
        del out

        return generated_output, output_residual, deterministic_output, mu, log_var , cond_mu, cond_log_var


    def sample( self, mu, log_var, sample_size = 1, seed = None, nstd = 1):
        if seed is not None:
            current_rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        var = torch.exp(log_var) + 1e-4

        if nstd !=1:
            N = torch.distributions.Normal(0, nstd)
        # Get sampling working on GPU
            N.loc = N.loc.to(self.device)
            N.scale = N.scale.to(self.device)
            out = mu + torch.sqrt(var)*N.sample((sample_size,*mu.shape))
        else:
            out = mu + torch.sqrt(var)*self.N.sample((sample_size,*mu.shape))
        
        if seed is not None:
            torch.random.set_rng_state(current_rng_state)
        
        return out
    

    
class prediction_small(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True , scale_factor_channels = 4, output_layer = False):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        self.output_layer = output_layer
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, scale_factor_channels = scale_factor_channels)

        # last conv of downsampling
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels *128

        self.last_conv_down = DoubleConvNext(128, 128,mid_channels=mid_channels, multi_channel=True, return_mask=False)
        # upsampling:
        self.up1 = Up(128, 64, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(64, 32, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(32, 16, scale_factor_channels = scale_factor_channels)

        if self.output_layer:
            self.last_conv = OutConv(16  , 1, sigmoid = sigmoid, NPS_proj= False)
			# self last layer:		

    def forward(self, x, mask):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

    # Downsampling
        x2, mask2 = self.d1(x1, mask1)  # (batch, 32, 50, 90)
        x3, mask3 = self.d2(x2, mask2)  # (batch, 64, 25, 45)
        x4, mask4 = self.d3(x3, mask3)  # (batch, 128, 12, 22)
        
        x5 = self.last_conv_down(x4, mask4)  # (batch, 128, 12, 22)
        
        # Upsampling
        x = self.up1(x5, pad = (0,1,0,1))  # (batch, 64, 25, 45)
        x = self.up2(x)  # (batch, 32, 50, 90)
        x = self.up3(x)  # (batch, 16, 100, 180)
        if self.output_layer:
            x = self.last_conv(x)
        
        return x
    

class prediction_NPS_small(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, scale_factor_channels = 4, output_layer = False ):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        self.output_layer = output_layer
    
        # input  (batch, n_channels_x, 100, 180)
        
        self.initial_conv = InitialConv(n_channels_x, 16)

        # downsampling:
        self.d1 = Down(16, 32, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, scale_factor_channels = scale_factor_channels)

        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels *256
        # last conv of downsampling
        self.last_conv_down = DoubleConvNext(256, 256,mid_channels=mid_channels, multi_channel=True, return_mask=False)

        # upsampling:

        self.up1 = Up(256, 128, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(128, 64, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(64, 32, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(32, 16, scale_factor_channels = scale_factor_channels)

        if self.output_layer:
            self.last_conv = OutConv(16  , 1, sigmoid = sigmoid, NPS_proj= True)
				
    def forward(self, x, mask, ind = None):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 304)

    # Downsampling
        x2, mask2  = self.d1(x1, mask1)  # (batch, 32, 216, 152)
        x3, mask3  = self.d2(x2, mask2)  # (batch, 64, 108, 76)
        x4, mask4  = self.d3(x3, mask3)  # (batch, 128, 54, 38)
        x5, mask5  = self.d4(x4, mask4)  # (batch, 256, 27, 19)

        x6 = self.last_conv_down(x5, mask5)  # (batch, 256, 27, 19)

        # Upsampling
        x = self.up1(x6)  # (batch, 128, 54, 38)
        x = self.up2(x)  # (batch, 64, 108, 76)
        x = self.up3(x)  # (batch, 32, 216, 152)
        x = self.up4(x)  # (batch, 16, 432, 304)
        if self.output_layer:
            x = self.last_conv(x)
        
        return x
    
