import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from models.partialconv2d import PartialConv2d

class cVAE(nn.Module):
	
    def __init__( self, VAE_latent_size, n_channels_x=1 ,  sigmoid = True, NPS_proj = False,scale_factor_channels = None, combined_prediction = False, VAE_MLP_encoder = False, device = 'cpu' ):
        super().__init__()
        self.combined_prediction = combined_prediction
        self.VAE_MLP_encoder = VAE_MLP_encoder
        self.n_channels_x = n_channels_x
        if not NPS_proj:
            self.unet = prediction(n_channels_x, sigmoid, scale_factor_channels = scale_factor_channels )
            self.recognition = prior_recognition(n_channels_x + 1, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.prior = prior_recognition(n_channels_x + 1, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.generation = generation(sigmoid = sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
        else:
            self.unet = prediction_NPS(n_channels_x, sigmoid, scale_factor_channels = scale_factor_channels)
            self.recognition = prior_recognition_NPS(n_channels_x + 1, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.prior = prior_recognition_NPS(n_channels_x + 1, sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)
            self.generation = generation_NPS(sigmoid = sigmoid, VAE_latent_size = VAE_latent_size, VAE_MLP_encoder = VAE_MLP_encoder, scale_factor_channels = scale_factor_channels)	
				
        self.last_conv = OutConv(16, 1, sigmoid = sigmoid, NPS_proj= NPS_proj)
        if combined_prediction:
            self.last_conv2 = OutConv(32, 1, sigmoid = True, NPS_proj = True)
                  
        self.N = torch.distributions.Normal(0, 1)
            # Get sampling working on GPU
        if device.type == 'cuda':
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        
    def forward(self, obs, obs_mask, model, model_mask, sample_size = 1, seed = None, nstd = 1):
            
        basic_unet = self.unet(model, model_mask)
        deterministic_output = self.last_conv(basic_unet)
        if self.combined_prediction:
            deterministic_output_extent = self.last_conv2(basic_unet)
            deterministic_output = (deterministic_output, deterministic_output_extent)

        obs_mask = obs_mask.expand_as(obs[0])
        if (type(model) == list) or (type(model) == tuple):  
            model_mask = model_mask.expand((self.n_channels_x , *model_mask.shape[1:]))


        mask_recognition = torch.concat([obs_mask, model_mask], dim = 0)
        mu, log_var = self.recognition(obs, cond = model, mask = mask_recognition)
        
        mask_prior = torch.concat([model_mask, torch.ones_like(deterministic_output[0]).to(deterministic_output)], dim = 0)
        cond_mu, cond_log_var = self.prior(model, cond = deterministic_output, mask = mask_prior)
        
        z = self.sample( mu, log_var, sample_size, seed, nstd = nstd)
        out_shape = z.shape
        z = torch.flatten(z, start_dim = 0, end_dim = 1)
        out = self.generation(z)
        out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
        out = out + basic_unet.unsqueeze(0).expand_as(out)
        out = torch.flatten(out, start_dim = 0, end_dim = 1)
        generated_output = self.last_conv(out)
        generated_output = torch.unflatten(generated_output, dim = 0 , sizes = out_shape[0:2])
        if self.combined_prediction:
            generated_output_extent = self.last_conv2(out)
            generated_output_extent = torch.unflatten(generated_output_extent, dim = 0 , sizes = out_shape[0:2])
            generated_output = (generated_output, generated_output_extent)

        return generated_output, deterministic_output, mu, log_var , cond_mu, cond_log_var

    def sample( self, mu, log_var, sample_size = 1, seed = None, nstd = 1):
        if seed is not None:
            current_rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
        var = torch.exp(log_var) + 1e-4

        if nstd !=1:
            N = torch.distributions.Normal(0, nstd)
        # Get sampling working on GPU
            if self.device.type == 'cuda':
                N.loc = N.loc.cuda()
                N.scale = N.scale.cuda()
            out = mu + torch.sqrt(var)*N.sample((sample_size,*mu.shape))
        else:
            out = mu + torch.sqrt(var)*self.N.sample((sample_size,*mu.shape))
        
        if seed is not None:
            torch.random.set_rng_state(current_rng_state)
        
        return out
    

class generation(nn.Module):
		
    def __init__( self, VAE_latent_size,  sigmoid = True, VAE_MLP_encoder = False, scale_factor_channels = 2  ):
        
        super().__init__()
        self.VAE_MLP_encoder = VAE_MLP_encoder
        if VAE_MLP_encoder:
            self.combine = nn.Linear(VAE_latent_size, 256 * 6*11)
        else:
            self.combine = nn.Conv2d(VAE_latent_size, 256, kernel_size=1)
        self.up1 = Up(256, 128, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(128, 64, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(64, 32, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(32, 16, scale_factor_channels = scale_factor_channels)
            # self last layer:	
    def forward(self, z):
        # Upsampling
        x = self.combine(z)
        if self.VAE_MLP_encoder:
            x = torch.unflatten(x, dim = 1, sizes = (256,6,11))
        x = self.up1(x)  # (batch, 128, 12, 22)
        x = self.up2(x, pad = (0,1,0,1))  # (batch, 64, 25, 45)
        x = self.up3(x)  # (batch, 32, 50, 90)
        x = self.up4(x)  # (batch, 16, 100, 180)      
        return x
	
class generation_NPS(nn.Module):
		
    def __init__( self, VAE_latent_size,   sigmoid = True, VAE_MLP_encoder = False , scale_factor_channels = 2  ):

        super().__init__()
        self.VAE_MLP_encoder = VAE_MLP_encoder
        if VAE_MLP_encoder:
            self.combine = nn.Linear(VAE_latent_size, 512 * 13*13)
        else:
            self.combine = nn.Conv2d(VAE_latent_size, 512, kernel_size=1)
        self.up1 = Up(512, 256, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(256, 128, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(128, 64, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(64, 32, scale_factor_channels = scale_factor_channels)
        self.up5 = Up(32, 16, scale_factor_channels = scale_factor_channels)

            # self last layer:	
    def forward(self, z):
        x = self.combine(z)
        if self.VAE_MLP_encoder:
            x = torch.unflatten(x, dim = 1, sizes = (512,13,13))
        # Upsampling
        x = self.up1(x, pad = (0,1,0,1))  # (batch, 256, 27, 27)
        x = self.up2(x)  # (batch, 128, 54, 54)
        x = self.up3(x)  # (batch, 64, 108, 108)
        x = self.up4(x)  # (batch, 32, 216, 216)
        x = self.up5(x)  # (batch, 16, 432, 432)     
        return x
	

class prior_recognition(nn.Module):
 
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, VAE_latent_size = None, VAE_MLP_encoder = False, scale_factor_channels = 2 ):
        
        super().__init__()
        self.n_channels_x = n_channels_x 
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16, multi_channel = True)
        # downsampling:
        self.d1 = Down(16, 32, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, multi_channel = True, scale_factor_channels = scale_factor_channels)
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
 
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, VAE_latent_size = None, VAE_MLP_encoder = False, scale_factor_channels = 2 ):
        
        super().__init__()
        self.n_channels_x = n_channels_x 
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16, multi_channel = True)
        # downsampling:
        self.d1 = Down(16, 32, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, multi_channel = True, scale_factor_channels = scale_factor_channels)
        self.d5 = Down(256, 512, multi_channel = True, scale_factor_channels = scale_factor_channels)
        # last conv of downsampling
        if VAE_latent_size is None:
              VAE_latent_size = 512
        if VAE_MLP_encoder:
            self.VAE_MLP_input_dim = 13 * 13 * 512
        else:
            self.VAE_MLP_input_dim = None
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels * 512
        self.last_conv_down = DoubleConvNext(512, 512,mid_channels=mid_channels, multi_channel=True, return_mask=False, VAE_latent_size = VAE_latent_size, VAE_MLP_input_dim = self.VAE_MLP_input_dim)

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
        x2, mask2  = self.d1(x1, mask1)  # (batch, 32, 216, 216)
        x3, mask3  = self.d2(x2, mask2)  # (batch, 64, 108, 108)
        x4, mask4  = self.d3(x3, mask3)  # (batch, 128, 54, 54)
        x5, mask5  = self.d4(x4, mask4)  # (batch, 256, 27, 27)
        x6, mask6  = self.d5(x5, mask5)  # (batch, 512, 13, 13)
        
        mu, log_var = self.last_conv_down(x6, mask6)  # (batch, 512, 6, 11)     
        return mu, log_var
    
class prediction(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True , scale_factor_channels = 2):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, return_skip = True, scale_factor_channels = scale_factor_channels)
        # self.d5 = Down(256, 512)
        # last conv of downsampling
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels *256
        self.last_conv_down = DoubleConvNext(256, 256,mid_channels=mid_channels, multi_channel=False, return_mask=False)
        # upsampling:
        self.up1 = Up(512, 128, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(256, 64, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(128, 32, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(64, 16, scale_factor_channels = scale_factor_channels)
			# self last layer:		

    def forward(self, x, mask):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)#.expand_as(x_in[0])  ## Uncomment if multi_channel is True
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

    # Downsampling
        x2, x2_bm, mask2, mask2_bm = self.d1(x1, mask1)  # (batch, 32, 50, 90) (batch, 32, 50, 90)
        x3, x3_bm, mask3, mask3_bm = self.d2(x2, mask2)  # (batch, 64, 25, 45) (batch, 64, 25, 45)
        x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 128, 12, 22) (batch, 128, 12, 22)
        x5, x5_bm, mask5, mask5_bm = self.d4(x4, mask4)  # (batch, 256, 6, 11) (batch, 256, 6, 11)
        
        x6 = self.last_conv_down(x5, mask5)  # (batch, 256, 6, 11)
        
        # Upsampling
        x = self.up1(x6, x5_bm, mask5_bm)  # (batch, 128, 12, 22)
        x = self.up2(x, x4_bm, mask4_bm)  # (batch, 64, 25, 45)
        x = self.up3(x, x3_bm, mask3_bm)  # (batch, 32, 50, 90)
        x = self.up4(x, x2_bm, mask2_bm)  # (batch, 16, 100, 180)
        
        return x
    

class prediction_NPS(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, scale_factor_channels = 2 ):
        
        super().__init__()
        self.n_channels_x = n_channels_x

    
        # input  (batch, n_channels_x, 100, 180)
        
        self.initial_conv = InitialConv(n_channels_x, 16)

        # downsampling:
        self.d1 = Down(16, 32, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d2 = Down(32, 64, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d3 = Down(64, 128, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d4 = Down(128, 256, return_skip = True, scale_factor_channels = scale_factor_channels)
        self.d5 = Down(256, 512, return_skip = True, scale_factor_channels = scale_factor_channels)
        if scale_factor_channels is None:
             mid_channels = None
        else:
             mid_channels = scale_factor_channels *512
        # last conv of downsampling
        self.last_conv_down = DoubleConvNext(512, 512,mid_channels=mid_channels, multi_channel=False, return_mask=False)

        # upsampling:

        self.up1 = Up(1024, 256, scale_factor_channels = scale_factor_channels)
        self.up2 = Up(512, 128, scale_factor_channels = scale_factor_channels)
        self.up3 = Up(256, 64, scale_factor_channels = scale_factor_channels)
        self.up4 = Up(128, 32, scale_factor_channels = scale_factor_channels)
        self.up5 = Up(64, 16, scale_factor_channels = scale_factor_channels)
			# self last layer:
				
    def forward(self, x, mask, ind = None):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)#.expand_as(x_in[0])   ## Uncomment if multi_channel is True
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 432)

    # Downsampling
        x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 32, 216, 216) (batch, 32, 216, 216)
        x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 64, 108, 108) (batch, 64, 108, 108)
        x4, x4_bm, mask4, mask4_bm  = self.d3(x3, mask3)  # (batch, 128, 54, 54) (batch, 128, 54, 54)
        x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 256, 27, 27) (batch, 256, 27, 27)
        x6, x6_bm, mask6, mask6_bm  = self.d5(x5, mask5)  # (batch, 512, 13, 13) (batch, 512, 13, 13)

        x7 = self.last_conv_down(x6, mask6)  # (batch, 512, 13, 13)

        # Upsampling
        x = self.up1(x7, x6_bm, mask6_bm)  # (batch, 256, 27, 27)
        x = self.up2(x, x5_bm, mask5_bm)  # (batch, 128, 54, 54)
        x = self.up3(x, x4_bm, mask4_bm)  # (batch, 64, 108, 108)
        x = self.up4(x, x3_bm, mask3_bm)  # (batch, 32, 216, 216)
        x = self.up5(x, x2_bm, mask2_bm)  # (batch, 16, 432, 432)
        return x
    



class DoubleConvNext(nn.Module):
    r"""Adopted from from https://github.com/m2lines/Samudra/blob/main/samudra/model.py"""
    def __init__(self, in_channels, out_channels, mid_channels=None, multi_channel=False, return_mask=False, VAE_latent_size = None, VAE_MLP_input_dim = None):
        super().__init__()
        self.VAE_latent_size = VAE_latent_size
        if self.VAE_latent_size is not None:
              self.return_mask = False
        if mid_channels is None:
                mid_channels = out_channels
        self.return_mask = return_mask
        self.multi_channel = multi_channel
        self.VAE_MLP_input_dim = VAE_MLP_input_dim
                # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
            self.lambda_skip = True
        else:
            self.lambda_skip = False
            self.skip_module = PartialConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=False)
                
        self.conv1 = PartialConv2d(in_channels, mid_channels, kernel_size=3, padding= 1, multi_channel=multi_channel, return_mask=True)
        # self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn1 = LayerNorm(mid_channels, data_format='channels_first' )
        self.act1 = nn.GELU()
        
        self.conv2 = PartialConv2d(mid_channels, mid_channels, kernel_size=3, padding= 1, multi_channel=multi_channel, return_mask=True)
        # self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn2 = LayerNorm(mid_channels, data_format='channels_first' )
        self.act2 = nn.GELU()

        self.mlp = PartialConv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=True)
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
            
            if self.multi_channel:
                assert mask is not None
            if  self.lambda_skip:
                skip = self.skip_module(x)     
            else:
                skip = self.skip_module(x, mask)
            x, mask = self.conv1(x, mask)
            x = self.bn1(x)
            x = self.act1(x)
            # if not self.multi_channel:    ### Uncomment if multi_channel is True
            #     mask = None
            x, mask = self.conv2(x, mask)
            x = self.bn2(x)
            x = self.act2(x)

            x, mask= self.mlp(x, mask)
            # x= self.mlp(x)
            x = x + skip

            if self.VAE_latent_size:
                x = self.bn_vae(x)
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


class Down(nn.Module):
        """Downscaling with double conv then maxpool"""

        def __init__(self, in_channels, out_channels, pooling_padding = 0, multi_channel = False, return_skip = False, scale_factor_channels = 2):
                super().__init__()
                self.return_skip = return_skip
                if scale_factor_channels is None:
                     mid_channels = None
                else:
                     mid_channels = scale_factor_channels * in_channels
                self.pool = PartialConv2d(out_channels, out_channels, kernel_size=2, stride = 2, padding=pooling_padding,  multi_channel=multi_channel, return_mask=True)
                self.doubleconv = DoubleConvNext(in_channels, out_channels,mid_channels= mid_channels, multi_channel=multi_channel, return_mask=True)	
        def forward(self, x, mask):
                x1, mask1 = self.doubleconv(x, mask)
                x2, mask2 = self.pool(x1, mask1)
                if self.return_skip:
                    return x2, x1, mask2, mask1
                else:
                    return x2, mask2

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, up_kernel = 3, scale_factor_channels = 2):
            super().__init__()
            if scale_factor_channels is None:
                mid_channels = None
            else:
                mid_channels = scale_factor_channels * in_channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = PartialConv2d(in_channels, in_channels, kernel_size=3, padding=  1, multi_channel = True, return_mask = True)
            self.conv = DoubleConvNext(in_channels, out_channels, mid_channels= mid_channels, multi_channel=True, return_mask=False)
    
    def forward(self, x, x2 = None, mask2 = None, pad = None):# input is CHW
        x = self.up(x) 
        mask = torch.ones_like(x[0,...]).to(x)
        
        if x2 is not None:
            diffY = x2.size()[2] - x.size()[2]
            diffX = x2.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                                            diffY // 2, diffY - diffY // 2])
            mask = F.pad(mask, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            mask2 = mask2.expand_as(x2[0]) 
        
            x = torch.cat([x2, x], dim=1)
            mask = torch.cat([mask2, mask], dim=0)         
        elif pad is not None:
            x = F.pad(x, pad)
            mask = F.pad(mask, pad)

        x, mask = self.conv_mid(x, mask)
        # if pad is not None:
        #     x = F.pad(x, pad)
        x = self.conv(x, mask)
        return x
		

class InitialConv(nn.Module):
    def __init__(self, in_channels, out_channels, multi_channel = False):
            super().__init__()
            self.firstconv = PartialConv2d(in_channels, out_channels ,kernel_size=3, padding= [1,0], multi_channel=multi_channel, return_mask=True)
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



