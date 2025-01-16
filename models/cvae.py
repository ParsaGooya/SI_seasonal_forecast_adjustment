import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from models.partialconv2d import PartialConv2d

class cVAE(nn.Module):
	
    def __init__( self, VAE_latent_size, n_channels_x=1 ,  sigmoid = True, NPS_proj = False, device = 'cpu'):
        super().__init__()
        if NPS_proj:
            self.unet = prediction(n_channels_x, sigmoid)
            self.recognition = prior_recognition(n_channels_x + 1, sigmoid, VAE_latent_size = VAE_latent_size)
            self.prior = prior_recognition(n_channels_x + 1, sigmoid, VAE_latent_size = VAE_latent_size)
            self.generation = generation(sigmoid = sigmoid, VAE_latent_size = VAE_latent_size)
        else:
            self.unet = prediction_NPS(n_channels_x, sigmoid)
            self.recognition = prior_recognition_NPS(n_channels_x + 1, sigmoid, VAE_latent_size)
            self.prior = prior_recognition_NPS(n_channels_x + 1, sigmoid, VAE_latent_size)
            self.generation = generation_NPS(sigmoid = sigmoid, VAE_latent_size = VAE_latent_size)	
				
        self.last_conv = OutConv(16, 1, sigmoid = sigmoid, NPS_proj= NPS_proj)
        self.N = torch.distributions.Normal(0, 1)
            # Get sampling working on GPU
        if device.type == 'cuda':
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        
    def forward(self, obs, obs_mask, model, model_mask, sample_size = 1, seed = None, nstd = 1):
            
        basic_unet = self.unet(model, model_mask)
        deterministic_output = self.last_conv(basic_unet)

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
		
    def __init__( self, VAE_latent_size,  sigmoid = True ):
        
        super().__init__()
        self.combine = nn.Conv2d(VAE_latent_size, 256, kernel_size=1)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
            # self last layer:	
    def forward(self, z):
        # Upsampling
        x = self.combine(z)
        x = self.up1(x)  # (batch, 128, 12, 22)
        x = self.up2(x)  # (batch, 64, 25, 45)
        x = self.up3(x)  # (batch, 32, 50, 90)
        x = self.up4(x)  # (batch, 16, 100, 180)      
        return x
	
class generation_NPS(nn.Module):
		
    def __init__( self, VAE_latent_size,   sigmoid = True ):

        super().__init__()
        self.combine = nn.Conv2d(VAE_latent_size, 512, kernel_size=1)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.up5 = Up(32, 16)

            # self last layer:	
    def forward(self, z):
        # Upsampling
        x = self.combine(z)
        x = self.up1(x, pad = 1)  # (batch, 256, 27, 27)
        x = self.up2(x)  # (batch, 128, 54, 54)
        x = self.up3(x)  # (batch, 64, 108, 108)
        x = self.up4(x)  # (batch, 32, 216, 216)
        x = self.up5(x)  # (batch, 16, 432, 432)     
        return x
	

class prior_recognition(nn.Module):
 
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, VAE_latent_size = None ):
        
        super().__init__()
        self.n_channels_x = n_channels_x 
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32)
        self.d2 = Down(32, 64)
        self.d3 = Down(64, 128)
        self.d4 = Down(128, 256)
        # self.d5 = Down(256, 512)
        # last conv of downsampling
        if VAE_latent_size is None:
              VAE_latent_size = 256
        self.last_conv_down = DoubleConvNext(256, 256, multi_channel=True, return_mask=False, VAE_latent_size = VAE_latent_size)


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

        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

    # Downsampling
        x2, mask2 = self.d1(x1, mask1)  # (batch, 32, 50, 90)
        x3, mask3 = self.d2(x2, mask2)  # (batch, 64, 25, 45)
        x4, mask4 = self.d3(x3, mask3)  # (batch, 128, 12, 22)
        x5, mask5 = self.d4(x4, mask4)  # (batch, 256, 6, 11)
        
        mu, log_var = self.last_conv_down(x5, mask5)  # (batch, 256, 6, 11)     
        return mu, log_var
	
class prior_recognition_NPS(nn.Module):
 
    def __init__( self,  n_channels_x=1 ,  sigmoid = True, VAE_latent_size = None ):
        
        super().__init__()
        self.n_channels_x = n_channels_x 
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32)
        self.d2 = Down(32, 64)
        self.d3 = Down(64, 128)
        self.d4 = Down(128, 256)
        self.d5 = Down(256, 512)
        # last conv of downsampling
        if VAE_latent_size is None:
              VAE_latent_size = 512
        self.last_conv_down = DoubleConvNext(512, 512, multi_channel=True, return_mask=False, VAE_latent_size = VAE_latent_size)

    def forward(self, x, cond, mask):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if cond is not None:
            x_in = torch.cat([x_in, cond], dim=1)
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
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True ):
        
        super().__init__()
        self.n_channels_x = n_channels_x
        # input  (batch, n_channels_x, 100, 180)   
        self.initial_conv = InitialConv(n_channels_x, 16)
        # downsampling:
        self.d1 = Down(16, 32)
        self.d2 = Down(32, 64)
        self.d3 = Down(64, 128)
        self.d4 = Down(128, 256)
        # self.d5 = Down(256, 512)
        # last conv of downsampling
        self.last_conv_down = DoubleConvNext(256, 256, multi_channel=True, return_mask=False)
        # upsampling:
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
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
        x = self.up2(x)  # (batch, 64, 25, 45)
        x = self.up3(x)  # (batch, 32, 50, 90)
        x = self.up4(x)  # (batch, 16, 100, 180)
        
        return x
    

class prediction_NPS(nn.Module):
	
    
    def __init__( self,  n_channels_x=1 ,  sigmoid = True ):
        
        super().__init__()
        self.n_channels_x = n_channels_x

    
        # input  (batch, n_channels_x, 100, 180)
        
        self.initial_conv = InitialConv(n_channels_x, 16)

        # downsampling:
        self.d1 = Down(16, 32)
        self.d2 = Down(32, 64)
        self.d3 = Down(64, 128)
        self.d4 = Down(128, 256)
        self.d5 = Down(256, 512)

        # last conv of downsampling
        self.last_conv_down = DoubleConvNext(512, 512, multi_channel=True, return_mask=False)

        # upsampling:

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.up5 = Up(32, 16)
			# self last layer:
				
    def forward(self, x, mask, ind = None):
    # input  (batch, n_channels_x, 100, 180)
        if (type(x) == list) or (type(x) == tuple):    
            x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).expand_as(x_in[0])
        x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 432)
        print(mask1.shape)
    # Downsampling
        x2, mask2  = self.d1(x1, mask1)  # (batch, 32, 216, 216)
        print(mask2.shape)
        x3, mask3  = self.d2(x2, mask2)  # (batch, 64, 108, 108)
        x4, mask4  = self.d3(x3, mask3)  # (batch, 128, 54, 54)
        x5, mask5  = self.d4(x4, mask4)  # (batch, 256, 27, 27)
        x6, mask6  = self.d5(x5, mask5)  # (batch, 512, 13, 13)

        x7 = self.last_conv_down(x6, mask6)  # (batch, 1024, 13, 13)

        # Upsampling
        x = self.up1(x7, pad = 1)  # (batch, 512, 27, 27)
        x = self.up2(x)  # (batch, 256, 54, 54)
        x = self.up3(x)  # (batch, 128, 108, 108)
        x = self.up4(x)  # (batch, 64, 216, 216)
        x = self.up5(x)  # (batch, 32, 432, 432)
        return x
    



class DoubleConvNext(nn.Module):
	
    def __init__(self, in_channels, out_channels, mid_channels=None, multi_channel=False, return_mask=False, VAE_latent_size = None):
        super().__init__()
        self.VAE_latent_size = VAE_latent_size
        if self.VAE_latent_size is not None:
              self.return_mask = False
        if not mid_channels:
                mid_channels = out_channels
        self.return_mask = return_mask
        self.multi_channel = multi_channel

                # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
            self.lambda_skip = True
        else:
            self.lambda_skip = False
            self.skip_module = PartialConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=False)
                
        self.conv1 = PartialConv2d(in_channels, mid_channels, kernel_size=3, bias=False, padding= 1, multi_channel=multi_channel, return_mask=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = PartialConv2d(mid_channels, mid_channels, kernel_size=3, bias=False, padding= 1, multi_channel=True, return_mask=True)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.GELU()

        self.mlp = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        if VAE_latent_size is not None:
            self.bn_vae = nn.BatchNorm2d(out_channels)
            self.acr_vae = nn.ReLU(inplace = True)
            # self.mu = PartialConv2d(out_channels, out_channels, kernel_size=1, bias=False, multi_channel=True, return_mask=False)
            # self.log_var = PartialConv2d(out_channels, out_channels, kernel_size=1, bias=False, multi_channel=True, return_mask=False)
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
            if not self.multi_channel:
                mask = None
            x, mask = self.conv2(x, mask)
            x = self.bn2(x)
            x = self.act2(x)

            # x= self.mlp(x, mask)
            x= self.mlp(x)
            x = x + skip

            if self.VAE_latent_size:
                # x = self.bn_vae(x)
                x = self.acr_vae(x)
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

		def __init__(self, in_channels, out_channels, pooling_padding = 0):
				super().__init__()
				self.maxpool = nn.MaxPool2d(2,stride = 2, padding = pooling_padding)
				self.doubleconv = DoubleConvNext(in_channels, out_channels,mid_channels= out_channels, multi_channel=True, return_mask=True)	
		def forward(self, x, mask):
				x1, mask1 = self.doubleconv(x, mask)
				x1 = self.maxpool(x1)
				mask1 = self.maxpool(mask1)
				return x1, mask1


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, up_kernel = 3):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = PartialConv2d(in_channels, in_channels, kernel_size=3, padding=  1, bias = False)
            self.conv = DoubleConvNext(in_channels, out_channels, mid_channels=out_channels, multi_channel=False, return_mask=False)
    
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
				self.BN = nn.BatchNorm2d(out_channels)
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
				if sigmoid:
					self.conv1 = PartialConv2d(in_channels, in_channels, kernel_size=3, padding= padding)
					self.conv2 = nn.Sequential(
								nn.BatchNorm2d(in_channels),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid())
						
				else:
					self.conv1 = PartialConv2d(in_channels, in_channels, kernel_size=3, padding= padding)
					self.conv2 = nn.Sequential(
								nn.BatchNorm2d(in_channels),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels, out_channels, kernel_size=1))
			
		def forward(self, x):
				if not self.NPS_proj:
					x = pad_ice(x, [0,1])
				x1 = self.conv1(x)
				return self.conv2(x1)
		



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


