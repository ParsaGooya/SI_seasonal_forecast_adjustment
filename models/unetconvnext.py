import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from models.partialconv2d import PartialConv2d
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model



class UNet2(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True,skip_connection = True,  skip_conv = False, combined_prediction = False ):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
			self.skip_connection = skip_connection
			self.combined_prediction = combined_prediction
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = InitialConv(n_channels_x, 32)

			# downsampling:
			self.d1 = Down(32, 64,  skip_conv=skip_conv)
			self.d2 = Down(64, 128,  skip_conv=skip_conv)
			self.d3 = Down(128, 256,  skip_conv=skip_conv)
			self.d4 = Down(256, 512,  skip_conv=skip_conv)
			# self.d5 = Down(256, 512)

			# last conv of downsampling
			self.last_conv_down = DoubleConvNext(512,  multi_channel=False, return_mask=False)

			# upsampling:
			if bilinear:
				self.up1 = Up(512, 256, concat_channel= 256, bilinear=  True)
				self.up2 = Up(256, 128, concat_channel= 128, bilinear=True)
				self.up3 = Up(128, 64, concat_channel= 64, bilinear= True)
				self.up4 = Up(64, 32, concat_channel= 32, bilinear= True)
			else:	
				# self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
				self.up1 = Up(512, 256, concat_channel= 256,bilinear=  False, up_kernel = 2)
				self.up2 = Up(256, 128, concat_channel= 128, bilinear= False, up_kernel = [3,3])
				self.up3 = Up(128, 64, concat_channel= 64, bilinear= False, up_kernel = 2)
				self.up4 = Up(64, 32, concat_channel= 32, bilinear= False, up_kernel = 2)
			# self last layer:
			self.last_conv = OutConv(32, 1, sigmoid = sigmoid)
			if combined_prediction:
				self.last_conv2 = OutConv(32, 1, sigmoid = True, NPS_proj = True)


		def forward(self, x, mask, ind = None):
        # input  (batch, n_channels_x, 100, 180)
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
			
			x1 = self.last_conv(x)
			if self.combined_prediction:
				x2 = self.last_conv2(x)
				return x1, x2
			else:
				return x1
    

class UNet2_NPS(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True,skip_connection = True, skip_conv = False, combined_prediction = False ):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
			self.skip_connection = skip_connection
			self.combined_prediction = combined_prediction
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = InitialConv(n_channels_x, 32)

			# downsampling:
			self.d1 = Down(32, 64,  skip_conv=skip_conv)
			self.d2 = Down(64, 128, skip_conv=skip_conv)
			self.d3 = Down(128, 256,  skip_conv=skip_conv)
			self.d4 = Down(256, 512,  skip_conv=skip_conv)
			self.d5 = Down(512, 1024, skip_conv=skip_conv)

			# last conv of downsampling
			self.last_conv_down = DoubleConvNext(1024,  multi_channel=False, return_mask=False)

			# upsampling:
			if bilinear:
				self.up1 = Up(1024, 512, concat_channel= 512, bilinear=  True)
				self.up2 = Up(512, 256, concat_channel= 256, bilinear= True)
				self.up3 = Up(256, 128, concat_channel= 128, bilinear= True)
				self.up4 = Up(128, 64, concat_channel= 64, bilinear= True)
				self.up5 = Up(64, 32, concat_channel= 32, bilinear= True)
			else:	
				self.up1 = Up(1024, 512, concat_channel= 512,bilinear=  False, up_kernel = 3)
				self.up2 = Up(512, 256, concat_channel= 256, bilinear= False, up_kernel = 2)
				self.up3 = Up(256, 128, concat_channel= 128, bilinear= False, up_kernel = 2)
				self.up4 = Up(128, 64, concat_channel= 64, bilinear= False, up_kernel = 2)
				self.up5 = Up(64, 32, concat_channel= 32, bilinear= False, up_kernel = 2)


			# self last layer:
			self.last_conv = OutConv(32, 1, sigmoid = sigmoid, NPS_proj = True)
			if combined_prediction:
				self.last_conv2 = OutConv(32, 1, sigmoid = True, NPS_proj = True)
			

		def forward(self, x, mask, ind = None):
        # input  (batch, n_channels_x, 100, 180)
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
			
			x1 = self.last_conv(x)
			if self.combined_prediction:
				x2 = self.last_conv2(x)
				return x1, x2
			else:
				return x1
    
	

class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()

				self.firstconv = PartialConv2d(in_channels, out_channels ,kernel_size=3, padding= [1,0], multi_channel=False, return_mask=True)
				# self.BN = nn.BatchNorm2d(out_channels)
				self.BN = LayerNorm(out_channels, eps=1e-6, data_format='channels_first')
				self.activation = nn.ReLU(inplace=True)
				
		def forward(self, x, mask):
				x1 = pad_ice(x, [0,1])
				mask1 = pad_ice(mask, [0,1])
				x1, mask1 = self.firstconv(x1, mask1)
				x1 = self.BN(x1)
				x1 = self.activation(x1)
				return x1, mask1


		
class Up(nn.Module):
		"""Upscaling then double conv"""
		def __init__(self, in_channels, out_channels, concat_channel = 0, up_kernel = 3, bilinear=False):
				super().__init__()
				self.bilinear = bilinear
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv_mid = PartialConv2d(in_channels + concat_channel, out_channels, kernel_size=3, padding=  1, multi_channel = True, return_mask = True)
						self.conv = DoubleConvNext(out_channels,  multi_channel=True, return_mask=False)
						self.conv_mid.apply(weights_init)

				else:
						self.up = nn.ConvTranspose2d(in_channels+ concat_channel, out_channels , kernel_size=up_kernel, stride=2)
						self.conv = DoubleConvNext(out_channels,  multi_channel=True, return_mask=False)
						self.up.apply(weights_init)

				
					
		def forward(self, x1, x2 = None, mask2 = None, pad = None):
				x = self.up(x1)
				mask1 = torch.ones_like(x[0,...]).to(x)
				# if you have padding issues, see
				# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
				# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd	

				if x2 is not None:
					diffY = x2.size()[2] - x.size()[2]
					diffX = x2.size()[3] - x.size()[3]
					x = F.pad(x, [diffX // 2, diffX - diffX // 2,
													diffY // 2, diffY - diffY // 2])
					mask1 = F.pad(mask1, [diffX // 2, diffX - diffX // 2,
									diffY // 2, diffY - diffY // 2])
					mask2 = mask2.expand_as(x2[0])
				
					x = torch.cat([x2, x], dim=1)
					mask1 = torch.cat([mask2, mask1], dim=0)
				elif pad is not None:
					x = F.pad(x, pad)
					mask1 = F.pad(mask1, pad)
				# input is CHW
				if self.bilinear:
					x, mask1 = self.conv_mid(x, mask1)

				x = self.conv(x, mask1)

				return x

class Down(nn.Module):
		"""Downscaling with double conv then maxpool"""
	
		def __init__(self, in_channels, out_channels,  skip_conv = False, pooling_padding = 0):
				super().__init__()
				self.skip_conv = False
				self.doubleconv = DoubleConvNext(in_channels, multi_channel=False, return_mask=True)
				self.pool = PartialConv2d(in_channels, out_channels, kernel_size=2, stride = 2, padding=pooling_padding,  multi_channel=False, return_mask=True)
				# self.maxpool = nn.MaxPool2d(2,stride = 2, padding = pooling_padding)
				
				if skip_conv:
					self.skip_conv = True
					self.skipconv = DoubleConvNext(out_channels, multi_channel=False, return_mask=True)
				
				self.pool.apply(weights_init)
			
		def forward(self, x, mask):
				x1, mask1 = self.doubleconv(x, mask)
				x2,mask2 = self.pool(x1, mask1)
				# mask2 = self.maxpool(mask1)
				if self.skip_conv:
					x1, mask1 = self.skipconv(x1, mask1)

				return x2, x1, mask2, mask1


class OutConv(nn.Module):
		def __init__(self, in_channels, out_channels, sigmoid = True, NPS_proj = False):
				super().__init__()
				self.NPS_proj = NPS_proj
				if NPS_proj:
					padding = 1
				else:
					padding= [1,0]
				self.conv1 = PartialConv2d(in_channels, in_channels, kernel_size=3, padding= padding)
				if sigmoid:
					self.conv2 = nn.Sequential(
								LayerNorm(in_channels, eps=1e-6, data_format='channels_first'),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid())
					
				else:
					self.conv2 = nn.Sequential(
								LayerNorm(in_channels, eps=1e-6, data_format='channels_first'),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels, out_channels, kernel_size=1))
					
				self.conv2.apply(weights_init)

		def forward(self, x):
				# if not self.NPS_proj:
				# 	x = pad_ice(x, [0,1])
				# x = self.conv1(x)
				return self.conv2(x)
		
		
class DoubleConvNext(nn.Module):
		r""" Adopted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py"""	
		def __init__(self, in_channels,  multi_channel=False, return_mask=False):
				super().__init__()
				self.return_mask = return_mask
				self.multi_channel = multi_channel
				self.block1 = Block( in_channels,  multi_channel = multi_channel, return_mask = True )
				self.block2 = Block( in_channels,  multi_channel = multi_channel, return_mask = return_mask )
					
		def forward(self, x, mask = None):				
				x, mask = self.block1(x, mask)
				# if not self.multi_channel:
				# 	mask = None
				if self.return_mask:
					x, mask = self.block2(x, mask)
					return x, mask
				else:
					x = self.block2(x, mask)
					return x		

class Block(nn.Module):
	r""" Adopted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py"""

	def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, multi_channel = False, return_mask = False ):
		super().__init__()
		self.return_mask = return_mask
		self.dwconv = PartialConv2d(dim, dim, kernel_size=7, padding=3, groups=dim, multi_channel=multi_channel, return_mask=return_mask) # depthwise conv
		self.norm = LayerNorm(dim, eps=1e-6)
		self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
		self.act = nn.GELU()
		self.pwconv2 = nn.Linear(4 * dim, dim)
		self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
									requires_grad=True) if layer_scale_init_value > 0 else None
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.pwconv1.apply(weights_init)
		self.pwconv2.apply(weights_init)


	def forward(self, x, mask = None):
		input = x
		if self.return_mask:
			x, mask = self.dwconv(x, mask)
		else:
			x = self.dwconv(x, mask)
		x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
		x = self.norm(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		if self.gamma is not None:
			x = self.gamma * x
		x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

		x = input + self.drop_path(x)
		if self.return_mask:
			return x, mask
		else:
			return x


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
		

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride = 1, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
	
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


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		trunc_normal_(m.weight, std=.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

################################################################ Samudra based version ##########################################################################
#################################################################################################################################################################



# class UNet2(nn.Module):
	
    
# 		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True, skip_conv = False ):
			
# 			super().__init__()
# 			self.n_channels_x = n_channels_x
# 			self.bilinear = bilinear
		
# 			# input  (batch, n_channels_x, 100, 180)
			
# 			self.initial_conv = InitialConv(n_channels_x, 16)

# 			# downsampling:
# 			self.d1 = Down(16, 32, skip_conv)
# 			self.d2 = Down(32, 64, skip_conv)
# 			self.d3 = Down(64, 128, skip_conv)
# 			self.d4 = Down(128, 256, skip_conv)
# 			# self.d5 = Down(256, 512)

# 			# last conv of downsampling
# 			self.last_conv_down = DoubleConvNext(256, 512, multi_channel=True, return_mask=False)

# 			# upsampling:
# 			if bilinear:
# 				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 3)
# 				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2)
# 				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3)
# 				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3)
# 			else:	
# 				# self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
# 				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 2)
# 				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = [3,3])
# 				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 2)
# 				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 2)


# 			# self last layer:
# 			self.last_conv = OutConv(32, 1, sigmoid = sigmoid)
				

# 		def forward(self, x, mask, ind = None):
#         # input  (batch, n_channels_x, 100, 180)
# 			if (type(x) == list) or (type(x) == tuple):    
# 				x_in = torch.cat([x[0], x[1]], dim=1)
# 			else:
# 				x_in = x
# 			mask = mask.unsqueeze(0).expand_as(x_in[0])
# 			x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

# 		# Downsampling
# 			x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 32, 50, 90)
# 			x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 64, 25, 45)
# 			x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 128, 12, 22)
# 			x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 256, 6, 11)
			
# 			x6 = self.last_conv_down(x5, mask5)  # (batch, 512, 6, 11)
			
# 			# Upsampling
# 			x = self.up1(x6, x5_bm, mask5_bm)  # (batch, 256, 12, 22)
# 			x = self.up2(x, x4_bm, mask4_bm)  # (batch, 128, 25, 45)
# 			x = self.up3(x, x3_bm, mask3_bm)  # (batch, 64, 50, 90)
# 			x = self.up4(x, x2_bm, mask2_bm)  # (batch, 32, 100, 180)
			
# 			x = self.last_conv(x)
			
# 			return x
    

# class UNet2_NPS(nn.Module):
	
    
# 		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True, skip_conv = False ):
			
# 			super().__init__()
# 			self.n_channels_x = n_channels_x
# 			self.bilinear = bilinear
		
# 			# input  (batch, n_channels_x, 100, 180)
			
# 			self.initial_conv = InitialConv(n_channels_x, 16)

# 			# downsampling:
# 			self.d1 = Down(16, 32, skip_conv)
# 			self.d2 = Down(32, 64, skip_conv)
# 			self.d3 = Down(64, 128, skip_conv)
# 			self.d4 = Down(128, 256, skip_conv)
# 			self.d5 = Down(256, 512, skip_conv)

# 			# last conv of downsampling
# 			self.last_conv_down = DoubleConvNext(512, 1024, multi_channel=True, return_mask=False)

# 			# upsampling:
# 			if bilinear:
# 				self.up1 = Up(1024, 512,bilinear=  self.bilinear, up_kernel = 2)
# 				self.up2 = Up(512, 256, bilinear= self.bilinear, up_kernel = 3)
# 				self.up3 = Up(256, 128, bilinear= self.bilinear, up_kernel = 3)
# 				self.up4 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3)
# 				self.up5 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3)
# 			else:	
# 				self.up1 = Up(1024, 512,bilinear=  self.bilinear, up_kernel = 3)
# 				self.up2 = Up(512, 256, bilinear= self.bilinear, up_kernel = 2)
# 				self.up3 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2)
# 				self.up4 = Up(128, 64, bilinear= self.bilinear, up_kernel = 2)
# 				self.up5 = Up(64, 32, bilinear= self.bilinear, up_kernel = 2)


# 			# self last layer:
# 			self.last_conv = OutConv(32, 1, sigmoid = sigmoid, NPS_proj = True)
				

# 		def forward(self, x, mask, ind = None):
#         # input  (batch, n_channels_x, 100, 180)
# 			if (type(x) == list) or (type(x) == tuple):    
# 				x_in = torch.cat([x[0], x[1]], dim=1)
# 			else:
# 				x_in = x
# 			mask = mask.unsqueeze(0).expand_as(x_in[0])
# 			x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 432)

# 		# Downsampling
# 			x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 32, 216, 216)
# 			x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 64, 108, 108)
# 			x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 128, 54, 54)
# 			x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 256, 27, 27)
# 			x6, x6_bm, mask6, mask6_bm  = self.d5(x5, mask5)  # (batch, 512, 13, 13)

# 			x7 = self.last_conv_down(x6, mask6)  # (batch, 1024, 13, 13)

# 			# Upsampling
# 			x = self.up1(x7, x6_bm, mask6_bm)  # (batch, 512, 27, 27)
# 			x = self.up2(x, x5_bm, mask5_bm)  # (batch, 256, 54, 54)
# 			x = self.up3(x, x4_bm, mask4_bm)  # (batch, 128, 108, 108)
# 			x = self.up4(x, x3_bm, mask3_bm)  # (batch, 64, 216, 216)
# 			x = self.up5(x, x2_bm, mask2_bm)  # (batch, 32, 432, 432)
			
# 			x = self.last_conv(x)
			
# 			return x
		


# class DoubleConvNext(nn.Module):
	
# 		def __init__(self, in_channels, out_channels, mid_channels=None, multi_channel=False, return_mask=False):
# 				super().__init__()
# 				if not mid_channels:
# 						mid_channels = out_channels
# 				self.return_mask = return_mask
# 				self.multi_channel = multi_channel

# 				        # 1x1 conv to increase/decrease channel depth if necessary
# 				if in_channels == out_channels:
# 					self.skip_module = lambda x: x  # Identity-function required in forward pass
# 					self.lambda_skip = True
# 				else:
# 					self.lambda_skip = False
# 					self.skip_module = PartialConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias = False, multi_channel=multi_channel, return_mask=False)
						

# 				self.conv1 = PartialConv2d(in_channels, mid_channels, kernel_size=3, bias=False, padding= 1, multi_channel=multi_channel, return_mask=True)
# 				self.bn1 = nn.BatchNorm2d(mid_channels)
# 				self.act1 = nn.GELU()
				
# 				self.conv2 = PartialConv2d(mid_channels, mid_channels, kernel_size=3, bias=False, padding= 1, multi_channel=True, return_mask=True)
# 				self.bn2 = nn.BatchNorm2d(mid_channels)
# 				self.act2 = nn.GELU()

# 				self.mlp = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
			
# 		def forward(self, x, mask = None):
				
# 				if self.multi_channel:
# 					assert mask is not None
# 				skip = self.skip_module(x, mask)
				
# 				x, mask = self.conv1(x, mask)
# 				x = self.bn1(x)
# 				x = self.act1(x)
# 				if not self.multi_channel:
# 					mask = None
# 				x, mask = self.conv2(x, mask)
# 				x = self.bn2(x)
# 				x = self.act2(x)

# 				# x= self.mlp(x, mask)
# 				x= self.mlp(x)
# 				x = x + skip

# 				if self.return_mask:
# 					return x, mask
# 				else:
# 					return x


# class Down(nn.Module):
# 		"""Downscaling with double conv then maxpool"""
	
# 		def __init__(self, in_channels, out_channels, skip_conv = False, pooling_padding = 0):
# 				super().__init__()
# 				self.skip_conv = False
# 				self.maxpool = nn.MaxPool2d(2,stride = 2, padding = pooling_padding)
# 				self.doubleconv = DoubleConvNext(in_channels, out_channels,mid_channels= out_channels, multi_channel=True, return_mask=True)
# 				if skip_conv:
# 					self.skip_conv = True
# 					self.skipconv = DoubleConvNext(out_channels, out_channels, mid_channels= out_channels, multi_channel=True, return_mask=True)

			
# 		def forward(self, x, mask):
	
# 				x1, mask1 = self.doubleconv(x, mask)
# 				x2 = self.maxpool(x1)
# 				mask2 = self.maxpool(mask1)
# 				if self.skip_conv:
# 					x1, mask1 = self.skipconv(x1, mask1)
# 				return x2, x1, mask2, mask1


# class Up(nn.Module):
# 		"""Upscaling then double conv"""
# 		def __init__(self, in_channels, out_channels, up_kernel = 3, bilinear=False):
# 				super().__init__()
# 				self.bilinear = bilinear
# 				# if bilinear, use the normal convolutions to reduce the number of channels
# 				if bilinear:
# 						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# 						self.conv_mid = PartialConv2d(in_channels, in_channels // 2, kernel_size=3, padding=  1, bias = False)
# 						self.conv = DoubleConvNext(in_channels, out_channels, mid_channels=out_channels, multi_channel=True, return_mask=False)
# 				else:
# 						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=up_kernel, stride=2)
# 						self.conv = DoubleConvNext(in_channels, out_channels, mid_channels=out_channels, multi_channel=True, return_mask=False)
					
# 		def forward(self, x1, x2, mask2):
# 				x = self.up(x1)
				
# 				# input is CHW
# 				if self.bilinear:
# 					x = self.conv_mid(x)

# 				mask1 = torch.ones_like(x[0,...]).to(x)
# 				diffY = x2.size()[2] - x.size()[2]
# 				diffX = x2.size()[3] - x.size()[3]
# 				x = F.pad(x, [diffX // 2, diffX - diffX // 2,
# 												diffY // 2, diffY - diffY // 2])
# 				mask1 = F.pad(mask1, [diffX // 2, diffX - diffX // 2,
# 								diffY // 2, diffY - diffY // 2])
# 				# if you have padding issues, see
# 				# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
# 				# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd	

# 				x = torch.cat([x2, x], dim=1)
# 				x = self.conv(x, torch.cat([mask2, mask1], dim=0))

# 				return x