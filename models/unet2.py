import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from models.partialconv2d import PartialConv2d
class UNet2(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True, skip_conv = False ):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
		
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = InitialConv(n_channels_x, 16)

			# downsampling:
			self.d1 = Down(16, 32, skip_conv)
			self.d2 = Down(32, 64, skip_conv)
			self.d3 = Down(64, 128, skip_conv)
			self.d4 = Down(128, 256, skip_conv)
			# self.d5 = Down(256, 512)

			# last conv of downsampling
			self.last_conv_down = DoubleConv(256, 512, multi_channel=True, return_mask=False)

			# upsampling:
			if bilinear:
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 3)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3)
			else:	
				# self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 2)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = [3,3])
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 2)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 2)


			# self last layer:
			self.last_conv = OutConv(32, 1, sigmoid = sigmoid)
				

		def forward(self, x, mask, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x
			mask = mask.unsqueeze(0).expand_as(x_in[0])
			x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 100, 180)

		# Downsampling
			x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 32, 50, 90)
			x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 64, 25, 45)
			x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 128, 12, 22)
			x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 256, 6, 11)
			
			x6 = self.last_conv_down(x5, mask5)  # (batch, 512, 6, 11)
			
			# Upsampling
			x = self.up1(x6, x5_bm, mask5_bm)  # (batch, 256, 12, 22)
			x = self.up2(x, x4_bm, mask4_bm)  # (batch, 128, 25, 45)
			x = self.up3(x, x3_bm, mask3_bm)  # (batch, 64, 50, 90)
			x = self.up4(x, x2_bm, mask2_bm)  # (batch, 32, 100, 180)
			
			x = self.last_conv(x)
			
			return x
    

class UNet2_NPS(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True, skip_conv = False ):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
		
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = InitialConv(n_channels_x, 16)

			# downsampling:
			self.d1 = Down(16, 32, skip_conv)
			self.d2 = Down(32, 64, skip_conv)
			self.d3 = Down(64, 128, skip_conv)
			self.d4 = Down(128, 256, skip_conv)
			self.d5 = Down(256, 512, skip_conv)

			# last conv of downsampling
			self.last_conv_down = DoubleConv(512, 1024, multi_channel=True, return_mask=False)

			# upsampling:
			if bilinear:
				self.up1 = Up(1024, 512,bilinear=  self.bilinear, up_kernel = 2)
				self.up2 = Up(512, 256, bilinear= self.bilinear, up_kernel = 3)
				self.up3 = Up(256, 128, bilinear= self.bilinear, up_kernel = 3)
				self.up4 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3)
				self.up5 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3)
			else:	
				self.up1 = Up(1024, 512,bilinear=  self.bilinear, up_kernel = 3)
				self.up2 = Up(512, 256, bilinear= self.bilinear, up_kernel = 2)
				self.up3 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2)
				self.up4 = Up(128, 64, bilinear= self.bilinear, up_kernel = 2)
				self.up5 = Up(64, 32, bilinear= self.bilinear, up_kernel = 2)


			# self last layer:
			self.last_conv = OutConv(32, 1, sigmoid = sigmoid, NPS_proj = True)
				

		def forward(self, x, mask, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x
			mask = mask.unsqueeze(0).expand_as(x_in[0])
			x1, mask1 = self.initial_conv(x_in, mask)  # (batch, 16, 432, 432)

		# Downsampling
			x2, x2_bm, mask2, mask2_bm  = self.d1(x1, mask1)  # (batch, 32, 216, 216)
			x3, x3_bm, mask3, mask3_bm  = self.d2(x2, mask2)  # (batch, 64, 108, 108)
			x4, x4_bm, mask4, mask4_bm = self.d3(x3, mask3)  # (batch, 128, 54, 54)
			x5, x5_bm, mask5, mask5_bm  = self.d4(x4, mask4)  # (batch, 256, 27, 27)
			x6, x6_bm, mask6, mask6_bm  = self.d5(x5, mask5)  # (batch, 512, 13, 13)

			x7 = self.last_conv_down(x6, mask6)  # (batch, 1024, 13, 13)

			# Upsampling
			x = self.up1(x7, x6_bm, mask6_bm)  # (batch, 512, 27, 27)
			x = self.up2(x, x5_bm, mask5_bm)  # (batch, 256, 54, 54)
			x = self.up3(x, x4_bm, mask4_bm)  # (batch, 128, 108, 108)
			x = self.up4(x, x3_bm, mask3_bm)  # (batch, 64, 216, 216)
			x = self.up5(x, x2_bm, mask2_bm)  # (batch, 32, 432, 432)
			
			x = self.last_conv(x)
			
			return x
    


class DoubleConv(nn.Module):
		"""(convolution => [BN] => ReLU) * 2"""
	
		def __init__(self, in_channels, out_channels, mid_channels=None, multi_channel=False, return_mask=False):
				super().__init__()
				if not mid_channels:
						mid_channels = out_channels
				self.return_mask = return_mask
				self.multi_channel = multi_channel

				self.conv1 = PartialConv2d(in_channels, mid_channels, kernel_size=3, bias=False, padding= 1, multi_channel=multi_channel, return_mask=True)
				self.bn1 = nn.BatchNorm2d(mid_channels)
				self.act1 = nn.ReLU(inplace=True)
				
				self.conv2 = PartialConv2d(mid_channels, out_channels, kernel_size=3, bias=False, padding= 1, multi_channel=True, return_mask=return_mask)
				self.bn2 = nn.BatchNorm2d(out_channels)
				self.act2 = nn.ReLU(inplace=True)
			
		def forward(self, x, mask = None):
				if self.multi_channel:
					assert mask is not None
				x, mask = self.conv1(x, mask)
				x = self.bn1(x)
				x = self.act1(x)
				if not self.multi_channel:
					mask = None
				if self.return_mask:
					x, mask = self.conv2(x, mask)
				else:
					x= self.conv2(x, mask)
				x = self.bn2(x)
				x = self.act2(x)
				if self.return_mask:
					return x, mask
				else:
					return x

class Down(nn.Module):
		"""Downscaling with double conv then maxpool"""
	
		def __init__(self, in_channels, out_channels, skip_conv = False, pooling_padding = 0):
				super().__init__()
				self.skip_conv = False
				self.maxpool = nn.MaxPool2d(2,stride = 2, padding = pooling_padding)
				self.doubleconv = DoubleConv(in_channels, out_channels, multi_channel=True, return_mask=True)
				if skip_conv:
					self.skip_conv = True
					self.skipconv = DoubleConv(out_channels, out_channels, mid_channels=2*out_channels, multi_channel=True, return_mask=True)

			
		def forward(self, x, mask):
	
				x1, mask1 = self.doubleconv(x, mask)
				x2 = self.maxpool(x1)
				mask2 = self.maxpool(mask1)
				if self.skip_conv:
					x1, mask1 = self.skipconv(x1, mask1)
				return x2, x1, mask2, mask1


class Up(nn.Module):
		"""Upscaling then double conv"""
		def __init__(self, in_channels, out_channels, up_kernel = 3, bilinear=False):
				super().__init__()
				self.bilinear = bilinear
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv_mid = PartialConv2d(in_channels, in_channels // 2, kernel_size=up_kernel, padding=  1, bias = False)
						self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, multi_channel=True, return_mask=False)
				else:
						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=up_kernel, stride=2)
						self.conv = DoubleConv(in_channels, out_channels, multi_channel=True, return_mask=False)
					
		def forward(self, x1, x2, mask2):
				x = self.up(x1)
				# input is CHW
				if self.bilinear:
					x = self.conv_mid(x)

				# diffY = x2.size()[2] - x1.size()[2]
				# diffX = x2.size()[3] - x1.size()[3]
				# x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
				# 								diffY // 2, diffY - diffY // 2])
				# if you have padding issues, see
				# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
				# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
				
				mask1 = torch.ones_like(x[0,...]).to(x)

				x = torch.cat([x2, x], dim=1)
				x = self.conv(x, torch.cat([mask2, mask1], dim=0))

				return x
		

class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()

				self.firstconv = PartialConv2d(in_channels, out_channels, kernel_size=3, padding= [1,0], multi_channel=True, return_mask=True)
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
					self.conv1 = PartialConv2d(in_channels, in_channels//2, kernel_size=3, padding= padding)
					self.conv2 = nn.Sequential(
								nn.BatchNorm2d(in_channels//2),
								nn.ReLU(inplace=True),
								PartialConv2d(in_channels//2, out_channels, kernel_size=1), nn.Sigmoid())
						
				else:
					self.conv1 = PartialConv2d(in_channels, in_channels//2, kernel_size=3, padding= padding)
					self.conv2 = nn.Sequential(
								nn.BatchNorm2d(in_channels//2),
								nn.ReLU(inplace=True),
								PartialConv2d(in_channels//2, out_channels, kernel_size=1))
			
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


