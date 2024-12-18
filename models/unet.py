import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class UNet(nn.Module):
	
    
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
			self.last_conv_down = DoubleConv(256, 512)

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
				

		def forward(self, x, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x
			x1 = self.initial_conv(x_in)  # (batch, 16, 100, 180)

		# Downsampling
			x2, x2_bm = self.d1(x1)  # (batch, 32, 50, 90)
			x3, x3_bm = self.d2(x2)  # (batch, 64, 25, 45)
			x4, x4_bm = self.d3(x3)  # (batch, 128, 12, 22)
			x5, x5_bm = self.d4(x4)  # (batch, 256, 6, 11)
		
			x6 = self.last_conv_down(x5)  # (batch, 512, 6, 11)
			
			# Upsampling
			x = self.up1(x6, x5_bm)  # (batch, 256, 12, 22)
			x = self.up2(x, x4_bm)  # (batch, 128, 25, 45)
			x = self.up3(x, x3_bm)  # (batch, 64, 50, 90)
			x = self.up4(x, x2_bm)  # (batch, 32, 100, 180)
			
			x = self.last_conv(x)
			
			return x
    

class UNet_NPS(nn.Module):
	
    
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
			# self.d5 = Down(256, 512)

			# last conv of downsampling
			self.last_conv_down = DoubleConv(512, 1024)

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
			self.last_conv = OutConv(32, 1, sigmoid = sigmoid)
				

		def forward(self, x, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x
			x1 = self.initial_conv(x_in)  # (batch, 16, 432, 432)

		# Downsampling
			x2, x2_bm = self.d1(x1)  # (batch, 32, 216, 216)
			x3, x3_bm = self.d2(x2)  # (batch, 64, 108, 108)
			x4, x4_bm = self.d3(x3)  # (batch, 128, 54, 54)
			x5, x5_bm = self.d4(x4)  # (batch, 256, 27, 27)
			x6, x6_bm = self.d5(x5)  # (batch, 512, 13, 13)

			x7 = self.last_conv_down(x6)  # (batch, 1024, 13, 13)
			
			# Upsampling
			x = self.up1(x7, x6_bm)  # (batch, 512, 27, 27)
			x = self.up2(x, x5_bm)  # (batch, 256, 54, 54)
			x = self.up3(x, x4_bm)  # (batch, 128, 108, 108)
			x = self.up4(x, x3_bm)  # (batch, 64, 216, 216)
			x = self.up5(x, x2_bm)  # (batch, 32, 432, 432)

			x = self.last_conv(x)
			
			return x    

class UNetLCL(nn.Module):
	
    
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
			self.last_conv_down = DoubleConv(256, 512)

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
			self.last_conv = OutConvLCL(32, 1, sigmoid = sigmoid)
				

		def forward(self, x, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x
			x1 = self.initial_conv(x_in)  # (batch, 16, 100, 180)

		# Downsampling
			x2, x2_bm = self.d1(x1)  # (batch, 32, 50, 90)
			x3, x3_bm = self.d2(x2)  # (batch, 64, 25, 45)
			x4, x4_bm = self.d3(x3)  # (batch, 128, 12, 22)
			x5, x5_bm = self.d4(x4)  # (batch, 256, 6, 11)
		
			x6 = self.last_conv_down(x5)  # (batch, 512, 6, 11)
			
			# Upsampling
			x = self.up1(x6, x5_bm)  # (batch, 256, 12, 22)
			x = self.up2(x, x4_bm)  # (batch, 128, 25, 45)
			x = self.up3(x, x3_bm)  # (batch, 64, 50, 90)
			x = self.up4(x, x2_bm)  # (batch, 32, 100, 180)
			
			x = self.last_conv(x)
			
			return x   

class DoubleConv(nn.Module):
		"""(convolution => [BN] => ReLU) * 2"""
	
		def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size = 3, padding = 1):
				super().__init__()
				if not mid_channels:
						mid_channels = out_channels
				self.start_conv = nn.Sequential(

						nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, bias=False, padding= padding),
						nn.BatchNorm2d(mid_channels),
						nn.ReLU(inplace=True))
				
				self.end_conv = nn.Sequential(
						nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, bias=False, padding= padding),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True))
			
		def forward(self, x):
				# x = pad_ice(x, [0,1])
				x = self.start_conv(x)
				# x = pad_ice(x, [0,1])
				return self.end_conv(x)
              
class Down(nn.Module):
		"""Downscaling with double conv then maxpool"""
	
		def __init__(self, in_channels, out_channels, skip_conv = False, pooling_padding = 0):
				self.skip_conv = False
				super().__init__()
				self.maxpool = nn.MaxPool2d(2,stride = 2, padding = pooling_padding)
				self.doubleconv = DoubleConv(in_channels, out_channels)
				if skip_conv:
					self.skip_conv = True
					self.skipconv = DoubleConv(out_channels, out_channels, mid_channels=2*out_channels)
			
		def forward(self, x):
	
				x1 = self.doubleconv(x)
				x2 = self.maxpool(x1)
				if self.skip_conv:
					x1 = self.skipconv(x1)
				return x2, x1


class Up(nn.Module):
		"""Upscaling then double conv"""
		def __init__(self, in_channels, out_channels, up_kernel = 3, bilinear=False):
				super().__init__()
				self.bilinear = bilinear
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv_mid = nn.Conv2d(in_channels, in_channels // 2, kernel_size=up_kernel, padding=  [1,1])
						self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
				else:
						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=up_kernel, stride=2)
						self.conv = DoubleConv(in_channels, out_channels)
					
		def forward(self, x1, x2):
				x = self.up(x1)
				# input is CHW
				if self.bilinear:
					# x1 = pad_ice(x1, [0,1])
					x = self.conv_mid(x)

				# diffY = x2.size()[2] - x1.size()[2]
				# diffX = x2.size()[3] - x1.size()[3]
				# x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
				# 								diffY // 2, diffY - diffY // 2])
				# if you have padding issues, see
				# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
				# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
				x = torch.cat([x2, x], dim=1)
				return self.conv(x)
		

class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()

				self.firstconv = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size=3, padding= [1,1]), 
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
		def forward(self, x):
				# x = pad_ice(x, [0,1])
				x1 = self.firstconv(x)
				return x1

class OutConv(nn.Module):
		def __init__(self, in_channels, out_channels, sigmoid = True):
				super().__init__()
				if sigmoid:
					self.conv = nn.Sequential(
								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= 1),
								nn.BatchNorm2d(in_channels//2),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels//2, out_channels, kernel_size=1), nn.Sigmoid())
						
				else:
					self.conv = nn.Sequential(
								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= 1),
								nn.BatchNorm2d(in_channels//2),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels//2, out_channels, kernel_size=1))
			
		def forward(self, x):
				return self.conv(x)
		

# class OutConv(nn.Module):
# 		def __init__(self, in_channels, out_channels, sigmoid = True):
			
# 				super().__init__()
# 				if sigmoid:
# 					self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid())
# 				else:
# 					self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1))
			
# 		def forward(self, x):
# 				return self.conv(x)
		
class OutConvLCL(nn.Module):
		def __init__(self, in_channels, out_channels, sigmoid = True):
			
				super().__init__()
				if sigmoid:
					self.conv = nn.Sequential(LocallyConnected2d(in_channels, out_channels, output_size = [100,180], kernel_size=1), nn.Sigmoid())
				else:
					self.conv = nn.Sequential(LocallyConnected2d(in_channels, out_channels, output_size = [100,180], kernel_size=1))
			
		def forward(self, x):
				return self.conv(x)


		
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





# class UNet2(nn.Module):
	
    
# 		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True ):
			
# 			super().__init__()
# 			self.n_channels_x = n_channels_x
# 			self.bilinear = bilinear
		
# 			# input  (batch, n_channels_x, 100, 180)
			
# 			self.initial_conv = InitialConv(n_channels_x, 16)

# 			# downsampling:
# 			self.d1 = Down(16, 32)
# 			self.d2 = Down(32, 64)
# 			self.d3 = Down(64, 128)
# 			self.d4 = Down(128, 256)
# 			# self.d5 = Down(256, 512)

# 			# last conv of downsampling
# 			self.last_conv_down = DoubleConv(256, 512)

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
# 			self.last_conv = OutConv2(32, 1, sigmoid = sigmoid)
				

# 		def forward(self, x):
#         # input  (batch, n_channels_x, 100, 180)
# 			if (type(x) == list) or (type(x) == tuple):    
# 				x_in = torch.cat([x[0], x[1]], dim=1)
# 			else:
# 				x_in = x
# 			x1 = self.initial_conv(x_in)  # (batch, 16, 100, 180)

# 		# Downsampling
# 			x2, x2_bm = self.d1(x1)  # (batch, 32, 50, 90)
# 			x3, x3_bm = self.d2(x2)  # (batch, 64, 25, 45)
# 			x4, x4_bm = self.d3(x3)  # (batch, 128, 12, 22)
# 			x5, x5_bm = self.d4(x4)  # (batch, 256, 6, 11)
		
# 			x6 = self.last_conv_down(x5)  # (batch, 512, 6, 11)
			
# 			# Upsampling
# 			x = self.up1(x6, x5_bm)  # (batch, 256, 12, 22)
# 			x = self.up2(x, x4_bm)  # (batch, 128, 25, 45)
# 			x = self.up3(x, x3_bm)  # (batch, 64, 50, 90)
# 			x = self.up4(x, x2_bm)  # (batch, 32, 100, 180)
			
# 			x = self.last_conv(x)
			
# 			return x   
	
		
# class OutConv2(nn.Module):
# 		def __init__(self, in_channels, out_channels, sigmoid = True):
			
# 				super().__init__()
# 				if sigmoid:
# 					self.conv = nn.Sequential(
# 								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= [1,0]),
# 								nn.BatchNorm2d(in_channels//2),
# 								nn.ReLU(inplace=True),
# 								nn.Conv2d(in_channels//2, out_channels, kernel_size=1), nn.Sigmoid())
						
# 				else:
# 					self.conv = nn.Sequential(
# 								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= [1,0]),
# 								nn.BatchNorm2d(in_channels//2),
# 								nn.ReLU(inplace=True),
# 								nn.Conv2d(in_channels//2, out_channels, kernel_size=1))
			
# 		def forward(self, x):
# 				x = pad_ice(x, [0,1])
# 				return self.conv(x)
		
# class OutConv2LCL(nn.Module):
# 		def __init__(self, in_channels, out_channels, sigmoid = True):
			
# 				super().__init__()
# 				if sigmoid:
# 					self.conv = nn.Sequential(
# 								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= [1,0]),
# 								nn.BatchNorm2d(in_channels//2),
# 								nn.ReLU(inplace=True),
# 								LocallyConnected2d(in_channels//2, out_channels, output_size = [100,180], kernel_size=1), nn.Sigmoid())
						
# 				else:
# 					self.conv = nn.Sequential(
# 								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= [1,0]),
# 								nn.BatchNorm2d(in_channels//2),
# 								nn.ReLU(inplace=True),
# 								LocallyConnected2d(in_channels//2, out_channels, output_size= [100,180], kernel_size=1))
			
# 		def forward(self, x):
# 				x = pad_ice(x, [0,1])
# 				return self.conv(x)

# class LinearUpsamplingZ(nn.Module):
#     # Linear upsampling of z to same number of channels as downsampled X
#     # so that can concatenate
#     def __init__(self, in_channels=4, out_channels=512):
#         super().__init__()
#         self.linearUp = nn.Sequential(
#             nn.Linear(in_channels, 128),
# 			nn.BatchNorm1d(128),
# 			nn.ReLU(inplace=True),
#             nn.Linear(128, 256),
# 			nn.BatchNorm1d(256),
# 			nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
# 			nn.BatchNorm1d(512),
# 			nn.ReLU(inplace=True),
#         )

#     def forward(self, z):
#         # transform to remove last dimensions
#         # z1 = z.view(z.shape[0], -1)  # (batch, 16)
#         z = self.linearUp(z)  # (batch, 1024)
#         # transform back to original shape
#         z = z.view(z.shape[0], 1, 1, -1).transpose(3, 1)  # (batch, 1024, 1, 1)
#         return z
		

# class UNet2(nn.Module):
	
    
# 		def __init__( self,  n_channels_x=1 ,add_features = 0,   bilinear=False, sigmoid = True ):
			
# 			super().__init__()
# 			self.n_channels_x = n_channels_x
# 			self.bilinear = bilinear
			
# 			# input  (batch, n_channels_x, 100, 180)
			
# 			self.initial_conv = InitialConv(n_channels_x, 16)

# 			if add_features > 0 :
# 				self.linear_z = LinearUpsamplingZ(add_features, 512)
# 				self.add_features =   True
# 				self.last_conv_down = DoubleConv(1024, 1024)
# 			else:
# 				self.last_conv_down = DoubleConv(512, 1024)
# 				self.add_features =   False
# 			# downsampling:
# 			self.d1 = Down(16, 32)
# 			self.d2 = Down(32, 64)
# 			self.d3 = Down(64, 128)
# 			self.d4 = Down(128, 256)
# 			self.d5 = Down(256, 512)

# 			# upsampling:

# 			self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = [3,2])
# 			self.up2 = Up(512, 256,bilinear=  self.bilinear, up_kernel = [2,3])
# 			self.up3 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2)
# 			self.up4 = Up(128, 64, bilinear= self.bilinear, up_kernel =  [3,2])
# 			self.up5 = Up(64, 32, bilinear= self.bilinear, up_kernel = 2)

# 			# self last layer:
# 			self.last_conv = OutConv(32, n_channels_x, sigmoid = sigmoid)
				

# 		def forward(self, x, ind = None):
#         # input  (batch, n_channels_x, 100, 180)
# 			if (type(x) == list) or (type(x) == tuple):    
# 				x_in = x[0]
# 				z_in = x[1]
# 			else:
# 				x_in = x

# 			x1 = self.initial_conv(x_in)  # (batch, 16, 50, 40) 

# 		# Downsampling
# 			x2, x2_bm = self.d1(x1)  # (batch, 32, 25, 20) 
# 			x3, x3_bm = self.d2(x2)  # (batch, 64, 12, 10) 
# 			x4, x4_bm = self.d3(x3)  # (batch, 128, 6, 5) 
# 			x5, x5_bm = self.d4(x4)  # (batch, 256, 3,  2) 
# 			x6, x6_bm = self.d5(x5)  # (batch, 512, 1, 1) 

# 			if self.add_features:
# 				z = self.linear_z(z_in) # (batch, 512, 1, 1) 
# 				x6 = torch.cat([x6, z], dim = 1)

# 			x7 = self.last_conv_down(x6)  # (batch, 1024, 6, 11) 1  1

# 			# Upsampling
# 			x = self.up1(x7, x6_bm)  # (batch, 512, 3, 2) 
# 			x = self.up2(x, x5_bm)  # (batch, 256, 6,  5) 
# 			x = self.up3(x, x4_bm)  # (batch, 128, 12, 10) 
# 			x = self.up4(x, x3_bm)  # (batch, 64, 25, 20) 
# 			x = self.up5(x, x2_bm) # (batch, 32, 50, 40) 

# 			x = self.last_conv(x)
			
# 			return x
    
