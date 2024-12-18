import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SCNN(nn.Module):
    def __init__(
        self,
        n_channels_x,
        hidden_dimensions:list,
        kernel_size = 3,
        decoder_kernel_size = 1,
        sigmoid = True):

        super().__init__()
        self.n_channels_x = n_channels_x

        assert np.mod(kernel_size,2) == 1 
        assert np.mod(decoder_kernel_size,2) == 1
        self.decoder_kernel = decoder_kernel_size


        layers = []
        layers.append(SphericalConv(n_channels_x , hidden_dimensions[0] , kernel_size = kernel_size))
        for layer in range(len(hidden_dimensions) - 1):

                layers.append(SphericalConv(hidden_dimensions[layer], hidden_dimensions[layer + 1], kernel_size=kernel_size))

        self.encoder = nn.Sequential(*layers)

        if sigmoid:
            self.decoder = nn.Sequential(nn.Conv2d(hidden_dimensions[-1], 1, kernel_size=decoder_kernel_size, padding= [int(self.decoder_kernel /2),0]), nn.Sigmoid())
        else:
              self.decoder = nn.Sequential(nn.Conv2d(hidden_dimensions[-1], 1, kernel_size=decoder_kernel_size, padding= [int(self.decoder_kernel /2),0]))

    
    def forward(self, x, ind = None):

        if (type(x) == list) or (type(x) == tuple):
                
                x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x

        x_encoded = self.encoder(x_in)
        x_encoded = pad_ice(x_encoded, [0,int(self.decoder_kernel /2)])
        x_out = self.decoder(x_encoded)

        return x_out
        
class CNN(nn.Module):
    def __init__(
        self,
        n_channels_x,
        hidden_dimensions:list,
        kernel_size = 3, 
        decoder_kernel_size = 1,
        sigmoid = True):

        super().__init__()
        self.n_channels_x = n_channels_x

        assert np.mod(decoder_kernel_size,2) == 1
        self.decoder_kernel = decoder_kernel_size

        layers = []
        layers.append(Convblock(n_channels_x , hidden_dimensions[0] ,  kernel_size = kernel_size))
        for layer in range(len(hidden_dimensions) - 1):
                layers.append(Convblock(hidden_dimensions[layer], hidden_dimensions[layer + 1] , kernel_size = kernel_size))

        self.encoder = nn.Sequential(*layers)
        if sigmoid:
            self.decoder = nn.Sequential(nn.Conv2d(hidden_dimensions[-1], 1, kernel_size=decoder_kernel_size, padding= int(self.decoder_kernel /2)), nn.Sigmoid())
        else:
              self.decoder = nn.Sequential(nn.Conv2d(hidden_dimensions[-1], 1, kernel_size=decoder_kernel_size, padding= int(self.decoder_kernel /2)))
    
    def forward(self, x, ind = None):

        if (type(x) == list) or (type(x) == tuple):
                
                x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x

        x_encoded = self.encoder(x_in)
        x_out = self.decoder(x_encoded)

        return x_out
    

class RegCNN(nn.Module):
    def __init__(
        self,
        n_channels_x,
        added_features,
        hidden_dimensions:list,
        kernel_size = 3, 
        decoder_kernel_size = 1,
        DSC = False,
        sigmoid = True):

        super().__init__()
        self.n_channels_x = n_channels_x

        assert np.mod(decoder_kernel_size,2) == 1
        self.decoder_kernel = decoder_kernel_size

        layers = []
        layers.append(Convblock(n_channels_x + added_features , hidden_dimensions[0] ,  kernel_size = kernel_size, DSC = DSC))

        for layer in range(len(hidden_dimensions) - 2):
                layers.append(Convblock(hidden_dimensions[layer], hidden_dimensions[layer + 1] , kernel_size = kernel_size, DSC = DSC))

        layers.append(Convblock(hidden_dimensions[-2] , hidden_dimensions[-1] ,  kernel_size = kernel_size//2 + 1, DSC = DSC))    

        self.encoder = nn.Sequential(*layers)

        if DSC:
            self.out_conv = DepthwiseSeparableConv(
                                        hidden_dimensions[-1],
                                        n_channels_x,
                                        kernel_size=decoder_kernel_size,
                                        padding= int(decoder_kernel_size/2),
                                        kernels_per_layer=1)
        else:
            self.out_conv = nn.Conv2d(hidden_dimensions[-1], n_channels_x, kernel_size=decoder_kernel_size, padding= int(self.decoder_kernel /2))
        
        if sigmoid:
            self.decoder = nn.Sequential(self.out_conv, nn.Sigmoid())
        else:
              self.decoder = nn.Sequential(self.out_conv)
    
    def forward(self, x, ind = None):

        if (type(x) == list) or (type(x) == tuple):
                
                x_in = torch.cat([x[0], x[1]], dim=1)
        else:
            x_in = x

        x_encoded = self.encoder(x_in)
        x_out = self.decoder(x_encoded)

        return x_out

class Convblock(nn.Module):
        def __init__( self, in_channels, out_channels, DSC = False,  kernel_size = 3,  kernels_per_layer=1 ):
                
                super().__init__()
                assert np.mod(kernel_size,2) == 1
                if DSC:
                        self.conv = nn.Sequential( 
                                    DepthwiseSeparableConv(
                                    in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding= int(kernel_size/2),
                                    kernels_per_layer=kernels_per_layer),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                else:
                        self.conv = nn.Sequential( 
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding = int(kernel_size /2) ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                

        def forward(self, x):

            out = self.conv(x)
            return out
        



class SphericalConv(nn.Module):
        def __init__( self, in_channels, out_channels, DSC = False, kernel_size = 3,  kernels_per_layer=1 ):
                
                super().__init__()
                self.kernel_size = kernel_size
                if DSC:
                        self.conv = nn.Sequential( 
                                    DepthwiseSeparableConv(
                                    in_channels,
                                    out_channels,
                                    padding = [int(self.kernel_size/2),0],
                                    kernel_size=kernel_size,
                                    kernels_per_layer=kernels_per_layer),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                else:
                        self.conv = nn.Sequential( 
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding = [int(self.kernel_size/2),0]),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),)
                

        def forward(self, x):
            
            x = pad_ice(x, [0,int(self.kernel_size/2)])
            out = self.conv(x)

            return out
        



class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=3):
        super().__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer, output_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
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