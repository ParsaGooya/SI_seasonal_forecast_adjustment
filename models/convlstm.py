import torch
import torch.nn as nn
# from models.unet import UNet
# Original ConvLSTM cell as proposed by Shi et al.


class CNNLSTM_monthly(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 , add_features=0, hidden_dims = [64,128,128,64], seq_length = 12, kernel_size = 5, decoder_kernel_size = 1, sigmoid = True, device =  'cpu'):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.add_features = add_features
			self.seq_length = seq_length
			# input  (batch, n_channels_x, 100, 180)
			layers = []
			layers.append(Convcell(n_channels_x + add_features ,  hidden_dims[0], frame_size =(100,180),kernel_size = kernel_size, device= device))

			for layer in range(len(hidden_dims) - 2):
				layers.append(Convcell(hidden_dims[layer], hidden_dims[layer + 1], frame_size =(100,180),kernel_size = kernel_size, device= device))
                     
			layers.append(Convcell(hidden_dims[-2] , hidden_dims[-1] , frame_size =(100,180),kernel_size = kernel_size//2 + 1, device= device)) 
			self.encoder = nn.Sequential(*layers)

			if sigmoid:
				self.decoder = nn.Sequential(Convcell(hidden_dims[-1], n_channels_x, frame_size =(100,180), kernel_size=decoder_kernel_size, device= device), nn.Sigmoid())
			else:
				self.decoder = nn.Sequential(Convcell(hidden_dims[-1], n_channels_x, frame_size =(100,180), kernel_size=decoder_kernel_size, device= device))
			

		def forward(self, x, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x

			x_encoded = self.encoder(x_in)
			x_out = self.decoder(x_encoded)

			return x_out[:,:,-1,...]   
		

class CNNLSTM(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 , add_features=0, hidden_dims = [64,128,128,64], seq_length = 12, kernel_size = 5, decoder_kernel_size = 1, sigmoid = True, device =  'cpu'):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.add_features = add_features
			self.seq_length = seq_length
			# input  (batch, n_channels_x, 100, 180)
			layers = []
			layers.append(Convcell(n_channels_x + add_features ,  hidden_dims[0], frame_size =(50,40),kernel_size = kernel_size, device= device))

			for layer in range(len(hidden_dims) - 2):
				layers.append(Convcell(hidden_dims[layer], hidden_dims[layer + 1], frame_size =(50,40),kernel_size = kernel_size, device= device))
                     
			layers.append(Convcell(hidden_dims[-2] , hidden_dims[-1] , frame_size =(50,40),kernel_size = kernel_size//2 + 1, device= device)) 
			self.encoder = nn.Sequential(*layers)

			if sigmoid:
				self.decoder = nn.Sequential(Convcell(hidden_dims[-1], n_channels_x, frame_size =(50,40), kernel_size=decoder_kernel_size, device= device), nn.Sigmoid())
			else:
				self.decoder = nn.Sequential(Convcell(hidden_dims[-1], n_channels_x, frame_size =(50,40), kernel_size=decoder_kernel_size, device= device))
			

		def forward(self, x, ind = None):
        # input  (batch, n_channels_x, 100, 180)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x

			x_encoded = self.encoder(x_in)
			x_out = self.decoder(x_encoded)

			return x_out 
		
# class UNetLSTM(nn.Module):
	
    
# 		def __init__( self,  n_channels_x=1 , seq_length = 12,  bilinear=False, sigmoid = True, device =  'cpu'):
			
# 			super().__init__()
# 			self.n_channels_x = n_channels_x
# 			self.bilinear = bilinear
# 			self.seq_length = seq_length
# 			hidden_dims = [64,128,128,64]
# 			kernel_size = 5
# 			decoder_kernel_size = 1
# 			# input  (batch, n_channels_x, 100, 180)

# 			self.unet = UNet(n_channels_x= n_channels_x , bilinear = bilinear, sigmoid = sigmoid)
# 			layers = []
# 			layers.append(Convcell(n_channels_x ,  hidden_dims[0], frame_size =(100,180), kernel_size = kernel_size, device= device))

# 			for layer in range(len(hidden_dims) - 2):
# 				layers.append(Convcell(hidden_dims[layer], hidden_dims[layer + 1], frame_size =(100,180),kernel_size = kernel_size, device= device))
                     
# 			layers.append(Convcell(hidden_dims[-2] , hidden_dims[-1] , frame_size =(100,180),kernel_size = kernel_size//2 + 1, device= device)) 
# 			self.encoder = nn.Sequential(*layers)

# 			if sigmoid:
# 				self.decoder = nn.Sequential(Convcell(hidden_dims[-1], 1, frame_size =(100,180), kernel_size=decoder_kernel_size, device= device), nn.Sigmoid())
# 			else:
# 				self.decoder = nn.Sequential(Convcell(hidden_dims[-1], 1, frame_size =(100,180), kernel_size=decoder_kernel_size, device= device))

# 		def forward(self, x, ind = None):
#         # input  (batch, n_channels_x, 100, 180)
# 			if (type(x) == list) or (type(x) == tuple):    
# 				x_in = torch.cat([x[0], x[1]], dim=1)
# 			else:
# 				x_in = x

# 			x_in = torch.cat([self.unet(x_in[:,:,i,:,:]).unsqueeze(2) for i in range(self.seq_length)], axis = 2) 

# 			if (type(x) == list) or (type(x) == tuple):
# 				x_in = torch.cat([x_in, x[1]], dim=1)

# 			x_in = self.encoder(x_in)
# 			x_out = self.decoder(x_in)

# 			return x_out   

class UNetLSTM_monthly(nn.Module):
	   
		def __init__( self,  n_channels_x=1 , seq_length = 12,  bilinear=False, sigmoid = True, device =  'cpu'):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
			self.seq_length = seq_length
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = Convcell(n_channels_x, 16, frame_size =(100,180), device= device)
                     
			# downsampling:
			self.d1 = Down(16, 32,device = device, frame_size =(100,180) )# (batch, 32, 50, 90)
			self.d2 = Down(32, 64, device= device, frame_size =(50,90))# (batch, 64, 25, 45)
			self.d3 = Down(64, 128, device= device, frame_size =(25,45))# (batch, 128, 12, 22)
			self.d4 = Down(128, 256, device= device, frame_size =(12,22) )# (batch, 256, 6, 11)

			# last conv of downsampling
			self.last_conv_down = Convcell(256, 512, frame_size =(6,11),device = device)

			# upsampling:

			if bilinear:
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 3, frame_size =(12,22) ,device = device)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2, frame_size =(25,45),device = device)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3, frame_size =(50,90), device = device)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3, frame_size =(100,180), device = device)
			else:	
				# self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = (1,2,2), frame_size =(12,22),device = device)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = (1,3,3), frame_size =(25,45),device = device)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = (1,2,2), frame_size =(50,90),device = device)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = (1,2,2), frame_size =(100,180),device = device)

			# self last layer:
			self.last_conv= Convcell(32, 16, frame_size =(100,180),device = device)
			self.out_conv= Conv_out(16, 1, sigmoid = sigmoid)

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
			
			# x_out = torch.cat([conv_(x_in[:,:,i,:,:]).unsqueeze(-3) for i, conv_ in enumerate(self.last_conv_list)] , dim = -3)
			x = self.last_conv(x)
			x_out = self.out_conv(x[:,:,-1,...] )

			return x_out
		

class UNetLSTM(nn.Module):
	   
		def __init__( self,  n_channels_x=1 , seq_length = 12,  bilinear=False, sigmoid = True, device =  'cpu'):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
			self.seq_length = seq_length
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = Convcell(n_channels_x, 16, frame_size =(100,180), device= device)
                     
			# downsampling:
			self.d1 = Down(16, 32,device = device, frame_size =(100,180) )# (batch, 32, 50, 90)
			self.d2 = Down(32, 64, device= device, frame_size =(50,90))# (batch, 64, 25, 45)
			self.d3 = Down(64, 128, device= device, frame_size =(25,45))# (batch, 128, 12, 22)
			self.d4 = Down(128, 256, device= device, frame_size =(12,22) )# (batch, 256, 6, 11)

			# last conv of downsampling
			self.last_conv_down = Convcell(256, 512, frame_size =(6,11),device = device)

			# upsampling:

			if bilinear:
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 3, frame_size =(12,22) ,device = device)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2, frame_size =(25,45),device = device)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3, frame_size =(50,90), device = device)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3, frame_size =(100,180), device = device)
			else:	
				# self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = (1,2,2), frame_size =(12,22),device = device)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = (1,3,3), frame_size =(25,45),device = device)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = (1,2,2), frame_size =(50,90),device = device)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = (1,2,2), frame_size =(100,180),device = device)

			# self last layer:
			self.last_conv = Convcell(32, 16, frame_size =(100,180),device = device)
			self.last_conv_list = nn.ModuleList([Conv_out(16, 1, sigmoid=sigmoid) for _ in range(seq_length)])

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
			x_out = torch.cat([conv_(x[:,:,i,:,:]).unsqueeze(-3) for i, conv_ in enumerate(self.last_conv_list)] , dim = -3)

			return x_out   

class PNet(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 ,  bilinear=False, sigmoid = True, device =  'cpu'):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
		
			# input  (batch, n_channels_x, 100, 180)
			
			self.initial_conv = Convcell(n_channels_x, 16, frame_size =(100,180), device= device)
                     
			# downsampling:
			self.d1 = Down(16, 32,device = device, frame_size =(100,180) )# (batch, 32, 50, 90)
			self.d2 = Down(32, 64, device= device, frame_size =(50,90))# (batch, 64, 25, 45)
			self.d3 = Down(64, 128, device= device, frame_size =(25,45))# (batch, 128, 12, 22)
			self.d4 = Down(128, 256, device= device, frame_size =(12,22) )# (batch, 256, 6, 11)

			# last conv of downsampling
			self.last_conv_down = Convcell(256, 512, frame_size =(6,11),device = device)

			# upsampling:

			if bilinear:
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 3, frame_size =(12,22) ,device = device)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2, frame_size =(25,45),device = device)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = 3, frame_size =(50,90), device = device)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = 3, frame_size =(100,180), device = device)
			else:	
				# self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
				self.up1 = Up(512, 256,bilinear=  self.bilinear, up_kernel = (1,2,2), frame_size =(12,22),device = device)
				self.up2 = Up(256, 128, bilinear= self.bilinear, up_kernel = (1,3,3), frame_size =(25,45),device = device)
				self.up3 = Up(128, 64, bilinear= self.bilinear, up_kernel = (1,2,2), frame_size =(50,90),device = device)
				self.up4 = Up(64, 32, bilinear= self.bilinear, up_kernel = (1,2,2), frame_size =(100,180),device = device)

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
			if ind is None:
				ind = 0
			
			x = self.last_conv(torch.cat([x[i,:,ind[i]-1].unsqueeze(0) for i in range(x_in.shape[0])], dim = 0))
			
			return x 
		  
class Conv_out(nn.Module):
		def __init__(self, in_channels, out_channels, sigmoid = True):
			
				super().__init__()
				if sigmoid:
					self.conv = nn.Sequential(
								nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid())
						
				else:
					self.conv = nn.Sequential(
								nn.Conv2d(in_channels, out_channels, kernel_size=1))
				
				self.conv.apply(weights_init)
			
		def forward(self, x):
				# x = pad_ice(x, [0,1])
				return self.conv(x)
		
class OutConv(nn.Module):
		def __init__(self, in_channels, out_channels, sigmoid = True):
			
				super().__init__()
				if sigmoid:
					self.conv = nn.Sequential(
								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= [1,1]),
								nn.BatchNorm2d(in_channels//2),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels//2, out_channels, kernel_size=1), nn.Sigmoid())
						
				else:
					self.conv = nn.Sequential(
								nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding= [1,1]),
								nn.BatchNorm2d(in_channels//2),
								nn.ReLU(inplace=True),
								nn.Conv2d(in_channels//2, out_channels, kernel_size=1))
				
				self.conv.apply(weights_init)
			
		def forward(self, x):
				# x = pad_ice(x, [0,1])
				return self.conv(x)
		


class Up(nn.Module):
		"""Upscaling then double conv"""
		def __init__(self, in_channels, out_channels,frame_size, up_kernel = 3, bilinear=False, device = 'cpu'):
				super().__init__()
				self.bilinear = bilinear
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv_mid = Convcell(in_channels, in_channels // 2, kernel_size=up_kernel,frame_size = frame_size, device = device)
						self.conv = Convcell(in_channels, out_channels,frame_size = frame_size, device = device)

				else:
						self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=up_kernel, stride=(1,2,2))
						self.conv = Convcell(in_channels, out_channels,frame_size = frame_size, device = device)
				
				self.up.apply(weights_init)	
				if bilinear:
					self.conv_mid.apply(weights_init)

		def forward(self, x1, x2):
				x = self.up(x1)
				# input is CHW
				if self.bilinear:
					x = self.conv_mid(x)
				return self.conv(torch.cat([x2, x], dim=1))
		
        
class Down(nn.Module):
	#"""Downscaling with double conv then maxpool"""
    def __init__(self, in_channels,out_channels, frame_size, device = 'cpu'):
        super().__init__()
        self.maxpool = nn.MaxPool3d((1,2,2),stride = (1,2,2))
        self.conv = Convcell(in_channels, out_channels, frame_size, kernel_size = 3, device = device)

    def forward(self, x):
                    
                x1 = self.conv(x)
                x2 = self.maxpool(x1)
                return x2, x1
		
		
class Convcell(nn.Module):
		def __init__(self, in_channels, out_channels, frame_size , kernel_size = 3 , device = 'cpu' ):
				super().__init__()

				self.firstconv = nn.Sequential(ConvLSTM(
                                in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding= int(kernel_size /2), 
                                activation='relu', frame_size=frame_size, device = device), ### remember to add pad_ice
                                nn.BatchNorm3d(out_channels))

		def forward(self, x):

				x = self.firstconv(x)
				return x


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size, device):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size, device=device)
        self.device = device
        self.frame_size = frame_size
    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, _, _ = X.size()
        height, width = self.frame_size
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, device=self.device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=self.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=self.device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H
        
        return output


class ConvLSTMCell(nn.Module):

		def __init__(self, in_channels, out_channels, 
		kernel_size, padding, activation, frame_size,device, batch_normalizing = False ):

			super(ConvLSTMCell, self).__init__()  
			self.batch_normalizing = batch_normalizing

			if activation == "tanh":
				self.activation = torch.tanh 
			elif activation == "relu":
				self.activation = torch.relu
				
			if type(padding) in [list, tuple]:
				self.pad_v = padding[0]
				self.pad_h = padding[1]
			else:
				self.pad_v = self.pad_h = padding
			# Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
			
			self.conv = nn.Conv2d(
				in_channels=in_channels + out_channels, 
				out_channels=4 * out_channels, 
				kernel_size=kernel_size, 
				padding=[self.pad_v,self.pad_h])
			nn.init.xavier_uniform(self.conv.weight) 
			nn.init.zeros_(self.conv.bias)

			if self.batch_normalizing:
				self.norm = nn.BatchNorm2d(4 * out_channels)  
			
			# Initialize weights for Hadamard Products
			self.register_parameter('W_ci', param = nn.Parameter(torch.Tensor(out_channels, *frame_size)))
			self.register_parameter('W_co', param = nn.Parameter(torch.Tensor(out_channels, *frame_size)))
			self.register_parameter('W_cf', param = nn.Parameter(torch.Tensor(out_channels, *frame_size)))
			nn.init.xavier_uniform_(self.W_ci)
			nn.init.xavier_uniform_(self.W_co)
			nn.init.xavier_uniform_(self.W_cf)

		def forward(self, X, H_prev, C_prev):

			# Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
			input = torch.cat([X, H_prev], dim=1)
			# input = pad_ice(input, [0,self.pad_h])
			conv_output = self.conv(input)
			if self.batch_normalizing:
				conv_output = self.norm(conv_output)
			# Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
			i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

			input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
			forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

			# Current Cell output
			C = forget_gate*C_prev + input_gate * self.activation(C_conv)

			output_gate = torch.sigmoid(o_conv + self.W_co * C )

			# Current Hidden State
			H = output_gate * self.activation(C)
			return H, C
    

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
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
