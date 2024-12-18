import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMSE:

    def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None):
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_area = loss_area
        self.device = device

        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):

        if self.loss_area is not None:


            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target

        m = torch.ones_like(y)
        m[(y < self.min_threshold) & (y_hat >= 0)] *= self.hyperparam
        m[(y > self.max_threshold) & (y_hat <= 0)] *= self.hyperparam

        if mask is not None:
            weight = self.weights * mask
        else:
            weight = self.weights

        if self.reduction == 'mean':
            loss = ((y_hat - y)**2 * m * weight).sum() / (torch.ones_like(y) * weight).sum()
        elif self.reduction == 'sum':
            loss = torch.mean((y_hat - y)**2 * m, dim=0)
            loss = torch.sum(loss * weight)
        elif self.reduction == 'none':
            loss = (y_hat - y)**2 * m * (weight / weight.sum())
        return loss

class WeightedMSELowRess:

    def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None):
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_area = loss_area
        self.device = device

        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)
        
        self.weights = F.avg_pool2d(self.weights.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)[0,0]

    def __call__(self, data, target, mask = None):

        if self.loss_area is not None:


            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target

        if mask is not None:
            weight= self.weights * F.avg_pool2d(mask, kernel_size=2, stride=2)
        else:
            weight= self.weights

        y_lowress =  F.avg_pool2d(y, kernel_size=2, stride=2)
        y_hat_lowress =  F.avg_pool2d(y_hat, kernel_size=2, stride=2)

        m = torch.ones_like(y_lowress)
        m[(y_lowress < self.min_threshold) & (y_hat_lowress >= 0)] *= self.hyperparam
        m[(y_lowress > self.max_threshold) & (y_hat_lowress <= 0)] *= self.hyperparam


        if self.reduction == 'mean':
            loss = ((y_hat_lowress - y_lowress)**2 * m *weight).sum() / (torch.ones_like(y_lowress) * weight).sum()
        elif self.reduction == 'sum':
            loss = torch.mean((y_hat_lowress - y_lowress)**2 * m, dim=0)
            loss = torch.sum(loss * weight)
        elif self.reduction == 'none':
            loss = (y_hat_lowress - y_lowress)**2 * m * (weight / weight.sum())
        return loss



# class WeightedMSESignLoss:  ## PG: penalizing negative anomalies
#     def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, min_val=0, max_val=None):
#         self.mse = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)
#         self.sign_loss = SignLoss( device=device, scale=scale, min_val=min_val, max_val=max_val, weights=weights, loss_area=loss_area, exclude_zeros=exclude_zeros)

#     def __call__(self, data, target, mask = None):
#         loss = 0
#         loss += self.mse(data, target, mask = mask)
#         loss += self.sign_loss(data, target, mask = mask)
#         return loss
    


class WeightedMSEGlobalLoss:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, map = True):
        self.mse = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)
        self.global_loss = GlobalLoss( device=device, scale=scale, weights=weights, loss_area=loss_area, map = map)

    def __call__(self, data, target, mask = None):
        loss = 0
        loss += self.mse(data, target, mask = mask)
        loss += self.global_loss(data, target, mask = mask)
        return loss

class WeightedMSEGlobalLossLowRess:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, map = True):
        self.mse = WeightedMSELowRess(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)
        self.global_loss = GlobalLoss( device=device, scale=scale, weights=weights, loss_area=loss_area, map = map)

    def __call__(self, data, target, mask = None):
        loss = 0
        loss += self.mse(data, target, mask = mask)
        loss += self.global_loss(data, target, mask = mask)
        return loss
    

class WeightedMSEOutlierLoss:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, min_val=0.15, max_val=1):
        self.mse = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)
        self.out_loss = OutlierLoss( device=device, scale=scale, min_val=min_val, max_val=max_val,  weights=weights, loss_area=loss_area, exclude_zeros=exclude_zeros)

    def __call__(self, data, target, mask = None):
        loss = 0
        loss += self.mse(data, target,mask = mask )
        loss += self.out_loss(data, target)
        return loss

    


# class SignLoss:  ## PG: Loss function based on negative anomalies

#     def __init__(self,  device, scale=1, min_val=0, max_val=None, weights=None, loss_area=None, exclude_zeros=True):
#         self.scale=scale
#         self.min_val = min_val
#         self.max_val=max_val
#         self.weights = weights
#         self.exclude_zeros = exclude_zeros
#         self.loss_area = loss_area
#         self.device = device
#         if loss_area is not None:
#             if weights.ndim>1:

#                 lat_min, lat_max, lon_min, lon_max = self.loss_area
#                 self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
#             else:
#                  indices =  self.loss_area
#                  self.weights = torch.from_numpy(weights[indices]).to(device)
#         else:
#             self.weights = torch.from_numpy(weights).to(device)

#     def __call__(self, data, target, mask = None):
#         if self.loss_area is not None:

#             if self.weights.ndim>1:

#                 lat_min, lat_max, lon_min, lon_max = self.loss_area
#                 y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
#                 y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

#             else:
                
#                 indices = self.loss_area
#                 y_hat = data[..., indices]
#                 y = target[..., indices]
#         else:
#             y_hat = data
#             y = target

#         if mask is not None:
#             weight = self.weights * mask
#         else:
#             weight = self.weights
        
#         l = torch.clamp((y * y_hat) * (-1) * self.scale, self.min_val, self.max_val) ## Check
#         if self.weights is None:
#             if self.exclude_zeros:
#                 loss = l.sum() / self.loss_mask.sum()
#             else:
#                 loss = torch.mean(l)
#         else:
#             loss = (l * weight).sum() / ( torch.ones_like(l) * weight).sum() ## Check
#         return loss
    


class OutlierLoss:  ## PG: Loss function based on negative anomalies

    def __init__(self,  device,  min_val=0, max_val=1, scale=1,  weights=None, loss_area=None, exclude_zeros=True):
        self.scale=scale
        self.weights = weights
        self.exclude_zeros = exclude_zeros
        self.min_val = min_val
        self.max_val=max_val
        self.loss_area = loss_area
        self.device = device
        if loss_area is not None:
            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):
        if self.loss_area is not None:

            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data 
            y = target 
   

        l1 = torch.clamp((y - 0.15) * (y_hat - 0.15 )* (-1) * self.scale, 0, None) ## Check
        l = torch.abs( y * y_hat) * self.scale
        l[l1 <= 0] = 0

        if self.weights is None:
            if self.exclude_zeros:
                loss = l.sum() / self.loss_mask.sum()
            else:
                loss = torch.mean(l)
        else:
            loss = (l * self.weights).sum() / ( torch.ones_like(l) * self.weights).sum() ## Check
        return loss


class GlobalLoss:  ## PG: Loss function based on negative anomalies

    def __init__(self,  device, weights, scale=1, loss_area=None,  map = True):
        self.scale=scale
        self.weights = weights
        self.loss_area = loss_area
        self.device = device
        self.map = map
        if loss_area is not None:
            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):
        if self.loss_area is not None:

            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target

        if self.map:
            l1 = (y_hat * self.weights).sum(dim=(-1,-2)) / ( self.weights).sum() 
            l2 = (y * self.weights).sum(dim=(-1,-2)) / ( self.weights).sum()
            if mask is not None:
                m = mask.sum(dim = (-1,-2))
                m[m != 0] = 1
                l1 = l1 * m
                l2 = l2 * m

        else:
            l1 = (y_hat * self.weights).sum(dim=(-1)) / (self.weights).sum() 
            l2 = (y * self.weights).sum(dim=(-1)) / (self.weights).sum()

        if mask is not None:
            loss = (((l1 - l2)**2)*self.scale).sum()/m.sum()
        else:
            loss = torch.mean(((l1 - l2)**2)*self.scale) ## Check
        return loss


class IceextentlLoss:  ## PG: Loss function based on negative anomalies

    def __init__(self,  device, weights, scale=1, loss_area=None,  map = True):
        self.scale=scale
        self.weights = weights
        self.loss_area = loss_area
        self.device = device
        self.map = map
        if loss_area is not None:
            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, data, target, mask = None):
        if self.loss_area is not None:

            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data[..., indices]
                y = target[..., indices]
        else:
            y_hat = data
            y = target

        w = torch.ones_like(y) * y
        w_hat = torch.ones_like(y_hat) * y_hat
        w[w<0.15] = 0
        w_hat[w_hat<0.15] = 0
        
        if self.map:
            l1 = (w_hat * self.weights).sum(dim=(-1,-2)) / ( self.weights).sum() 
            l2 = (w * self.weights).sum(dim=(-1,-2)) / ( self.weights).sum()
            if mask is not None:
                m = mask.sum(dim = (-1,-2))
                m[m != 0] = 1
                l1 = l1 * m
                l2 = l2 * m

        else:
            l1 = (w_hat * self.weights).sum(dim=(-1)) / (self.weights).sum() 
            l2 = (w * self.weights).sum(dim=(-1)) / (self.weights).sum()

        if mask is not None:
            loss = (((l1 - l2)**2)*self.scale).sum()/m.sum()
        else:
            loss = torch.mean(((l1 - l2)**2)*self.scale) ## Check
        return loss