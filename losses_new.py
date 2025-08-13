import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class WeightedMSE:

    def __init__(self, weights, device, weights_mask = None,  hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, multi_ress_loss_kernel_size = None):
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_area = loss_area
        self.device = device
        self.multi_ress_loss_kernel_size =  multi_ress_loss_kernel_size
        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            if weights_mask is not None:
                weights = weights * weights_mask
            self.weights = torch.from_numpy(weights).to(device)
        if self.multi_ress_loss_kernel_size is not None:
            self.mse_lowress = WeightedMSELowRess(weights=weights, device=device, weights_mask = weights_mask, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area, kernel = self.multi_ress_loss_kernel_size)

    def __call__(self, target, data_mu ,data_logvar = None,  mask = None, print_loss = False):

        assert data_logvar == None

        if self.loss_area is not None:


            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data_mu[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data_mu[..., indices]
                y = target[..., indices]
        else:
            y_hat = data_mu
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
            loss = (y_hat - y)**2 * m
            loss = torch.sum(loss * weight, dim = (-1,-2)).mean()
        elif self.reduction == 'none':
            loss = (y_hat - y)**2 * m * (weight / weight.sum())
        if self.multi_ress_loss_kernel_size is not None:

            loss = loss + self.mse_lowress( target, data_mu, data_logvar, mask)
            loss = loss / 2
        if print_loss:
            print(f'MSE : {loss}')
        return loss

class WeightedMSELowRess:

    def __init__(self, weights, device, weights_mask = None, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, kernel = 4):
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.loss_area = loss_area
        self.device = device
        assert np.mod(kernel,2) == 0, 'choose even kernel size'
        self.kernel = kernel

        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            if weights_mask is not None:
                weights_mask = torch.from_numpy(weights_mask).to(device)
                krn = torch.ones(1, 1, self.kernel, self.kernel).to(weights_mask)
                weights_mask = F.conv2d(weights_mask.unsqueeze(0).unsqueeze(0), krn, bias=None, stride=self.kernel//2 )[0,0]
                weights_mask = torch.clamp(weights_mask, 0, 1)
                del krn

            self.weights = torch.from_numpy(weights).to(device) 
        
        self.weights = F.avg_pool2d(self.weights.unsqueeze(0).unsqueeze(0), kernel_size=self.kernel, stride=self.kernel//2)[0,0]
        if weights_mask is not None:
            self.weights = self.weights* weights_mask

    def __call__(self, target, data_mu ,data_logvar = None, mask = None, print_loss = False):

        assert data_logvar == None

        if self.loss_area is not None:


            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data_mu[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data_mu[..., indices]
                y = target[..., indices]
        else:
            y_hat = data_mu
            y = target

        if mask is not None:
            weight= self.weights * F.avg_pool2d(mask, kernel_size=4, stride=2)
        else:
            weight= self.weights

        y_lowress =  F.avg_pool2d(torch.flatten(y, start_dim = 0, end_dim = 1), kernel_size=self.kernel, stride=self.kernel//2)
        y_hat_lowress =  F.avg_pool2d(torch.flatten(y_hat, start_dim = 0, end_dim = 1), kernel_size=self.kernel, stride=self.kernel//2)

        m = torch.ones_like(y_lowress)
        m[(y_lowress < self.min_threshold) & (y_hat_lowress >= 0)] *= self.hyperparam
        m[(y_lowress > self.max_threshold) & (y_hat_lowress <= 0)] *= self.hyperparam


        if self.reduction == 'mean':
            loss = ((y_hat_lowress - y_lowress)**2 * m *weight).sum() / (torch.ones_like(y_lowress) * weight).sum()
        elif self.reduction == 'sum':
            loss = (y_hat_lowress - y_lowress)**2 * m
            loss = torch.sum(loss * weight, dim = (-1,-2)).mean()
        elif self.reduction == 'none':
            loss = (y_hat_lowress - y_lowress)**2 * m * (weight / weight.sum())
        if print_loss:
            print(f'MSE : {loss}')
        return loss
    


class WeightedCRPS:

    def __init__(self, weights, device, weights_mask = None,  reduction='mean' , loss_area = None):
        self.reduction = reduction
        self.device = device
        if weights_mask is not None:
            weights = weights * weights_mask
        self.weights = torch.from_numpy(weights).to(device)

    def __call__(self, truth, pred ,data_logvar = None,  mask = None, print_loss = False):

        assert data_logvar == None

        if mask is not None:
            weight = self.weights * mask
        else:
            weight = self.weights

        if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
            raise ValueError(
                "Expected pred to have one extra sample dim on left. "
                "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
            )
        opts = dict(device=pred.device, dtype=pred.dtype)
        num_samples = pred.size(0)
        if num_samples == 1:
            return (pred[0] - truth).abs()

        pred = pred.sort(dim=0).values
        diff = pred[1:] - pred[:-1]
        weight_crps = torch.arange(1, num_samples, **opts) * torch.arange(
            num_samples - 1, 0, -1, **opts
        )
        weight_crps = weight_crps.reshape(weight_crps.shape + (1,) * (diff.dim() - 1))
        loss = (pred - truth).abs().mean(0) - (diff * weight_crps).sum(0) / num_samples**2

        if self.reduction == 'mean':
            loss = (loss * weight).sum() / (torch.ones_like(loss) * weight).sum()
        elif self.reduction == 'sum':
            loss = torch.sum(loss * weight, dim = (-1,-2)).mean()

        if print_loss:
            print(f'CRPS : {loss}')

        return loss
##adapted from https://github.com/climagination/ClimatExML/blob/stochastic/ClimatExML/losses.py - https://docs.pyro.ai/en/stable/_modules/pyro/ops/stats.html#crps_empirical
    


class Weightedloglikelihood:

    def __init__(self, weights, device, weights_mask = None, reduction='mean', loss_area=None, multi_ress_loss_kernel_size = None):
        self.reduction = reduction
        self.loss_area = loss_area
        self.device = device
        self.multi_ress_loss_kernel_size =  multi_ress_loss_kernel_size

        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            if weights_mask is not None:
                weights = weights * weights_mask
            self.weights = torch.from_numpy(weights).to(device)
        if self.multi_ress_loss_kernel_size is not None:
            self.mse_lowress = WeightedloglikelihoodLowRess(weights=weights, device=device, weights_mask = weights_mask, reduction=reduction, loss_area=loss_area, kernel = self.multi_ress_loss_kernel_size)

    def __call__(self, target, data_mu ,data_logvar, mask = None, print_loss = False):

        if self.loss_area is not None:
            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data_mu[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data_mu[..., indices]
                y = target[..., indices]
        else:
            y_hat = data_mu
            y = target

        m = torch.ones_like(y)
        if mask is not None:
            weight = self.weights * mask
        else:
            weight = self.weights
        
        loss = (0.5 * (y_hat - y)**2 / (torch.exp(data_logvar) + 1e-4 ) ) + 0.5 * data_logvar + 0.5 * np.log(2 * np.pi)

        if self.reduction == 'mean':
            loss = ( loss * weight).sum() / (torch.ones_like(y) * weight).sum()

        elif self.reduction == 'sum':
            loss = torch.sum(loss * weight, dim = (-1,-2)).mean()

        elif self.reduction == 'none':
            loss = loss * (weight / weight.sum())

        if self.multi_ress_loss_kernel_size is not None:
            loss = loss + self.mse_lowress(target, data_mu, data_logvar, mask)
            loss = loss / 2

        if print_loss:
            print(f'LLH : {loss}')

        return loss
    
class WeightedloglikelihoodLowRess:

    def __init__(self, weights, device, weights_mask = None, reduction='mean', loss_area=None, kernel = 4):
        self.reduction = reduction
        self.loss_area = loss_area
        self.device = device
        assert np.mod(kernel,2) == 0, 'choose even kernel size'
        self.kernel = kernel

        if self.loss_area is not None:

            if weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                self.weights = torch.from_numpy(weights[lat_min:lat_max+1, lon_min:lon_max+1]).to(device)
            else:
                 indices =  self.loss_area
                 self.weights = torch.from_numpy(weights[indices]).to(device)
        else:
            if weights_mask is not None:
                weights_mask = torch.from_numpy(weights_mask).to(device)
                krn = torch.ones(1, 1, self.kernel, self.kernel).to(weights_mask)
                weights_mask = F.conv2d(weights_mask.unsqueeze(0).unsqueeze(0), krn, bias=None, stride=self.kernel//2 )[0,0]
                weights_mask = torch.clamp(weights_mask, 0, 1)
                del krn

            self.weights = torch.from_numpy(weights).to(device) 
        
        self.weights = F.avg_pool2d(self.weights.unsqueeze(0).unsqueeze(0), kernel_size=self.kernel, stride=self.kernel//2)[0,0]
        if weights_mask is not None:
            self.weights = self.weights* weights_mask

    def __call__(self, target, data_mu ,data_logvar, mask = None, print_loss = False):

        if self.loss_area is not None:


            if self.weights.ndim>1:

                lat_min, lat_max, lon_min, lon_max = self.loss_area
                y_hat = data_mu[..., lat_min:lat_max+1, lon_min:lon_max+1]
                y = target[..., lat_min:lat_max+1, lon_min:lon_max+1]

            else:
                
                indices = self.loss_area
                y_hat = data_mu[..., indices]
                y = target[..., indices]
        else:
            y_hat = data_mu
            y = target

        if mask is not None:
            weight= self.weights * F.avg_pool2d(mask, kernel_size=4, stride=2)
        else:
            weight= self.weights

        y_lowress =  F.avg_pool2d(torch.flatten(y, start_dim = 0, end_dim = 1), kernel_size=self.kernel, stride=self.kernel//2)
        y_hat_lowress =  F.avg_pool2d(torch.flatten(y_hat, start_dim = 0, end_dim = 1), kernel_size=self.kernel, stride=self.kernel//2)
        data_var_lowress = F.avg_pool2d(torch.flatten((torch.exp(data_logvar) + 1e-4 ), start_dim = 0, end_dim = 1), kernel_size=self.kernel, stride=self.kernel//2)

        loss = (0.5 * (y_hat_lowress - y_lowress)**2 / data_var_lowress ) + 0.5 * torch.log(data_var_lowress) + 0.5 * np.log(2 * np.pi)


        if self.reduction == 'mean':
            loss = ( loss * weight).sum() / (torch.ones_like(y_lowress) * weight).sum()

        elif self.reduction == 'sum':
            loss = torch.sum(loss * weight, dim = (-1,-2)).mean()

        elif self.reduction == 'none':
            loss = loss * (weight / weight.sum())

        if print_loss:
            print(f'LLH : {loss}')

        return loss
    


class VAEloss:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, weights_mask = None, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True,  min_val=0, max_val=None, multi_ress_loss_kernel_size = None, learn_decoder_variance = False, decoder_inject_noise = False):
        self.reduction = reduction
        self.device = device
        self.multi_ress_loss_kernel_size = multi_ress_loss_kernel_size
        self.learn_decoder_variance = learn_decoder_variance
        self.decoder_inject_noise = decoder_inject_noise

        if learn_decoder_variance:
           self.mse = Weightedloglikelihood(weights=weights, device=device, weights_mask = weights_mask, reduction=reduction, loss_area=loss_area)
        elif decoder_inject_noise:
            self.mse = WeightedCRPS(weights=weights, device=device, weights_mask = weights_mask, reduction=reduction, loss_area=loss_area)
        else: 
            self.mse = WeightedMSE(weights=weights, device=device, weights_mask = weights_mask, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)

        if multi_ress_loss_kernel_size is not None:
            if learn_decoder_variance:
                self.mse_lr = WeightedloglikelihoodLowRess(weights=weights, device=device, weights_mask = weights_mask,reduction=reduction, loss_area=loss_area, kernel = multi_ress_loss_kernel_size)
            else:
                self.mse_lr = WeightedMSELowRess(weights=weights, device=device, weights_mask = weights_mask, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area, kernel = multi_ress_loss_kernel_size)

    def __call__(self,  target, data_mu, data_logvar = None, mu = None, log_var = None, cond_mu = None, cond_log_var = None, beta = 1, mask = None, return_ind_loss = False, print_loss = True, normalized_flow = None):
        loss = 0

        MSE = self.mse(target,data_mu, data_logvar , mask)

        if self.multi_ress_loss_kernel_size is not None :
            MSE += self.mse_lr(target,data_mu, data_logvar , mask)
            MSE = MSE/2

        loss += MSE#.mean()/(MSE.max() - MSE.min())

        if print_loss:
            if self.learn_decoder_variance:
                print(f'LLH : {loss}')
            elif self.decoder_inject_noise:
                print(f'CRPS : {loss}')
            else:
                print(f'MSE : {loss}')


        if any([mu is not None, log_var is not None]):
            assert all([mu is not None, log_var is not None])
            var = (torch.exp(log_var) + 1e-4)
            std = torch.sqrt(var)
            shape = mu.shape

            if all([cond_mu is None, cond_log_var is None]):
                KL = kl_divergence(
                                Normal(torch.flatten(mu, start_dim = 1, end_dim = -1), torch.flatten(std, start_dim = 1, end_dim = -1) ),
                                Normal(torch.zeros_like(torch.flatten(mu, start_dim = 1, end_dim = -1)), torch.ones_like(torch.flatten(mu, start_dim = 1, end_dim = -1))))
            else:
                cond_var = (torch.exp(cond_log_var) + 1e-4)
                cond_std = torch.sqrt(cond_var)
                KL = kl_divergence(
                                Normal(torch.flatten(mu, start_dim = 1, end_dim = -1),  torch.flatten(std, start_dim = 1, end_dim = -1)),
                                Normal(torch.flatten(cond_mu, start_dim = 1, end_dim = -1),torch.flatten(cond_std, start_dim = 1, end_dim = -1) ))
            if self.reduction == 'sum':
                KL = KL.sum(dim=-1).mean()
            if self.reduction == 'mean':
                KL = KL.mean()
            
            loss += KL * beta 
            if print_loss: 
                print(f'KLD : {KL}')
        else:
            return_ind_loss = False
        if return_ind_loss:
            return loss, MSE, KL
        else:
            return loss
        
class VAElossLowRess:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, weights_mask = None, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, min_val=0, max_val=None, kernel = 4, learn_decoder_variance = False):
        self.reduction = reduction
        self.device = device
        self.learn_decoder_variance = learn_decoder_variance
        if learn_decoder_variance:
            self.mse = WeightedloglikelihoodLowRess(weights=weights, device=device, weights_mask = weights_mask, reduction=reduction, loss_area=loss_area, kernel = kernel)
        else:
            self.mse = WeightedMSELowRess(weights=weights, device=device, weights_mask = weights_mask, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area, kernel = kernel)


    def __call__(self, target, data_mu ,data_logvar = None, mu = None, log_var = None, cond_mu = None, cond_log_var = None, beta = 1, mask = None, return_ind_loss = False, print_loss = True, normalized_flow = None):
        loss = 0
        MSE = self.mse(target, data_mu ,data_logvar , mask)
        loss += MSE
        if print_loss:
            if self.learn_decoder_variance:
                print(f'LLH : {loss}')
            else:
                print(f'MSE : {loss}')

        if any([mu is not None, log_var is not None]):
            assert all([mu is not None, log_var is not None])
            var = (torch.exp(log_var) + 1e-4)
            std = torch.sqrt(var)
            shape = mu.shape

            if all([cond_mu is None, cond_log_var is None]):
                KL = kl_divergence(
                                Normal(torch.flatten(mu, start_dim = 1, end_dim = -1), torch.flatten(std, start_dim = 1, end_dim = -1) ),
                                Normal(torch.zeros_like(torch.flatten(mu, start_dim = 1, end_dim = -1)), torch.ones_like(torch.flatten(mu, start_dim = 1, end_dim = -1))))
            else:
                cond_var = (torch.exp(cond_log_var) + 1e-4)
                cond_std = torch.sqrt(cond_var)
                KL = kl_divergence(
                                Normal(torch.flatten(mu, start_dim = 1, end_dim = -1),  torch.flatten(std, start_dim = 1, end_dim = -1)),
                                Normal(torch.flatten(cond_mu, start_dim = 1, end_dim = -1),torch.flatten(cond_std, start_dim = 1, end_dim = -1) ))
            if self.reduction == 'sum':
                KL = KL.sum(dim=-1).mean()
            if self.reduction == 'mean':
                KL = KL.mean()
            
            loss += KL * beta 
            if print_loss: 
                print(f'KLD : {KL}')
        else:
            return_ind_loss = False
        if return_ind_loss:
            return loss, MSE, KL
        else:
            return loss
        



class BCElossKLD:  ## PG: penalizing negative anomalies
    def __init__(self, device, reduction = 'mean'):
        self.reduction = reduction
        self.device = device
        self.bce = nn.BCELoss(reduction = self.reduction)

    def __call__(self, target, data_mu, data_logvar = None, mu = None, log_var = None, cond_mu = None, cond_log_var = None, beta = 1, mask = None, return_ind_loss = False, print_loss = True, normalized_flow = None):
        loss = 0
        BCE = self.bce(data_mu, target)
        loss += BCE
        if print_loss:
            print(f'BCE : {loss}')

        if any([mu is not None, log_var is not None]):
            assert all([mu is not None, log_var is not None])
            var = (torch.exp(log_var) + 1e-4)
            std = torch.sqrt(var)
            shape = mu.shape

            if all([cond_mu is None, cond_log_var is None]):
                KL = kl_divergence(
                                Normal(torch.flatten(mu, start_dim = 1, end_dim = -1), torch.flatten(std, start_dim = 1, end_dim = -1) ),
                                Normal(torch.zeros_like(torch.flatten(mu, start_dim = 1, end_dim = -1)), torch.ones_like(torch.flatten(mu, start_dim = 1, end_dim = -1))))
            else:
                cond_var = (torch.exp(cond_log_var) + 1e-4)
                cond_std = torch.sqrt(cond_var)
                KL = kl_divergence(
                                Normal(torch.flatten(mu, start_dim = 1, end_dim = -1),  torch.flatten(std, start_dim = 1, end_dim = -1)),
                                Normal(torch.flatten(cond_mu, start_dim = 1, end_dim = -1),torch.flatten(cond_std, start_dim = 1, end_dim = -1) ))
            if self.reduction == 'sum':
                KL = KL.sum(dim=-1).mean()
            if self.reduction == 'mean':
                KL = KL.mean()
            
            loss += KL * beta 
            if print_loss: 
                print(f'KLD : {KL}')
        else:
            return_ind_loss = False
        if return_ind_loss:
            return loss, BCE, KL
        else:
            return loss
        






class WeightedMSEGlobalLoss:  ## PG: penalizing negative anomalies
    def __init__(self, weights, device, hyperparam=1.0, min_threshold=0, max_threshold=0, reduction='mean', loss_area=None, exclude_zeros=True, scale=1, map = True):
        self.mse = WeightedMSE(weights=weights, device=device, hyperparam=hyperparam, reduction=reduction, loss_area=loss_area)
        self.global_loss = GlobalLoss( device=device, scale=scale, weights=weights, loss_area=loss_area, map = map)

    def __call__(self, data, target, mask = None):
        loss = 0
        loss += self.mse(data, target, mask = mask)
        loss += self.global_loss(data, target, mask = mask)
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