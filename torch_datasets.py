import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


class XArrayDataset(Dataset):

    def __init__(self, data: xr.DataArray, target: xr.DataArray, mask=None, zeros_mask = None, lead_time=None, time_features=None, ensemble_features = False, in_memory=True, to_device=None, aligned = False, month_min_max = None, model = 'Autoencoder'):
        assert data.time.equals(target.time)
        self.model = model
        self.mask = mask
        self.data = data
        self.target = target
        
        if model == 'PNet':
            self.data = xr.concat([self.data[:,:i,...].rename({'lead_time' : 'seq'}).expand_dims('lead_time',1) for i in range(1,len(data.lead_time)+1)], dim = 'lead_time').transpose('time','lead_time',...,'channels','seq','lat','lon').assign_coords(lead_time = data.lead_time)
            self.data = self.data.fillna(0.0)

        if lead_time is not None:
            self.data = self.data.sel(lead_time = slice(lead_time, lead_time))
            self.target = self.target.sel(lead_time = slice(lead_time, lead_time))
            self.mask = self.mask[:,lead_time - 1][:,None] if mask is not None else None

        self.data = self.data.stack(flattened=('time','lead_time')).transpose('flattened',...)
        self.target = self.target.stack(flattened=('time','lead_time')).transpose('flattened',...)

        if self.mask is not None:
            self.data = self.data[~self.mask.flatten()]
            if aligned:
                self.target = self.target[~self.mask.flatten()]

        if aligned:
            target_idx = np.arange(len(self.data.flattened))
        else:
            target_idx = (years_in_months + self.data.lead_time - 1).to_numpy()
        self.target = self.target[target_idx,...]
        


        if time_features is not None:
            self.use_time_features = True
            time_features = np.array([time_features]).flatten()
            feature_indices = {'year': 0, 'lead_time': 1, 'month_sin': 2, 'month_cos': 3, 'imonth_sin' : 4, 'imonth_cos' : 5}
            # y = self.data.year.to_numpy() / np.max(self.data.year.to_numpy())
            yr, mn = np.divmod((self.data.time.to_numpy().astype(int) - month_min_max[0].astype(int)*100),100)
            y = ((yr * 12 + mn )/month_min_max[1])
            
 
            lt = self.data.lead_time.to_numpy() / np.max(self.data.lead_time.to_numpy())
            # msin = np.sin(2 * np.pi * lt/12.0)
            # mcos = np.cos(2 * np.pi * lt/12.0)
            init_month = np.mod(self.data.time.to_numpy(),100)
            isin = np.sin(2 * np.pi * init_month/12.0)
            icos = np.cos(2 * np.pi * init_month/12.0)
            if model == 'PNet':
                isin = np.stack([isin for _ in range(12)], axis=1)
                icos = np.stack([icos for _ in range(12)], axis=1)
                y = np.stack([y for _ in range(12)], axis=1)
                lt = np.stack([lt for _ in range(12)], axis=1)
                target_month  = np.mod(self.data.time.to_numpy(),100)[...,None] + np.arange(1,13) - 1
            else:
                target_month  = np.mod(self.data.time.to_numpy(),100) + self.data.lead_time.to_numpy() - 1
            msin = np.sin(2 * np.pi * target_month/12.0)
            mcos = np.cos(2 * np.pi * target_month/12.0)
            self.time_features = np.stack([y, lt, msin, mcos, isin, icos], axis=1)
            self.time_features = self.time_features[:, [feature_indices[k] for k in time_features if k not in ['active_mask', 'full_ice_mask', 'land_mask']],...]
        else:
            self.use_time_features = False


        
        if all([self.use_time_features, model not in ['Autoencoder']]):
                
                if 'ensembles' in self.data.dims:
                    ref_shape = self.data.isel(ensembles = 0).isel(channels = 0).expand_dims('channels', axis=1).shape
                else:
                    ref_shape = self.data.isel(channels = 0).expand_dims('channels', axis=1).shape

                if model == 'PNet':
                    self.time_features = np.concatenate([np.broadcast_to(self.time_features[:, ind,:][:,None,:, None, None], ref_shape ) for ind in range(self.time_features.shape[1])] , axis = 1)
                else:
                    self.time_features = np.concatenate([np.broadcast_to(self.time_features[:, ind,None, None, None], ref_shape ) for ind in range(self.time_features.shape[1])] , axis = 1)


        if model == 'PNet':
            self.zeros_mask = xr.ones_like(self.data.isel(seq = 0)).copy()
        else:
            self.zeros_mask = xr.ones_like(self.data).copy()
        if zeros_mask is not None:
            target_month = np.mod(self.data.time.to_numpy(),100) + self.data.lead_time.to_numpy() - 1
            if 'active_grid' in zeros_mask.data_vars:
                self.zeros_mask[:] = np.broadcast_to(zeros_mask['active_grid'][np.mod(target_month.astype('int') ,12) - 1].to_numpy(), self.zeros_mask.shape)
            zeros_mask = zeros_mask.isel(ensembles = 0) if 'ensembles' in zeros_mask.dims else zeros_mask
            for item in ['active_mask', 'full_ice_mask']:
                if all([time_features is not None, item in time_features]):
                    self.time_features = np.concatenate([self.time_features, zeros_mask[item][np.mod(target_month.astype('int') ,12) - 1].to_numpy()], axis = 1)
        else:
            if model not in ['PNet','UNetLSTM', 'CNNLSTM', 'CNNLSTM_monthly', 'UNetLSTM_monthly']:
                self.zeros_mask = None

        
        self.lead_time_indices = self.data.lead_time

        if model in ['UNetLSTM','CNNLSTM','CNNLSTM_monthly', 'UNetLSTM_monthly']:
            if self.use_time_features:
                channels_axis = self.data.dims.index('channels')
                shape = xr.concat([xr.full_like(self.data.isel(channels = 0).expand_dims('channels', axis = channels_axis), np.nan) for _ in range(self.time_features.shape[1])], dim = 'channels')
                if 'ensembles' in self.data.dims:
                    shape[:] = np.broadcast_to(self.time_features[:,None,...], shape.shape)
                else:  
                    shape[:] = self.time_features
                self.time_features = shape.unstack('flattened').transpose('time',..., 'channels','lead_time','lat','lon').fillna(0.0)
                del shape
                if 'ensembles' in self.data.dims:
                    self.time_features = self.time_features.stack(flattened=('ensembles','time')).transpose('flattened',...)
                self.time_features = self.time_features.to_numpy()    
        
            self.data = self.data.unstack('flattened').transpose('time',...,'channels','lead_time','lat','lon').fillna(0.0)
            self.target = self.target.unstack('flattened').transpose('time',...,'channels','lead_time','lat','lon').fillna(0.0)
            self.zeros_mask = self.zeros_mask.unstack('flattened').transpose('time',...,'channels','lead_time','lat','lon').fillna(0.0)

            if 'ensembles' in self.data.dims:
                self.data['time'] = np.arange(0,len(self.data.time))  ## PG: create new coords for the ('year','lead_time') multi-index that shows indices
                self.data = self.data.stack(flattened=('ensembles','time')).transpose('flattened',...) ## PG: Unwrap the ensemble dim
                target_idx = self.data.time.values ## PG: Extract target indices based on the unwrapped ensemble dim
                self.target = self.target[target_idx,...] ## PG: Sample the target at the new unwrapped indices
                self.zeros_mask = self.zeros_mask.stack(flattened=('ensembles','time')).transpose('flattened',...)           
        
        elif 'ensembles' in self.data.dims: ## PG: if not ensemble mean:
            self.data = self.data.reset_index('lead_time','time').rename({'flattened':'flat'})  ## PG: Change the flattened multi-index coord with a simple coord
            self.data['flat'] = np.arange(0,len(self.data.flat))  ## PG: create new coords for the ('year','lead_time') multi-index that shows indices
            self.data = self.data.stack(flattened=('ensembles','flat')).transpose('flattened',...) ## PG: Unwrap the ensemble dim
            target_idx = self.data.flat.values ## PG: Extract target indices based on the unwrapped ensemble dim
            self.target = self.target[target_idx,...] ## PG: Sample the target at the new unwrapped indices
            self.lead_time_indices = self.lead_time_indices[target_idx,...]
            if self.use_time_features:
                self.time_features = self.time_features[target_idx,...] ## PG: sample time features with the same indices due to the unwrapping the ensemble dim
            if self.zeros_mask is not None:
                self.zeros_mask = self.zeros_mask.reset_index('lead_time','time').rename({'flattened':'flat'}) 
                self.zeros_mask['flat'] = np.arange(0,len(self.zeros_mask.flat)) 
                self.zeros_mask = self.zeros_mask.stack(flattened=('ensembles','flat')).transpose('flattened',...)

        if model  in ['CNNLSTM_monthly', 'UNetLSTM_monthly']:
            if self.mask is not None:
                year_mask = np.full(self.mask.sum(axis = 1).shape, False, dtype=bool)
                year_mask[self.mask.sum(axis = 1) >0 ] = True
            else:
                year_mask = np.full(self.data.shape[0], False, dtype=bool)
            self.data = self.data[~year_mask]
            self.time_features = self.time_features[~year_mask]
            self.target = self.target[~year_mask].sel(lead_time = lead_time)
            self.zeros_mask = self.zeros_mask[~year_mask].sel(lead_time = lead_time)

        if in_memory:
            
            self.data = torch.from_numpy(self.data.to_numpy()).float()
            self.target = torch.from_numpy(self.target.to_numpy()).float()
            self.lead_time_indices = torch.from_numpy(self.lead_time_indices.to_numpy().astype(int))
            if self.zeros_mask is not None:
                self.zeros_mask = torch.from_numpy(self.zeros_mask.to_numpy()).float()

            if self.use_time_features:
                self.time_features = torch.from_numpy(self.time_features).float()

            if to_device:
                self.data.to(to_device)
                self.target.to(to_device)
                if self.zeros_mask is not None:
                    self.zeros_mask.to(to_device)
                if self.use_time_features:
                    self.time_features = self.time_features.to(to_device)
            
    def __getitem__(self, index):
        x = self.data[index,...]
        y = self.target[index,...]
        s = self.lead_time_indices[index,...]
        
        if self.zeros_mask is not None:
            m = self.zeros_mask[index,...]

        if torch.is_tensor(x):
        
            if self.zeros_mask is not None:
                y_ = (y, m)
            else:
                y_ = y
        

            if self.use_time_features: 
                t = self.time_features[index,...]
                x_ = (x,t)
            else: 
                x_ = x
            if self.model == 'PNet':
                x_ = (*x_, s)
            return x_, y_
        else:

            x = torch.from_numpy(x.to_numpy()).float()
            y = torch.from_numpy(y.to_numpy()).float()

            if self.zeros_mask is not None:
                m = torch.from_numpy(m.to_numpy()).float()
                y_ = (y, m)
            else:
                y_ = y

            if self.use_time_features:
                t = self.time_features[index,...]
                t = torch.from_numpy(t).float()
                x_ = (x,t)
            else:
                x_ = x
            if self.model == 'PNet':
                x_ = (*x_, torch.from_numpy(s.to_numpy()).float())
            return x_, y_

    def __len__(self):
        return len(self.data)
    



class ConvLSTMDataset(Dataset):

    def __init__(self, data: xr.DataArray, target: xr.DataArray, mask=None, n_timesteps = 12, moving_window=1, zeros_mask = None, lead_time=1, time_features=None, ensemble_features = False, in_memory=True, to_device=None,  month_min_max = None):
        assert data.time.equals(target.time)
        assert lead_time is not None, 'Specify lead_time of prediction'
        self.mask = mask
        self.data = data
        self.target = target

        if moving_window is None:
            moving_window = n_timesteps + 1
  
        self.data = self.data.sel(lead_time = lead_time)
        self.target = self.target.sel(lead_time = lead_time)
        self.mask = self.mask[:,lead_time - 1] if mask is not None else None

        if self.mask is not None:
            self.data = self.data[~self.mask]
            self.target = self.target[~self.mask]

        if time_features is not None:
            for element in time_features:
                assert element not in ['lt'], f'{element} cannot be a time feature when lead_time is not None'

            self.use_time_features = True
            time_features = np.array([time_features]).flatten()
            feature_indices = {'year': 0, 'month_sin' : 1, 'month_cos' : 2, 'imonth_sin' : 3, 'imonth_cos' : 4}
            # y = self.data.year.to_numpy() / np.max(self.data.year.to_numpy())
            yr, mn = np.divmod((self.data.time.to_numpy().astype(int) - month_min_max[0].astype(int)*100),100)
            y = ((yr * 12 + mn )/month_min_max[1])
            target_month  = np.mod(self.data.time.to_numpy(),100) + lead_time - 1
            msin = np.sin(2 * np.pi * target_month/12.0)
            mcos = np.cos(2 * np.pi * target_month/12.0)
            init_month = np.mod(self.data.time.to_numpy(),100)
            isin = np.sin(2 * np.pi * init_month/12.0)
            icos = np.cos(2 * np.pi * init_month/12.0)
            self.time_features = np.stack([y,msin, mcos, isin, icos], axis=1)
            self.time_features = self.time_features[:, [feature_indices[k] for k in time_features if k not in ['active_mask', 'full_ice_mask', 'land_mask']],...]
        else:
            self.use_time_features = False

        if self.use_time_features:
                if 'ensembles' in self.data.dims:
                    ref_shape = self.data.isel(ensembles = 0).isel(channels = 0).expand_dims('channels', axis=1).shape
                else:
                    ref_shape = self.data.isel(channels = 0).expand_dims('channels', axis=1).shape
                self.time_features = np.concatenate([np.broadcast_to(self.time_features[:, ind,None, None, None], ref_shape ) for ind in range(self.time_features.shape[1])] , axis = 1)

        self.zeros_mask = xr.ones_like(self.data).copy()
        if zeros_mask is not None:
            target_month = np.mod(self.data.time.to_numpy(),100) + lead_time - 1
            if 'active_grid' in zeros_mask.data_vars:
                self.zeros_mask[:] = np.broadcast_to(zeros_mask['active_grid'][np.mod(target_month.astype('int') ,12) - 1].to_numpy(), self.zeros_mask.shape)
            zeros_mask = zeros_mask.isel(ensembles = 0) if 'ensembles' in zeros_mask.dims else zeros_mask
            for item in ['active_mask', 'full_ice_mask']:
                if all([time_features is not None, item in time_features]):
                    self.time_features = np.concatenate([self.time_features, zeros_mask[item][np.mod(target_month.astype('int') ,12) - 1].to_numpy()], axis = 1)
        else:
            self.zeros_mask = None

        x = self.data.isel(time=slice(None, None, -1))
        self.data = xr.concat([x[i:i + n_timesteps,...].isel(time=slice(None, None, -1)).rename({'time' : 'seq'}).expand_dims('time', axis = 0).assign_coords(seq = np.arange(1, n_timesteps +1)) for i in range(0, len(x) - n_timesteps +1, moving_window)], dim = 'time')
        self.data = self.data.assign_coords(time = [x[i:i + n_timesteps,...].isel(time=slice(None, None, -1)).time[-1].values for i in range(0, len(x) - n_timesteps +1, moving_window)]).transpose('time',...,'channels','seq','lat','lon')
        x = self.target.isel(time=slice(None, None, -1))
        self.target = xr.concat([x[i,...] for i in range(0, len(x) - n_timesteps +1, moving_window)], dim = 'time')
        # self.target = self.target.assign_coords(time = [x[i:i + n_timesteps,...].isel(time=slice(None, None, -1)).time[-1].values for i in range(0, len(x) - n_timesteps +1, moving_window)]).transpose('time',...,'channels','seq','lat','lon')
        if self.zeros_mask is not None:
            x = self.zeros_mask.isel(time=slice(None, None, -1))
            self.zeros_mask = xr.concat([x[i:i + n_timesteps,...].isel(time=slice(None, None, -1)).rename({'time' : 'seq'}).expand_dims('time', axis = 0).assign_coords(seq = np.arange(1, n_timesteps +1)) for i in range(0, len(x) - n_timesteps +1, moving_window)], dim = 'time')
            self.zeros_mask = self.zeros_mask.assign_coords(time = [x[i:i + n_timesteps,...].isel(time=slice(None, None, -1)).time[-1].values for i in range(0, len(x) - n_timesteps +1, moving_window)]).transpose('time',...,'channels','seq','lat','lon')
        if self.use_time_features:
            x = np.flip(self.time_features, axis = 0)
            self.time_features = np.concatenate([np.flip(x[i:i + n_timesteps,...], axis = 0)[None, ...] for i in range(0, len(x) - n_timesteps +1, moving_window)], axis = 0).reshape((self.data.shape[0], self.time_features.shape[1], self.data.shape[-3],self.data.shape[-2],self.data.shape[-1]))

        if 'ensembles' in self.data.dims: ## PG: if not ensemble mean:
            target_idx = np.arange(0,self.data.shape[0]) 
            target_idx = np.concatenate([target_idx for _ in range(len(self.data.ensembles))], axis = 0 )
            self.data = self.data.stack(flattened=('ensembles','time')).transpose('flattened',...) ## PG: Unwrap the ensemble dim
            ## PG: Extract target indices based on the unwrapped ensemble dim
            self.target = self.target[target_idx,...] ## PG: Sample the target at the new unwrapped indices
            if self.use_time_features:
                self.time_features = self.time_features[target_idx,...] ## PG: sample time features with the same indices due to the unwrapping the ensemble dim
            if self.zeros_mask is not None:
                self.zeros_mask = self.zeros_mask.stack(flattened=('ensembles','time')).transpose('flattened',...)


        if in_memory:
            self.data = torch.from_numpy(self.data.to_numpy()).float()
            self.target = torch.from_numpy(self.target.to_numpy()).float()
            if self.zeros_mask is not None:
                self.zeros_mask = torch.from_numpy(self.zeros_mask.to_numpy()).float()

            if self.use_time_features:
                self.time_features = torch.from_numpy(self.time_features).float()

            if to_device:
                self.data.to(to_device)
                self.target.to(to_device)
                if self.zeros_mask is not None:
                    self.zeros_mask.to(to_device)
                if self.use_time_features:
                    self.time_features = self.time_features.to(to_device)


    def __getitem__(self, index):

        x = self.data[index,...]
        y = self.target[index,...]
        
        if self.zeros_mask is not None:
            m = self.zeros_mask[index,...]

        if torch.is_tensor(x):
        
            if self.zeros_mask is not None:
                y_ = (y, m)
            else:
                y_ = y
            if self.use_time_features: 
                t = self.time_features[index,...]
                x_ = (x,t)
            else: 
                x_ = x
            return x_, y_
        else:

            x = torch.from_numpy(x.to_numpy()).float()
            y = torch.from_numpy(y.to_numpy()).float()

            if self.zeros_mask is not None:
                m = torch.from_numpy(m.to_numpy()).float()
                y_ = (y, m)
            else:
                y_ = y

            if self.use_time_features:
                t = self.time_features[index,...]
                t = torch.from_numpy(t).float()
                x_ = (x,t)
            else:
                x_ = x
            return x_, y_


    def __len__(self):
        return len(self.data)
