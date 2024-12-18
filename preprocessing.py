import numpy as np
from xarray import DataArray
import xarray as xr
import math

def select_test_years(dataset, n_test_years=1, test_years=None):
    if test_years:
        years = np.sort([test_years]).flatten()
        if years[-1] > dataset.year[-1].item():
            raise ValueError(f"Test year(s) {years} not valid - last possible test year: {dataset.year[-1].item()}")
    else:
        years = dataset.year[- n_test_years:].to_numpy()
    return years


def create_mask(dataset, exclude_idx=0):
    mask = np.full((dataset.shape[0], dataset.shape[1]), False, dtype=bool)
    x = np.arange(0, dataset.shape[0])   
    y = np.arange(1, dataset.shape[1] + 1)
    idx_array = x[..., None] + y
    mask[idx_array >= idx_array[-1, exclude_idx +1 ]] = True
    return mask


def get_coordinate_indices(data, area):
    lat_min, lat_max, lon_min, lon_max = area
    coord_indices = []
    coords = []
    for min_val, max_val, data in [(lat_min, lat_max, data.lat.data), (lon_min, lon_max, data.lon.data)]:
        x = data - min_val
        if np.min(x) > 0:
            min_idx = 0
        else:
            min_idx = np.abs(x[x <= 0]).argmin()
        coord_indices.append(min_idx)
        coords.append(data[min_idx])
        y = data - max_val
        if np.max(y) < 0:
            max_idx = len(data) - 1
        else:
            max_idx = np.argwhere(y == np.min(y[y >= 0])).flatten()[0]
        coord_indices.append(max_idx)
        coords.append(data[max_idx])
    return coord_indices, coords


def align_data_and_targets(data, targets, lead_months):



    if lead_months > data.shape[1]:
            raise ValueError(f'Maximum available lead months: {int(data.shape[1])}')
    
    data = data.sel(lead_time = np.arange(1, lead_months + 1 ))
    
    ls = [targets[ind:ind + lead_months , ...] for ind in range(len(targets.time))]  
    obs =  xr.concat([ds.rename({'time':'lead_time'}).assign_coords(lead_time = np.arange(1,len(ds.time) + 1))  for ds in ls], dim = 'time').assign_coords(time =  [ds.time[0].values for ds in ls])
    obs = obs.transpose('time','lead_time', ...)

    # last_target = targets.time[-1] - lead_months +1 if np.mod(targets.time[-1],100) >= lead_months else targets.time[-1] - 100 - lead_months + 13
    # if last_target > data.time[-1]:
    #     obs = obs.where(obs.time <= data.time[-1] , drop = True)
    # else:
    #     data = data.where(data.time <= last_target, drop = True)

    # return data.where(data <= 1 , 0), obs.sel(time = data.time).where(obs <= 1 , 0)

    if data.time[-1] >= obs.time[-1]:
        data = data.where( data.time <= obs.time[-1], drop = True)

    else:  
        obs = obs.where( obs.time <= data.time[-1], drop = True)
    
    obs = obs.where( (obs.time >= data.time.min()) & (obs.time <= data.time.max()) , drop = True)
    return data, obs



def reshape_obs_to_data(obs, data, return_xarray=False):
    lead_years = int(data.shape[1] / 12)
    base_index = obs.shape[0] - (lead_years-1)
    obs_reshaped = np.concatenate([obs[y:base_index + y,...] for y in range(lead_years)], axis=1)
    if return_xarray:
        coords = dict(lon = obs.lon,
                lat = obs.lat,
                year = obs.year[0:obs_reshaped.shape[0]],
                month = np.arange(1, obs_reshaped.shape[1] + 1))
        obs_reshaped = DataArray(obs_reshaped, coords=coords, dims=obs.dims)
    return obs_reshaped


def check_array_integrity(x, x_reshaped, size=50):
    rand_year = np.random.randint(0, x_reshaped.shape[0], size=size)
    rand_month = np.random.randint(0, x_reshaped.shape[1], size=size)
    for year_idx, month_idx in zip(rand_year, rand_month):
        year_overflow, month_overflow = np.divmod(month_idx, 12)
        x_year = year_idx + year_overflow
        x_month = month_overflow
        if not np.array_equal(x[x_year, x_month,...], x_reshaped[year_idx, month_idx]):
            print('ERROR')
    print('No errors found')


def batch_create_spatial_subsets(data, dim0_min, dim0_max, dim1_min, dim1_max):
    subsets = []
    if type(data) != list:
        data = [data]
    for array in data:
        subsets.append(array[..., dim0_min:dim0_max, dim1_min:dim1_max])
    return subsets
    

def calculate_climatology(ds, mask = None):
    if mask is not None:
        ds = ds.where(~mask)
    if 'ensembles' in ds.dims:
        mean_dims = ['time','ensembles'] 
    else:
        mean_dims = ['time']  
    
    return xr.concat([ds.isel(time = np.arange(0,len(ds.time),12) + init_month).mean(mean_dims) for init_month in range(12) ], dim = 'init_month').assign_coords(init_month = np.arange(1,13))

def calculate_anomalies(ds, monthly_climatology):
    return xr.concat([ds.where(np.mod(ds.time,100) == init_month, drop = True) - monthly_climatology.sel(init_month = init_month) for init_month in range(1,13)], dim = 'time').sortby('time')

def bias_adj(fct, fct_clim, obs_clim):
        fct_anom  = calculate_anomalies(fct, fct_clim)
        return xr.concat([fct_anom.where(np.mod(fct_anom.time,100) == init_month, drop = True) + obs_clim.sel(init_month = init_month) for init_month in range(1,13)], dim = 'time').sortby('time')
        


def detrend(data, detrend_base, trend_dim, deg=1, mask=None, with_intercept=True):
    detrend_base[trend_dim] = np.arange(len(detrend_base[trend_dim]))
    if mask is not None:
        trend_coefs = detrend_base.where(~mask).polyfit(dim=trend_dim, deg=deg, skipna=True)
    else:
        trend_coefs = detrend_base.polyfit(dim=trend_dim, deg=deg, skipna=True)
    slope = trend_coefs['polyfit_coefficients'][0].to_numpy()
    intercept = trend_coefs['polyfit_coefficients'][1].to_numpy()
    trend_axis = int(np.where(np.array(data.dims) == trend_dim)[0])
    timesteps = np.expand_dims(np.arange(data.shape[trend_axis]), axis=[i for i in range(0, data.ndim) if i != trend_axis])
    slope = np.expand_dims(slope, axis=trend_axis)
    intercept = np.expand_dims(intercept, axis=trend_axis)
    if with_intercept:
        trend = timesteps * slope + intercept
    else:
        trend = timesteps * slope
    data_detrend = data - trend
    return data_detrend, slope, intercept


def standardize(data, base, mask=None):
    if mask is not None:
        marray = np.ma.array(base.to_numpy(), mask=mask)
        mean = marray.mean()
        std = marray.std()
    else:
        mean = base.to_numpy().mean()
        std = base.to_numpy().std()
    data_standardized = (data - mean) / std
    coeffs = {'mean': mean,
              'std': std}
    return data_standardized, coeffs


def normalize(data, base, mask=None):
    if mask is not None:
        marray = np.ma.array(base.to_numpy(), mask=mask)
        max_val = marray.max()
        min_val = marray.min()
    else:
        max_val = base.to_numpy().max()
        min_val = base.to_numpy().min()
    data_normalized = (data - min_val) / (max_val - min_val)
    coeffs = {'min': min_val,
              'max': max_val}
    return data_normalized, coeffs


def linear_debiasing(data, baseline_data, baseline_targets, mask=None):
    # See Kharin et al. 2012: Statistical adjustment of decadal predictions in a changing climate. Geophysical Research Letters 39
    if mask is not None:
        baseline_data_mean = np.ma.array(baseline_data.to_numpy(), mask=mask).mean(axis=0).data
    else:
        baseline_data_mean = baseline_data.to_numpy().mean(axis=0)
    baseline_target_mean = baseline_targets.to_numpy().mean(axis=0)
    lead_time_years = int(np.ceil(baseline_data_mean.shape[0] / 12))
    baseline_target_mean = np.concatenate([baseline_target_mean] * lead_time_years, axis=0)
    bias = baseline_data_mean - baseline_target_mean
    data_debiased = data - bias
    return data_debiased


class AnomaliesScaler:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self, data, mask=None):
        if mask is not None:
            self.mean = np.ma.array(data.to_numpy(), mask=mask).mean(axis=self.axis).data
        else:
            self.mean = data.to_numpy().mean(axis=self.axis)
        return self

    def transform(self, data):
        data_anomalies = data - self.mean
        return data_anomalies
    
    def inverse_transform(self, data):
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            lead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(lead_years)], axis=0)
            data_raw = data + mean
        else:
            data_raw = data + self.mean
        return data_raw
    



 
class AnomaliesScaler_v2:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self, data, mask=None):
        
        if data.ndim>5: ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            self.large_ensemble = True
            axis = (self.axis, 2) ## PG: Tell the object to average over both years and ensembles for calculating anomalies.
        else:
            axis = self.axis

        if mask is not None:
            
            self.mean = np.ma.array(data.to_numpy(), mask=mask).mean(axis=axis).data[0:12,...] #PG

        else:
            self.mean = data.to_numpy().mean(axis=axis)[0:12,...] #PG
        
        nly = int(data.shape[1]/12) #PG
        self.mean = np.concatenate([self.mean for _ in range(nly)], axis = 0) #PG

        return self

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        if data.ndim>5:  ## PG: if ensemble exists in the dimentions.

            data_anomalies = data.transpose('ensembles', 'year', ...) ## PG: move ensemble dim to axis = 0 so that we can substract the mean that averaged over both years and ensembles
 

        data_anomalies = data_anomalies - self.mean

        return data_anomalies.transpose(*shape) ## PG: Move ensemble back to the original axis.
    
    def inverse_transform(self, data):
        
        shape = data.shape
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            lead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(lead_years)], axis=0)
            if data.ndim>5: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                
                data = data.reshape(data.shape[2],data.shape[0] ,*mean.shape)
            data_raw = data + mean
        else:
            if data.ndim>5: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data = data.reshape(data.shape[2],data.shape[0] ,*self.mean.shape)
            data_raw = data + self.mean
        return data_raw.reshape(shape) ## Move ensemble back to its original position
    
class AnomaliesScaler_v1:
    def __init__(self, axis=0) -> None:
        self.mean = None
        self.axis=axis
    
    def fit(self, data, mask=None):
        
        if data.ndim>5: ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            self.large_ensemble = True
            axis = (self.axis, 2) ## PG: Tell the object to average over both years and ensembles for calculating anomalies.
        else:
            axis = self.axis

        if mask is not None:
            
            self.mean = np.ma.array(data.to_numpy(), mask=mask).mean(axis=axis).data#PG

        else:
            self.mean = data.to_numpy().mean(axis=axis) #PG
        
        
        return self

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        if data.ndim>5:  ## PG: if ensemble exists in the dimentions.

            data_anomalies = data.transpose('ensembles', 'year', ...) ## PG: move ensemble dim to axis = 0 so that we can substract the mean that averaged over both years and ensembles
 

        data_anomalies = data_anomalies - self.mean
        
        return data_anomalies.transpose(*shape) ## PG: Move ensemble back to the original axis.
    
    def inverse_transform(self, data):
        
        shape = data.shape
        if data.shape[1] > 12 and self.mean.shape[0] <= 12:
            lead_years = int(data.shape[1] / 12)
            mean = np.concatenate([self.mean for _ in range(lead_years)], axis=0)
            if data.ndim>5: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                
                data = data.reshape(data.shape[2],data.shape[0] ,*mean.shape)
            data_raw = data + mean
        else:
            if data.ndim>5: ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data = data.reshape(data.shape[2],data.shape[0] ,*self.mean.shape)
            data_raw = data + self.mean
        return data_raw.reshape(shape) ## Move ensemble back to its original position
    


class AnomaliesScaler_v1_seasonal:
    def __init__(self, dim='time', global_weights = None) -> None:
        self.mean = None
        self.dim=dim
        self.global_weights = global_weights
    
    def fit(self, data, mask=None):
        self.data = data
        if 'ensembles' in data.dims: ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            self.large_ensemble = True
            axis = [self.dim, 'ensembles'] ## PG: Tell the object to average over both years and ensembles for calculating anomalies.
        else:
            axis = self.dim
     
        if mask is not None:
            self.data = self.data.where(~mask)

        self.mean = xr.concat([self.data.isel(time = np.arange(0,len(self.data.time),12) + init_month).mean(axis) for init_month in range(12) ], dim = 'init_month').assign_coords(init_month = np.arange(1,13)).transpose('init_month', ...)
        
        if self.global_weights is not None:
            self.mean = (self.mean * self.global_weights).sum(['lat','lon']) /  (self.global_weights).sum(['lat','lon'])
        del self.data
        return self

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        if 'ensembles' in data.dims: ## PG: if ensemble exists in the dimentions.
            data_anomalies = data_anomalies.transpose('ensembles', ...) ## PG: move ensemble dim to axis = 0 so that we can substract the mean that averaged over both years and ensembles
 
        climatology = np.concatenate([self.mean[:len(data_anomalies.time[ind:ind+12])].data for ind in range(0,len(data_anomalies.time),12)], axis = 0)
        if self.global_weights is not None:
            climatology = np.broadcast_to(climatology[...,None,None], data_anomalies.shape)

        data_anomalies = data_anomalies - climatology
        
        return data_anomalies.transpose(*shape) ## PG: Move ensemble back to the original axis.
    
    def inverse_transform(self, data, month, lead_time):
        
        shape = data.shape
        climatology = self.mean[month  - 1 : month - 1 + data.shape[0]].data
        if lead_time is not None:
            climatology = climatology[:,lead_time-1,...]
            
        if self.global_weights is not None:
            climatology = np.broadcast_to(climatology[...,None,None], data.shape)

        try:
                data_raw = data + climatology
        except : ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data_raw = data.reshape(data.shape[2],*climatology.shape)
                data_raw = data_raw + climatology
        return data_raw.reshape(shape) ## Move ensemble back to its original position


class AnomaliesScaler_v2_seasonal:
    def __init__(self, dim='time', global_weights = None) -> None:
        self.mean = None
        self.dim=dim
        self.global_weights = global_weights
    
    def fit(self, data, mask=None):
        self.data = data
        if 'ensembles' in data.dims: ## PG: if ensemble exists in the dimentions. Note that we always pass a map like data to this function. Even if it is flattened, we first write back to maps.
            self.large_ensemble = True
            axis = [self.dim, 'ensembles'] ## PG: Tell the object to average over both years and ensembles for calculating anomalies.
        else:
            axis = self.dim
     
        if mask is not None:
            self.data = self.data.where(~mask)

        self.mean = xr.concat([self.data.isel(time = np.arange(0,len(self.data.time),12) + init_month).mean(axis) for init_month in range(12) ], dim = 'init_month').assign_coords(init_month = np.arange(1,13)).transpose('init_month', ...).isel(lead_time = 0)
        self.mean = xr.concat([self.mean for _ in range(len(self.data.lead_time))], dim = 'lead_time').transpose('init_month','lead_time',...)
        if self.global_weights is not None:
            self.mean = (self.mean * self.global_weights).sum(['lat','lon']) /  (self.global_weights).sum(['lat','lon'])
        del self.data
        return self

    def transform(self, data):

        shape = data.dims
        data_anomalies = data.copy()
        if 'ensembles' in data.dims: ## PG: if ensemble exists in the dimentions.
            data_anomalies = data_anomalies.transpose('ensembles', ...) ## PG: move ensemble dim to axis = 0 so that we can substract the mean that averaged over both years and ensembles
 
        climatology = np.concatenate([self.mean[:len(data_anomalies.time[ind:ind+12])].data for ind in range(0,len(data_anomalies.time),12)], axis = 0)
        if self.global_weights is not None:
            climatology = np.broadcast_to(climatology[...,None,None], data_anomalies.shape)

        data_anomalies = data_anomalies - climatology
        
        return data_anomalies.transpose(*shape) ## PG: Move ensemble back to the original axis.
    
    def inverse_transform(self, data, month, lead_time):
        
        shape = data.shape
        climatology = self.mean[month  - 1 : month - 1 + data.shape[0]].data

        if data.shape[1] == 1:
            climatology = climatology[:,0,...]

        if self.global_weights is not None:
            climatology = np.broadcast_to(climatology[...,None,None], data.shape)

        try:
                data_raw = data + climatology
        except : ## PG: if ensemble exists in the dimentions we need to move it to axis = 0 to be able to add the self.mean.
                data_raw = data.reshape(data.shape[2],*climatology.shape)
                data_raw = data_raw + climatology
        return data_raw.reshape(shape) ## Move ensemble back to its original position
    


class Spatialnanremove: ## PG


    def __init__(self):
        pass

    def fit(self, data, target): ## PG: extract common grid points based on trainig and target data


        self.reference_shape = xr.full_like(target[0,0,0,...], fill_value = np.nan).drop(['month','year']) ## PG: Extract initial spatial shape of traiing data for later
        temp = target.stack(ref = ['lat','lon']).sel(ref =  data.stack(ref = ['lat','lon']).dropna(dim = 'ref').ref)  ## PG: flatten target in space and choose space points where data is not NaN.
        self.final_locations = temp.dropna('ref').ref ## PG: Extract locations common to target and training data by dropping the remaining NaN values
 
        return self

    def sample(self, data, mode = None, loss_area = None): ## PG: Pass a DataArray and sample at the extracted locations
        
           
        conditions = ['lat' in data.dims, 'lon' in data.dims]

        if all(conditions): ## PG: if a map get passeed
                
                sampled = data.stack(ref = ['lat','lon']).sel(ref = self.final_locations)

        else: ## PG: If a flattened dataset is passed (in space)
                sampled = data.sel(ref = self.final_locations)
    

        if mode == 'Eval':   ## PG: if we are sampling the test_set, remmeber the shepe of the test Dataset in a template
            self.shape = xr.full_like(sampled, fill_value = np.nan)


        return sampled

    def extract_indices(self, loss_area): ## PG: Extract indices of the flattened dimention over a specific region

            lat_min, lat_max, lon_min, lon_max = loss_area
            subregion_indices = self.final_locations.where((self.final_locations.lat < lat_max) &  (self.final_locations.lat > lat_min) )
            subregion_indice = subregion_indices.where((self.final_locations.lon < lon_max) &  (self.final_locations.lon > lon_min) )

            return ~ subregion_indices.isnull().values

    def to_map(self, data): ## PG: Write back the flattened data to maps
        if not isinstance(data, np.ndarray): ## PG: if you pass a numpy array (the output of the network)
            return data.unstack().combine_first(self.reference_shape) ## Unstack the flattened spatial dim and write back to the initial format as saved in self.reference_shape using NaN as fill value
        
        else:  ## PG: if you pass a numpy array (the output of the network), we use the test_set template that we saved to create a Datset.

            output = self.shape
            output[:] =  data[:]
            return output.unstack().combine_first(self.reference_shape)

       



class Detrender:
    def __init__(self, trend_dim='year', deg=1, remove_intercept=True, version = None) -> None:
        self.trend_dim = trend_dim
        self.deg = deg
        self.slope = None
        self.intercept = None
        self.trend_axis = None
        self.remove_intercept = remove_intercept

        self.version = version
        if version == None:
                self.lead_time_dim = False
        else:
                self.lead_time_dim = True

    def fit(self, data, mask=None):

        if self.version == 2:
            data = data[:,:12,...]
            mask = mask[:,:12,...]


        data[self.trend_dim] = np.arange(len(data[self.trend_dim]))
        if mask is not None:
            trend_coefs = data.where(~mask).polyfit(dim=self.trend_dim, deg=self.deg, skipna=True)
        else:
            trend_coefs = data.polyfit(dim=self.trend_dim, deg=self.deg, skipna=True)


        slope = trend_coefs['polyfit_coefficients'][0].to_numpy()
        intercept = trend_coefs['polyfit_coefficients'][1].to_numpy()

        self.trend_axis = int(np.where(np.array(data.dims) == self.trend_dim)[0])
        self.slope = np.expand_dims(slope, axis=self.trend_axis)
        self.intercept = np.expand_dims(intercept, axis=self.trend_axis)
        return self

    def transform(self, data, start_timestep=0, remove_intercept=None):
        if remove_intercept is None:
            remove_intercept = self.remove_intercept        
        timesteps = self._make_timesteps(data.shape[self.trend_axis], data.ndim, start_timestep=start_timestep)
        if self.lead_time_dim :
            if data.shape[1] > 12 and self.slope.shape[1] <= 12:
                lead_years = int(data.shape[1] / 12)
                trend = np.concatenate([self._compute_trend(timesteps + i, with_intercept=remove_intercept) for i in range(lead_years)], axis=1)
            else:
                trend = self._compute_trend(timesteps, with_intercept=remove_intercept)        
        else:
                trend = self._compute_trend(timesteps, with_intercept=remove_intercept)
        data_detrended = data - trend
        return data_detrended

    def inverse_transform(self, data, start_timestep=0, add_intercept=None):
        timesteps = self._make_timesteps(data.shape[self.trend_axis], data.ndim, start_timestep=start_timestep)
        if add_intercept is None:
            add_intercept = self.remove_intercept
        if self.lead_time_dim:
            if data.shape[1] > 12 and self.slope.shape[1] <= 12:
                lead_years = int(data.shape[1] / 12)
                trend = np.concatenate([self._compute_trend(timesteps + i, with_intercept=add_intercept) for i in range(lead_years)], axis=1)
            else:
                trend = self._compute_trend(timesteps, with_intercept=add_intercept)
        else:
                trend = self._compute_trend(timesteps, with_intercept=add_intercept)
        data_trended = data + trend
        return data_trended

    def get_trend(self, sequence_length, start_timestep=0, with_intercept=True):
        timesteps = self._make_timesteps(sequence_length, self.slope.ndim, start_timestep=start_timestep)
        trend = self._compute_trend(timesteps, with_intercept=with_intercept)
        return trend
    
    def get_trend_coeffs(self):
        return self.slope, self.intercept

    def _make_timesteps(self, sequence_length, ndims, start_timestep=0):
        timesteps = np.expand_dims(np.arange(sequence_length) + start_timestep, axis=[i for i in range(ndims) if i != self.trend_axis])
        return timesteps

    def _compute_trend(self, timesteps, with_intercept=True):
        if with_intercept:
            trend = timesteps * self.slope + self.intercept
        else:
            trend = timesteps * self.slope
        return trend

class Standardizer:

    def __init__(self, axis = None) -> None:
        self.mean = None
        self.std = None
        self.axis = axis

    def fit(self, data, mask=None):

        if mask is not None:
            marray = np.ma.array(data, mask=mask)
        else:
            marray = data.to_numpy()
        
        if self.axis is None:

            if np.isnan(marray.mean()):
                self.mean = np.ma.masked_invalid(marray).mean()
                self.std = np.ma.masked_invalid(marray).std()
            else:            
                self.mean = marray.mean()
                self.std = marray.std()
        else:
                self.mean = np.ma.masked_invalid(marray).mean(self.axis)
                self.std = np.ma.masked_invalid(marray).std(self.axis)  


        return self

    def transform(self, data):
        data_standardized = (data - self.mean) / self.std
        return data_standardized

    def inverse_transform(self, data):
        data_raw = data * self.std + self.mean
        return data_raw


class Normalizer:

    def __init__(self) -> None:
        self.min = None
        self.max = None
    
    def fit(self, data, mask=None):
        if mask is not None:
            marray = np.ma.array(data, mask=mask)
            self.min = marray.min()
            self.max = marray.max()
        else:
            self.min = data.min()
            self.max = data.max()
        return self

    def transform(self, data):
        data_normalized = (data - self.min) / (self.max - self.min)
        return data_normalized

    def inverse_transform(self, data):
        data_raw = data * (self.max - self.min) + self.min
        return data_raw


class PreprocessingPipeline:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.steps = []
        self.fitted_preprocessors = []

    def fit(self, data, mask=None):
        data_processed = data
        for step_name, preprocessor in self.pipeline:
            preprocessor.fit(data_processed, mask=mask)
            data_processed = preprocessor.transform(data_processed)
            self.steps.append(step_name)
            self.fitted_preprocessors.append(preprocessor)
        return self

    def transform(self, data, step_arguments=None):
        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")
            
        data_processed = data
        for step, preprocessor in zip(self.steps, self.fitted_preprocessors):
            if step in step_arguments.keys():
                args = step_arguments[step]
            else:
                args = dict()
            data_processed = preprocessor.transform(data_processed, **args)
        return data_processed

    def inverse_transform(self, data, step_arguments=None):
        if step_arguments is None:
            step_arguments = dict()
        for a in step_arguments.keys():
            if a not in self.steps:
                raise ValueError(f"{a} not in preprocessing steps!")
            
        data_processed = data
        for step, preprocessor in zip(reversed(self.steps), reversed(self.fitted_preprocessors)):
            if step in step_arguments.keys():
                args = step_arguments[step]
            else:
                args = dict()
            data_processed = preprocessor.inverse_transform(data_processed, **args)
        return data_processed

    def get_preprocessors(self, name=None):
        if name is None:
            return self.fitted_preprocessors
        else:
            idx = np.argwhere(np.array(self.steps) == name).flatten()
            if idx.size == 0:
                raise ValueError(f"{name} not in preprocessing steps!")
            return self.fitted_preprocessors[int(idx)]
    
    def add_fitted_preprocessor(self, preprocessor, name, index=None):
        if index is None:
            self.fitted_preprocessors.append(preprocessor)
            self.steps.append(name)
        else:
            self.fitted_preprocessors.insert(index, preprocessor)
            self.steps.insert(index, name)


from itertools import product
import random
class config_grid:
    def __init__(self, hyperparameter_dict = None):
            self.hyperparameter_dict = hyperparameter_dict 
    def full_grid(self):
        if self.hyperparameter_dict is None:
            raise ValueError(f"Provide a range of possibilities for at least one parameter!")
        output = []
        ranges = list(self.hyperparameter_dict.values())
        combinations = list(product(*ranges))
        for combo in combinations:
            output.append(dict(zip(self.hyperparameter_dict.keys(), combo)))
        return output 
    def draw_random(self, num_samples, seed = None ):
        population = self.full_grid()
        if seed is not None:
            random.seed(seed)
        return random.sample(population, num_samples)
    


def pole_centric(ds, pole = 'North'):
    
    d1 = ds.isel(lon = np.arange(0,len(ds.lon)/2).astype('int'))
    d2 = ds.isel(lon = np.arange(len(ds.lon)/2,len(ds.lon)).astype('int')).isel(lat=slice(None, None, -1)).isel(lon=slice(None, None, -1)).assign_coords(lon = d1.lon)

    if pole == 'North':
        return xr.concat([d1, d2], dim = 'lat').transpose(..., 'lat','lon')
    else:
        return xr.concat([d2,d1], dim = 'lat').transpose(..., 'lat','lon')
    

   
def reverse_pole_centric(ds,  pole = 'North'):
    
    d1 = ds.isel(lat = np.arange(0,len(ds.lat)/2).astype('int'))
    d2 = ds.isel(lat = np.arange(len(ds.lat)/2,len(ds.lat)).astype('int')).isel(lon=slice(None, None, -1))

    if pole == 'North':
        return xr.concat([d1, d2.isel(lat=slice(None, None, -1)).assign_coords(lon = d1.lon + 180)], dim = 'lon').transpose(..., 'lat','lon')
    else:
        return xr.concat([d2,d1.isel(lat=slice(None, None, -1)).assign_coords(lon = d1.lon + 180)], dim = 'lon').transpose(..., 'lat','lon')
    

def smoother(ds, smoother_kernel = 10):
    from scipy.ndimage import convolve
    kernel_lat = np.ones(( smoother_kernel, 1)) / smoother_kernel
    # Apply the moving average filter in latitude using convolution
    smoothed_data_lat = convolve(ds, kernel_lat, mode='constant')
    # Create the filter kernel for longitude
    kernel_lon = np.ones(( 1, smoother_kernel)) / smoother_kernel
    # Apply the moving average filter in longitude using convolution
    smoothed_data_lon = convolve(smoothed_data_lat, kernel_lon, mode='constant')
    # Print the smoothed data in latitude and longitude
    return smoothed_data_lon

def zeros_mask_gen(obs, smoother_kernel = 10):
    ls = [obs[np.arange(0,len(obs),12)+i,...].max('time').squeeze() for i in range(12)]
    ls = [ds.where(ds>0, 0) for ds in ls]
    ls = [ds.where(ds==0, 1) for ds in ls]
    for ind, ds in enumerate(ls):
        ls[ind][:] = smoother(ds, smoother_kernel)
    ls = [ds.where(ds>0, 0) for ds in ls]
    ls = [ds.where(ds==0, 1) for ds in ls]
    active_mask = xr.concat(ls, dim = 'month').assign_coords(month = np.arange(1,13)).to_dataset(name = 'active_mask')

    active_grid= (active_mask * (active_mask.sum(['lat','lon']).max('month')/active_mask.sum(['lat','lon']))).rename({'active_mask' : 'active_grid'})

    ls = [obs[np.arange(0,len(obs),12)+i,...].min('time').squeeze() for i in range(12)]
    ls = [ds.where(ds>=0.90, 0) for ds in ls]
    ls = [ds.where(ds==0, 1) for ds in ls]
    full_ice_mask = xr.concat(ls, dim = 'month').assign_coords(month = np.arange(1,13)).to_dataset(name = 'full_ice_mask')

    return xr.combine_by_coords([active_grid, active_mask, full_ice_mask])



def segment(ds, num_segments = 9):
    lon = ds.shape[-1]
    return xr.concat([ds[..., i * int(lon/num_segments): (i+1)*int(lon/num_segments)].assign_coords(lon = ds[...,0:int(lon/num_segments)].lon) for i in range(num_segments)], dim = 'channels')

def reverse_segment(ds):
    channels_axis = ds.dims.index('channels')
    return xr.concat([ds.isel(channels = i).expand_dims('channels', axis = channels_axis).assign_coords(lon = ds.lon + i* ds.shape[-1]) for i in range(ds.shape[channels_axis])], dim = 'lon')


def pad_xarray(ds, padding = 1, model = 'constant', constant_values = 0):
    padded = np.pad(ds, pad_width=padding, mode=model, constant_values=constant_values) 
    return xr.DataArray(
    padded, 
    dims=ds.dims, 
    coords={dim: np.arange(-1, len(ds[dim]) + 1) for dim in ds.dims})

