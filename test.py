# supress warnings
import warnings
warnings.filterwarnings('ignore') # don't output warnings

import os
import xarray as xr
import cftime
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
# import xesmf as xe
from pathlib import Path
import glob
from data_locations import LOC_FORECASTS_SI, LOC_OBSERVATIONS_SI
from preprocessing import align_data_and_targets, create_mask
from preprocessing import AnomaliesScaler_v1, Standardizer, PreprocessingPipeline
from torch_datasets import XArrayDataset
import numpy as np
from numpy import meshgrid, deg2rad, gradient, sin, cos
from xarray import DataArray
import xarray as xr


from module_FINAL_global_avg import earth_radius, area_grid
from module_FINAL_plotting_maps import plot_single_map_wmo
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.transforms as transforms
from preprocessing import Detrender

def detrending(ds, reference = None, remove_intercept = False, version = 1):
     ref_lead_time = False
     if reference is None:
          reference = ds
     if 'lead_time' in reference.dims:
          version = version
     else:
          version = None


     time = ds['time']
     detrender = Detrender(trend_dim='time', deg=1, version = version)
     detrender.fit(reference) 
     out = detrender.transform(ds,  remove_intercept=remove_intercept)
     out['time'] = time
     return out
     

def ice_extent(ds, Pole = 'North', NPSProj = False):

     area = area_grid(obs_in.lat, obs_in.lon) 
     if NPSProj:
          area = xr.ones_like(area) * 26450**2 * land_mask
          pole = None
     else:
          area = area * land_mask
     
    
     if Pole == 'North':
          ds = ds.where(ds.lat > 40, drop = True)
          area = area.where(area.lat > 40, drop = True)
     else:
          ds = ds.where(ds.lat < -40, drop = True)
          area = area.where(area.lat < -40, drop = True)

     area = xr.ones_like(ds) * area
     out =  area.where(ds >= 0.15,0).sum(['lat','lon'])/1e12
     out = out.where(out>0, np.nan)

     return out


def ice_area(ds, Pole = 'North', NPSProj = False):

     area = area_grid(obs_in.lat, obs_in.lon) 
     if NPSProj:
          area = xr.ones_like(area) * 26450**2 * land_mask
          pole = None
     else:
          area = area * land_mask

     if Pole == 'North':
        ds = ds.where(ds.lat > 40, drop = True)
        area = area.where(area.lat > 40, drop = True)
     else:
        ds = ds.where(ds.lat < -40, drop = True)
        area = area.where(area.lat < -40, drop = True)

     out =  (ds * area).sum(['lat','lon'])/1e12
     out = out.where(out>0, np.nan)

     return out

def IIEE(ds1,ds2, NPSProj = False):

     
     d1 = xr.ones_like(ds1)
     d1 = d1.where(ds1 >= 0.15 , 0)
     d1 = d1.where(~np.isnan(ds1))

     d2 = xr.ones_like(ds2)
     d2 = d2.where(ds2 >= 0.15 , 0)
     d2 = d2.where(~np.isnan(ds2))

     area = area_grid(obs_in.lat, obs_in.lon) 
     if NPSProj:
          area = xr.ones_like(area) * 26450**2 * land_mask
     else:
          area = area * land_mask
     return (np.abs(d1 - d2) * area).sum(['lat','lon'])/1e12
 

def rmse_full(d1,d2):
     #  area = area_grid(d1.lat, d1.lon)
      #  return (np.abs(d1 - d2) * area).sum(['lat','lon'])/area.sum(['lat','lon'])
      return np.sqrt(((d1 - d2)**2).mean())

def corr_map(ds1, ds2):
     #    mean1 = ds1.mean(['lat','lon'])
     #    mean2 = ds2.mean(['lat','lon'])
     #    std1 = ds1.std(['lat','lon'])
     #    std2 = ds2.std(['lat','lon'])
     #    covariance = ((ds1 - mean1) * (ds2 - mean2)).mean(['lat','lon'])
     #    return (covariance / (std1 * std2))
     se1 = np.sqrt((ds1**2).mean(['lat','lon']))
     se2 = np.sqrt((ds2**2).mean(['lat','lon']))
     return (ds1 * ds2).mean(['lat','lon']) / (se1 * se2)

def rmse(ds1,ds2):
     return np.sqrt(((ds1 - ds2)**2).mean('time'))

def pole_finder(ds, pole = 'North'):
     if  pole == 'North':
          return ds.where(ds.lat>40, drop = True)
     elif pole == 'South':
          return ds.where(ds.lat<-40, drop = True)
     else:
          return ds
     
def extract_tragte_month(ds, lead_time = 12,target_month = 3):      
    ls = [ds.sel(lead_time = lead_month + 1 ).where(np.mod(np.mod(ds.sel(lead_time = lead_month + 1 ).time,100) + lead_month,12) == np.mod(target_month,12), drop = True) for lead_month in range(lead_time)]
    for ind, ds in enumerate(ls):
        (added_year, month) = np.divmod(np.mod(ds.time,100) + ind,12)
        if target_month == 12 :
             added_year = 0
             month = 12
        ds['time'] = (ds.time/100 + added_year).astype(int)*100 + month
        ls[ind] = ds
    return xr.concat(ls, dim = 'lead_time').transpose('time','lead_time',...)

def call_to_plot(months_to_plot, adj_data_list = ['Deb', 'Det', 'NN'] ,Antarctic = False, show_polar_stereo = True, central_longitude= 0 , show_SIE = False, fig_name = None, fig_dir = None, save  =False, NPSProj = False):

          if Antarctic:
               lat_lims = [-90,-40]
               pole = 'South'
          else:
               lat_lims = [40,90]
               pole = 'North'

          if show_polar_stereo:
               crs = ccrs.NorthPolarStereo()    
               if max(lat_lims) < 0:
                    crs = ccrs.SouthPolarStereo()   
          else:
               crs = ccrs.PlateCarree(central_longitude=central_longitude)

          fig, ax = plt.subplots(nrows=months_to_plot ,
                           ncols=len(adj_data_list) + 2, 
                           figsize=(25,10*months_to_plot), 
                           subplot_kw={'projection' : crs}) 

          if months_to_plot == 1:
               ax = ax[None, :]
          
          figure = 'FigX'
          vmin  = 0
          vmax  = 1
          
          if show_SIE:
               varx = 'SIE'
               units = 'm$^2$'
          cmap_var = mpl.cm.Blues
          if NPSProj:
               pole = None
          
          for lead_time in range(months_to_plot):
               
               axis = ax[lead_time, 0]
               axis.text(-0.5,1.1, f'lead_month: {lead_time + 1}',  fontsize=15, transform=axis.transAxes)

               data = model.sel(lead_time = lead_time + 1 )
               data = data.where(~np.isnan(data), drop = True)
               ref = observation.sel(time = data.time.values)

               iiee = IIEE(pole_finder(data, pole), pole_finder(ref, pole), NPSProj = NPSProj).mean('time').values
               corr = corr_map(pole_finder(data, pole), pole_finder(ref, pole)).mean('time').values
               rmse_ = rmse_full(pole_finder(data, pole), pole_finder(ref, pole)).values

               if lead_time ==  0:
                    label = f'CanCM4 \n corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'
               else:
                    label = f'corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'

               plot_single_map_wmo(fig, ax[lead_time, 0] , data.mean('time') , title = label, cbar = False, vmin=vmin, ds_sig=None,
                        vmax=vmax,polar_stereo=show_polar_stereo, central_longitude=central_longitude,   lat_lims=lat_lims, cmap=cmap_var, NPSProj = NPSProj)
               
               count = 0
               if 'Deb' in adj_data_list:
                    count = count + 1
                    corr = corr_map( pole_finder(bias_adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole)).mean('time').values
                    iiee = IIEE( pole_finder(bias_adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole), NPSProj = NPSProj).mean('time').values
                    rmse_ = rmse_full( pole_finder(bias_adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole)).values

                    if lead_time ==  0:
                         label = f'bias_adj \n corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'
                    else:
                         label = f'corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'

                    plot_single_map_wmo(fig, ax[lead_time, count] ,  bias_adj.sel(time = data.time.values).isel(lead_time = lead_time).mean('time'), title = label, cbar = False, vmin=vmin,ds_sig=None,
                         vmax=vmax,polar_stereo=show_polar_stereo, central_longitude=central_longitude,    lat_lims=lat_lims, cmap=cmap_var, NPSProj = NPSProj)

               if 'Det' in adj_data_list:
                    count = count + 1
                    corr = corr_map( pole_finder(trend_adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole)).mean('time').values
                    iiee = IIEE( pole_finder(trend_adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole), NPSProj = NPSProj).mean('time').values
                    rmse_ = rmse_full( pole_finder(trend_adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole)).values
          
                    if lead_time ==  0:
                         label = f'Trend_adj \n corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'
                    else:
                         label = f' corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'

                    plot_single_map_wmo(fig, ax[lead_time, count] ,  trend_adj.sel(time = data.time.values).isel(lead_time = lead_time).mean('time'), title = label, cbar = False, vmin=vmin,ds_sig=None,
                         vmax=vmax,polar_stereo=show_polar_stereo, central_longitude=central_longitude,    lat_lims=lat_lims, cmap=cmap_var, NPSProj = NPSProj)
               
               if 'NN' in adj_data_list:
                    count = count + 1
                    corr = corr_map(pole_finder(adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole)).mean('time').values
                    iiee = IIEE(pole_finder(adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole), NPSProj = NPSProj).mean('time').values
                    rmse_ = rmse_full(pole_finder(adj.sel(time = data.time.values).isel(lead_time = lead_time), pole), pole_finder(ref, pole)).values
                    
                    if lead_time ==  0:
                         label = f'nn_adj \n corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'
                    else:
                         label = f'corr : {str(np.round(corr,2))}, rmse = {str(np.round(rmse_,2))}, IIEE : {str(np.round(iiee,2))} million Km^2'
                    
                    plot_single_map_wmo(fig, ax[lead_time, count] ,  adj.sel(time = data.time.values).isel(lead_time = lead_time).mean('time'), title = label, cbar = False, vmin=vmin,ds_sig=None,
                         vmax=vmax,polar_stereo=show_polar_stereo, central_longitude=central_longitude,    lat_lims=lat_lims, cmap=cmap_var, NPSProj = NPSProj)

               

               if lead_time == months_to_plot -1 :
                    cbar = True
               else:
                    cbar = False
               corr = corr_map(pole_finder(ref, pole), pole_finder(observation.sel(time = data.time.values), pole)).mean('time').values   

               if lead_time ==  0:
                    label = f'obs \n ({str(np.round(corr,2))})'
               else:
                    label = f'({str(np.round(corr,2))})'
               
               plot_single_map_wmo(fig, ax[lead_time, -1] ,  observation.sel(time = data.time.values).mean('time'), title = label, cbar = cbar, vmin=vmin,ds_sig=None,
                        vmax=vmax,polar_stereo=show_polar_stereo, central_longitude=central_longitude,    lat_lims=lat_lims, cmap=cmap_var, NPSProj = NPSProj)
               

               
               if save:
                    Path(fig_dir).mkdir(parents=True, exist_ok=True)
                    plt.savefig(fig_dir+'/'+fig_name,
                    bbox_inches='tight',
                    dpi=300)

        


def ice_extent_plot(target_month, lead_months,  adj_data_list : list ,pole = 'North', metric = 'RMSE', detrend=False, remove_intercept = False, NPSProj = False):


    colors = ['b','r','g','c','m','purple']
    labels = {'Deb': 'bias adjusted', 'Det': 'trend adjusted', 'NN' : 'NN adjusted'}


    adj_dict = {}
    for key in adj_data_list:
          adj_dict[key] = extract_tragte_month( data_dicts[key], lead_months, target_month)

    
    model = extract_tragte_month(ds_model, lead_months, target_month)
    observation = obs.sel(time = model.time)

    plt.figure(figsize = (8,8 * len(adj_data_list)))
    
    time = adj_dict[adj_data_list[0]].time
    ref = ice_extent(observation.sel(time = time), pole, NPSProj = NPSProj)
    CanESM5 = ice_extent(model.sel(time = time), pole, NPSProj = NPSProj)

    if detrend:
     #     trend_ref = ref.copy()

         CanESM5 = detrending(CanESM5,  remove_intercept = remove_intercept)
         ref = detrending(ref,  remove_intercept = remove_intercept)

    for ind, key in enumerate(adj_data_list):

        data_to_plot = ice_extent(adj_dict[key].sel(time = time), pole, NPSProj = NPSProj)
        if detrend:
               data_to_plot = detrending(data_to_plot, remove_intercept = remove_intercept)

        plt.subplot(len(adj_data_list),1,ind+1)
        ref.plot(linewidth = 2, color = 'k')
        for lead_month in range(1,lead_months + 1):
            if metric == 'RMSE':
               corr = rmse(CanESM5.sel(lead_time = lead_month) ,ref ).values
            else:
                corr = xr.corr(CanESM5.sel(lead_time = lead_month) ,ref , dim = 'time' ).values 
               

            CanESM5.sel(lead_time = lead_month).plot(color = colors[lead_month-1], linestyle = 'dashed', label = f'CanCM4 lead month {lead_month} {metric}: ({str(np.round(corr,2))})')

            if metric == 'RMSE':
               corr = rmse(ref,data_to_plot.sel(lead_time = lead_month)).values
            else:
                 corr = xr.corr(ref,data_to_plot.sel(lead_time = lead_month) , dim = 'time' ).values

            data_to_plot.sel(lead_time = lead_month).plot(color = colors[lead_month-1], label = f'{labels[key]} lead month {lead_month} {metric}: ({str(np.round(corr,2))})')
        
        if ind == 0:
               plt.title(pole + ' SIE')
        else:
               plt.title('')

        plt.ylabel('million km$^2$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.xticks(time.values,rotation = 45)
        
def ice_area_plot(target_month, lead_months,  adj_data_list : list ,pole = 'North', metric = 'RMSE', detrend=False, remove_intercept = False, NPSProj = False):


    colors = ['b','r','g','c','m','purple']
    labels = {'Deb': 'bias adjusted', 'Det': 'trend adjusted', 'NN' : 'NN adjusted'}


    adj_dict = {}
    for key in adj_data_list:
          adj_dict[key] = extract_tragte_month( data_dicts[key], lead_months, target_month)

    
    model = extract_tragte_month(ds_model, lead_months, target_month)
    observation = obs.sel(time = model.time)

    plt.figure(figsize = (8,8 * len(adj_data_list)))
    
    time = adj_dict[adj_data_list[0]].time
    ref = ice_area(observation.sel(time = time), pole, NPSProj = NPSProj)
    CanESM5 = ice_area(model.sel(time = time), pole, NPSProj = NPSProj)


    if detrend:
     #     trend_ref = ref.copy()

         CanESM5 = detrending(CanESM5,  remove_intercept = remove_intercept)
         ref = detrending(ref,remove_intercept = remove_intercept)
         

    for ind, key in enumerate(adj_data_list):

        data_to_plot = ice_area(adj_dict[key].sel(time = time), pole, NPSProj = NPSProj)
        if detrend:
               data_to_plot = detrending(data_to_plot,  remove_intercept = remove_intercept)

        plt.subplot(len(adj_data_list),1,ind+1)
        ref.plot(linewidth = 2, color = 'k')
        for lead_month in range(1,lead_months + 1):
            if metric == 'RMSE':
               corr = rmse(CanESM5.sel(lead_time = lead_month) ,ref ).values
            else:
                corr = xr.corr(CanESM5.sel(lead_time = lead_month) ,ref , dim = 'time' ).values 
               

            CanESM5.sel(lead_time = lead_month).plot(color = colors[lead_month-1], linestyle = 'dashed', label = f'CanCM4 lead month {lead_month} {metric}: ({str(np.round(corr,2))})')

            if metric == 'RMSE':
               corr = rmse(ref,data_to_plot.sel(lead_time = lead_month)).values
            else:
                 corr = xr.corr(ref,data_to_plot.sel(lead_time = lead_month) , dim = 'time' ).values

            data_to_plot.sel(lead_time = lead_month).plot(color = colors[lead_month-1], label = f'{labels[key]} lead month {lead_month} {metric}: ({str(np.round(corr,2))})')
        
        if ind == 0:
               plt.title(pole + ' SI Area')
        else:
               plt.title('')

        plt.ylabel('million km$^2$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.xticks(time.values,rotation = 45)       


import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns

def heat_map_data(ds, metric= 'RMSE', lead_times = 12, pole = 'North', mode = 'ice_extent', detrend = False, remove_intercept = False, NPSProj = False):
    ls = []
    
    if NPSProj:
          pole = None
    for target_month in range(12):
        ds_ = extract_tragte_month(ds,lead_times,target_month+1)
        ref = obs.sel(time = ds_.time)
        
        if mode == 'IIEE':
               iiee_list = []
               ref = pole_finder(ref, pole)
               ds_ = pole_finder(ds_, pole)
               for lt in range(lead_times):
                    d1 = ds_.isel(lead_time = lt)                  
                    d1 = d1.where(~np.isnan(d1), drop = True)
                    iiee_list.append(IIEE(d1, ref.sel(time = d1.time), NPSProj = NPSProj))

               ls.append(xr.concat(iiee_list , dim = 'lead_time').assign_coords(lead_time = ds_.lead_time))
               metric = 'IIEE'
        elif mode == 'grid_wise':
               d_ref = pole_finder(ref, pole)
               d_plot = pole_finder(ds_, pole)
               area = area_grid(d_ref.lat, d_ref.lon) 
               if NPSProj:
                    area = xr.ones_like(area) * 26450**2 * land_mask
               else:
                    area = area * land_mask

               if detrend:
               # trend_ref = d_ref.copy()

                    d_plot = detrending(d_plot,  remove_intercept = remove_intercept )
                    d_ref = detrending(d_ref, remove_intercept = remove_intercept )       

               if metric == 'RMSE':
                    met = rmse(d_ref, d_plot)
               else:
                    met = xr.corr(d_ref, d_plot, dim = 'time')

               ls.append((met * area).sum(['lat','lon'])/area.sum(['lat','lon']))

        else:

          if mode == 'ice_extent':
               d_ref = ice_extent(ref, pole, NPSProj = NPSProj)
               d_plot = ice_extent(ds_, pole, NPSProj = NPSProj)

          elif mode == 'ice_area':
               d_ref = ice_area(ref, pole, NPSProj = NPSProj)
               d_plot = ice_area(ds_, pole, NPSProj = NPSProj)
               
          if detrend:
               # trend_ref = d_ref.copy()

               d_plot = detrending(d_plot,  remove_intercept = remove_intercept )
               d_ref = detrending(d_ref, remove_intercept = remove_intercept )

          if metric == 'RMSE':
               ls.append(rmse(d_ref, d_plot))
          else:
               ls.append(xr.corr(d_ref, d_plot, dim = 'time'))
         
    return xr.concat( ls , dim = 'target_month').assign_coords(target_month = np.arange(1,13)).squeeze().to_dataset(name = metric)



def plot_heat_maps(adj_data_list:list ,lead_times = 12, metric= 'RMSE', pole = 'North', mode = "ice_extent", detrend = False,remove_intercept = False, NPSProj = False):
    if mode == 'grid_wise': 
          vmax = 0.1
          vmin = 0
    else:
          vmax = 1.5
          vmin = 0
    labels = {'Deb': 'bias adjusted', 'Det': 'trend adjusted', 'NN' : 'NN adjusted', 'Raw' : 'Raw'}

    plt.figure(figsize = (10 * len(adj_data_list),8))
    cmap = 'inferno'
    if metric == 'Corr':
        vmax = 1
        vmin = 0
        cmap = 'inferno_r'

     

    for ind, name in enumerate(adj_data_list):
        ax = plt.subplot(1,len(adj_data_list),ind + 1 )
        dataframe = heat_map_data(data_dicts[name],lead_times = lead_times, metric = metric, pole = pole, mode = mode, detrend = detrend, remove_intercept = remove_intercept, NPSProj = NPSProj).to_dataframe().reset_index().pivot(index='target_month', columns=f'lead_time')[metric]

        ax = sns.heatmap(dataframe, annot=True,  cmap = cmap ,linewidth=.5, vmax = vmax , vmin = vmin)
        plt.yticks(rotation = 0)
     #    plt.yticks(ticks = np.arange(1,13) - 0.5, labels = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec' ],rotation=0)
        if detrend:
          plt.text(0.01,1.05, f'{labels[name]} {mode} detrended test {metric} ', transform=ax.transAxes)
        else:
          plt.text(0.01,1.05, f'{labels[name]} {mode} test {metric}', transform=ax.transAxes)




def plot_IIEE_heat_maps(adj_data_list:list ,lead_times = 12,  pole = 'North', NPSProj = False):

    vmax = 2.5
    vmin = 0.5

    labels = {'Deb': 'bias adjusted', 'Det': 'trend adjusted', 'NN' : 'NN adjusted', 'Raw' : 'Raw'}

    plt.figure(figsize = (10 * len(adj_data_list),8))
    cmap = 'inferno'

    for ind, name in enumerate(adj_data_list):
        ax = plt.subplot(1,len(adj_data_list),ind + 1 )
        dataframe = heat_map_data(data_dicts[name], lead_times = lead_times , pole = pole, mode = 'IIEE', NPSProj =NPSProj ).mean('time').to_dataframe().reset_index().pivot(index='target_month', columns=f'lead_time')['IIEE']

        ax = sns.heatmap(dataframe, annot=True,  cmap = cmap ,linewidth=.5, vmax = vmax , vmin = vmin)
        plt.yticks(rotation = 0)
        plt.yticks(ticks = np.arange(1,13) - 0.5, labels = ['Jan','Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov', 'Dec' ],rotation=0)

        plt.text(0.01,1.05, f'{labels[name]} IIEE test', transform=ax.transAxes)