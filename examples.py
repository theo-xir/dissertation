#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:00:43 2020

@author: al18242
"""


import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

'''
netcdf stores structured multi-dimensional data common in atmospheric science
xarray is a library built to manipulate netcdf files in python by extending pandas functionality
xarray reads in a netcdf file as a 'dataset' which contains 'dataarray's, coordinate information and metadata
A datarray gives information on a single variable, and is esentially a highly extended numpy array
'''

#open the netcdf file using xarray and print a summary of the contents
#data is stored monthly as EUROPE_Met_YYYYMM.nc
metfile = "EUROPE_Met_slp_201802.nc"
met_data = xr.open_dataset(metfile)
# print(met_data.time[0])

'''
Temperature is in degrees c
Pressure is in Pascals
PBLH (Planetary boundary layer height) is in meters
Wind_Speed is in m/s
Wind_Direction is in degrees from north
'''

#to subset a single variable as a DataArray:
wind_speed = met_data["Wind_Speed"]
#or
temperature = met_data.Temperature

#the underlying numerical data can be extracted as a numpy array:
wind_speed_numpy = wind_speed.values

#datasets and dataarrays can be subset by coordinates as well, for example by using pandas datetimes:
time_subset = met_data.sel({"time":pd.to_datetime("2018-02-01 13:00")})

'''
Here is some code to plot the data. 
You should be able to make out some of the coastlines of africa and europe on the right and north america on the left
This is due to differences in land and sea roughness
In this time slice there is a nice storm heading across the ocean thats very clear as well
cartopy library can be used to add country border lines to the plot
'''

LON, LAT = np.meshgrid(met_data.lon, met_data.lat)
wind_to_plot = met_data["Wind_Speed"].sel({"time":pd.to_datetime("2018-02-01 13:00")})
fig, ax = plt.subplots()
cs = ax.contourf(LON, LAT, wind_to_plot,levels=51)
fig.colorbar(cs,ax=ax)
plt.show()

#Note that the default order of the data is not as expected for an image, and so using imshow gives a different result:
fig, ax = plt.subplots(1,2)
ax[0].imshow(wind_to_plot)
#this can be fixed by flipping the data:
ax[1].imshow(np.flip(wind_to_plot,axis=0))
plt.show()

#and similarly for footprints:
footprint_file = "MHD-10magl_EUROPE_201802.nc"
fp_data = xr.open_dataset(footprint_file)
# print(fp_data)

LON, LAT = np.meshgrid(fp_data.lon, fp_data.lat)
#here we take log10 of the footprints due to the exponential drop off of sensitivity 

print(fp_data.fp.sel({"time":pd.to_datetime("2018-02-15 13:00")}).values.max())

fig, ax = plt.subplots()
print(fp_data.fp.sel({"time":pd.to_datetime("2018-02-15 13:00")}).values)
ax.contourf(LON, LAT, np.log10(fp_data.fp.sel({"time":pd.to_datetime("2018-02-15 13:00")})),levels=51)
plt.show()

fig, ax = plt.subplots()
print(fp_data.fp.sel({"time":pd.to_datetime("2018-02-15 13:00")}).values)
ax.contourf(LON, LAT, fp_data.fp.sel({"time":pd.to_datetime("2018-02-15 13:00")}),levels=51)
plt.show()

'''
plot the wind direction over this
Note the wind direction may only make sense very close to the release location if the wind has changed rapidly, as the footprint is
a 30 day integration, while the wind field is a snapshot of a single time. 
As the wind direction is defined as clockwise from north, we have to convert this to a standard east = 0 degrees format to use with sin and cos
in order to extract the x and y components
'''
x=np.cos(3*np.pi/2-2*np.pi/360.*met_data["Wind_Direction"].sel({"time":pd.to_datetime("2018-02-15 13:00")}))[::10, ::10]
y=np.sin(3*np.pi/2-2*np.pi/360.*met_data["Wind_Direction"].sel({"time":pd.to_datetime("2018-02-15 13:00")}))[::10, ::10]
print(x[:10])
print(y[:10])
fig, ax = plt.subplots()
ax.quiver(LON[::10, ::10], LAT[::10, ::10],
                 np.cos(3*np.pi/2-2*np.pi/360.*met_data["Wind_Direction"].sel({"time":pd.to_datetime("2018-02-15 13:00")}))[::10, ::10],
                 np.sin(3*np.pi/2-2*np.pi/360.*met_data["Wind_Direction"].sel({"time":pd.to_datetime("2018-02-15 13:00")}))[::10, ::10],
                 angles='xy', scale_units='xy', scale=0.5)

# #the 'release' location of the footprint is stored in the file as well. For macehead (MHD) this is constant in time
ax.scatter(fp_data.release_lon[0], fp_data.release_lat[0], color="red")
# print("hi")
plt.show()

# '''
# You can also combine multiple files with xarray
# This is how you could load multiple months at once using the 'MultiFile dataset' function
# You could similarly combine the met and fp files together. This new dataset can then be used exactly as above
# This may be useful, as adjacent times likely contain similar met data and footprints and you may want to take every nth hour instead to get a more
# representative sample
# '''

# #met_data_3months = xr.open_mfdataset("/work/al18242/ML_summer_2020/EUROPE_Met_20180[1-3].nc", combine="by_coords")

# '''
# Calculating timeseries of 'mole fractions' - the quantity we observe.
# This is calculated as mole fractions = sum_over_x.y(footprint * emissions map) * units

# - you can think of this as summing the footprint, weighting by emissions (or vice versa, which is how we think of it scientifically)
# '''
# #emissions_map = xr.open_dataset("/work/al18242/ML_summer_2020/ch4_EUROPE_2013.nc")

# #use the first month of emissions as a constant emissions field to calculate the timeseries
# #mole_fraction_timeseries = 1e9*np.sum(emissions_map.flux.values[:,:,0:1] * fp_data.fp.values,axis=(0,1))

# #fig, ax = plt.subplots()
# #ax.plot(fp_data.time, mole_fraction_timeseries)
