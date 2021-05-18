import pickle
import numpy as np
import math
from matplotlib import pyplot as plt
import yaml
from skimage import img_as_float
from image_similarity_measures.quality_metrics import issm, psnr, fsim, ssim, sre, sam, uiq
from sewar.full_ref import uqi
from skimage.metrics import structural_similarity
import matplotlib.lines as mlines
from statistics import mean
import pathlib
from scipy import interpolate
import sklearn.feature_selection
# import torch
# from models import UNetSmall as UNet
import xarray as xr
from scipy import stats
import json
import os

def MSE(preds,truth):
    return ((preds - truth)**2).mean()

def getWeightMatrix(size):
    m=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            dist=math.sqrt((i+1-(size+1)/2)**2+(j+1-(size+1)/2)**2)
            if not dist==0:
                m[i,j]=1/dist
            else: m[i,j]=1
    return m


timestep=1
variables=["Wind_Speed", "Wind_Direction","PBLH", "Pressure", "Sea_level_pressure", "Temperature"]
met={}
for x in variables:
    met[x]=[]
for f in os.listdir( str(pathlib.Path(__file__).parent)+"/met"):
    print(f)
    if f.endswith(".nc"):
        met_data = xr.open_dataset("met/" + f)
        for x in variables:
            met[x].append(met_data[x].isel(time=slice(0,-1,timestep)))
for x in variables:
    met[x]=xr.concat(met[x],"time")
size=32
lat=53.32611
lon=-9.90387
x=min(met_data.lat.values, key=lambda x:abs(x-lat))
x=np.where(met_data.lat.values==x)[0][0]
y=min(met_data.lon.values, key=lambda x:abs(x-lon))
y=np.where(met_data.lon.values==y)[0][0]
import csv
#f=str(35)
predictions=np.load('predictions.npy',allow_pickle=True)
groundtruth=np.load('groundtruth.npy',allow_pickle=True)

with open('timesused.csv', newline='') as f:
    reader = csv.reader(f)
    times = list(reader)

means={}
means["MSE"]=[]
means["UQI"]=[]
means["SSIM"]=[]
means["MI"]=[]
for v in ["xWind", "yWind","PBLH", "Pressure", "Sea_level_pressure", "Temperature"]:
    means[v]=[]
#print(times)
for timepoint in reversed(times):
    #print(len(means["MSE"]))
    sample={}
    timepoint=timepoint[0][:-10]
    #print(timepoint)
    #print(timepoint in met["Wind_Speed"].time.values)
    temp=met["Wind_Speed"].sel({"time":timepoint})[x-size:x+size,y-size:y+size].values
    temp2=met["Wind_Direction"].sel({"time":timepoint})
    xwind=np.cos(3*np.pi/2-2*np.pi/360.*temp2[x-size:x+size,y-size:y+size]).values
    ywind=np.sin(3*np.pi/2-2*np.pi/360.*temp2[x-size:x+size,y-size:y+size]).values
    sample["xWind"]=np.multiply(temp,xwind)
    sample["yWind"]=np.multiply(temp,ywind)
    for v in variables:
        if v not in ["Wind_Speed","Wind_Direction"]:
            sample[v]=met[v].sel({"time":timepoint})[x-size:x+size,y-size:y+size].values
    if not (np.isnan(list(sample.values())).any()) and len(list(sample.values()))>0:
        c=len(means["MSE"])+1
        means["MSE"].append(MSE(predictions[-c], groundtruth[-c]))
        means["UQI"].append(uqi(predictions[-c], groundtruth[-c]))
        means["SSIM"].append(structural_similarity(predictions[-c], groundtruth[-c]))
        means["MI"].append(sklearn.feature_selection.mutual_info_regression(predictions[-c].squeeze().flatten().reshape(-1, 1),groundtruth[-c].flatten())[0])
        for v in sample.keys():
            if np.isnan(sample[v].mean()):
                print("nan point")
                print(list(sample[v].flatten()))
                plt.imshow(predictions[-c])
                plt.savefig("prediction.png")
                plt.imshow(groundtruth[-c])
                plt.savefig("groundtruth.png")
                print(np.isnan(list(sample[v])).any())
            means[v].append(sample[v].mean())
    if len(means["MSE"])==1200:
        break
print(means)
output = open('myfile.pkl', 'wb')
pickle.dump(means, output)
output.close()

