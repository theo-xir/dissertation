import pathlib
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from statistics import mean
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder', metavar='folder', type=str, help='the name of the target folder')

parserargs = parser.parse_args()

file1 = str(pathlib.Path(__file__).parent)+ "/ch4-ukghg-total_EUROPE_2016.nc"
data = xr.open_dataset(file1)

with open("coords.pkl", "rb") as input_file:
    coords = pickle.load(input_file)
lat=coords['lat']
lon=coords['lon']
x=min(data.lat.values, key=lambda x:abs(x-lat))
x=np.where(data.lat.values==x)[0][0]
y=min(data.lon.values, key=lambda x:abs(x-lon))
y=np.where(data.lon.values==y)[0][0]
size=32
temp=data.flux.values[x-size:x+size,y-size:y+size]

f=parserargs.folder
predictions=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/predictions.npy',allow_pickle=True)
groundtruth=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/groundtruth.npy',allow_pickle=True)
estimates=[]
for i in range(len(predictions)):
    estimates.append(float(np.tensordot(10**predictions[i].squeeze(),temp, axes=((0,1),(0,1)))))
truths=[]
for i in range(len(predictions)):
    truths.append(float(np.tensordot(10**groundtruth[i],temp, axes=((0,1),(0,1)))))
fig, ax = plt.subplots()
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.scatter(estimates,truths,s=2)
plt.xlabel("Prediction Based Estimate (e-8)")
plt.ylabel("Ground Truth Based Estimate (e-8)")
plt.show()

meanerror=mean([abs(estimates[i]-truths[i]) for i in range(len(estimates))])
print(meanerror)
print(mean(estimates))
print(mean(truths))
print(meanerror/mean(truths))
print(np.corrcoef(estimates,truths))