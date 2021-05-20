import xarray as xr
import random
import numpy as np
import torch 
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import pathlib
import csv
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
hardmax={"PBLH":4000}
hardmin={"PBLH":100}

def load(num,filename=None, balancewind=False, fixwind=True, variables=["Wind","PBLH", "Pressure", "Sea_level_pressure", "Temperature"],size=32, timestep=1,readymaxs=None, readymins=None,exclude=[], focus=None,torb="t", getnum=10, randomise=True):
    if "Wind" in variables:
        variables.remove("Wind")
        variables.append("Wind_Speed")
        variables.append("Wind_Direction")
    global hardmax
    global hardmin
    met = {}
    keys=[]
    fp_data=[]
    if readymaxs:
        hardmax = readymaxs
    if readymins:
        hardmin = readymins

    for x in variables:
        met[x]=[]
    for f in os.listdir( str(pathlib.Path(__file__).parent)+"/met"):
        if f.endswith(".nc"):
            footprints=[ x for x in os.listdir( str(pathlib.Path(__file__).parent)+"/footprints") if x.endswith(f[-9:])]
            if len(footprints)>0:
                met_data = xr.open_dataset("met/" + f)
                for x in variables:
                    temp=met_data[x]
                    met[x].append(temp.isel(time=slice(0,len(temp.time),timestep)))
                temp=xr.open_dataset("footprints/"+footprints[0])
                fp_data.append(temp.isel(time=slice(0,len(temp.time),timestep)))
    for x in variables:
        met[x]=xr.concat(met[x],"time")
    fp_data=xr.concat(fp_data,"time")
    # print(fp_data)
    fpmin=0.999*np.log10(np.nanmin(fp_data.fp.values[np.nonzero(fp_data.fp.values)]))
    # fpmin=-10
    # input()
    # print(met_data.time.values)
    x=min(met_data.lat.values, key=lambda x:abs(x-fp_data.release_lat.values[0]))
    # print(x)
    x=np.where(met_data.lat.values==x)[0][0]
    # print(met_data.lat[x])
    y=min(met_data.lon.values, key=lambda x:abs(x-fp_data.release_lon.values[0]))
    # print(y)
    y=np.where(met_data.lon.values==y)[0][0]

    print(met_data.lat[x-32])
    print(met_data.lon[y-32])
    print(met_data.lat[x+32])
    print(met_data.lon[y+32])

    print(met_data.lat[0])
    print(met_data.lon[0])
    print(met_data.lat[-1])
    print(met_data.lon[-1])
    # input()
    # print(fp_data.release_lat.values[0])
    # print(fp_data.release_lon.values[0])
    if focus:
        timestofocus=[]
        temp=[met[focus].sel({"time":timepoint})[x-size:x+size,y-size:y+size].mean() for timepoint in met[focus].time.values if timepoint not in exclude]
        if torb=="t":
            temp=sorted(temp,reverse=True)
        else:
            temp=sorted(temp,reverse=False)
        for i in range(getnum):
            time=temp[i].time.values
            flag=False
            for v in variables:
                if np.isnan(list(met[v].sel({"time":time})[x-size:x+size,y-size:y+size].values)).any():
                    flag=True
            if time in exclude:
                flag=True
            if not flag:
                timestofocus.append(time)
    if focus:
        times=timestofocus
    else:
        times=fp_data.time.values
    
    # print(times)
    print(len(times))
    # print(timestep)
    # input()
    data = []
    truth = []
    maxs={}
    mins={}
    dates=[]
    negxwind=0
    posxwind=0
    negywind=0
    posywind=0
    while len(data)<num:
        print(len(data))
        if randomise:
            index= random.randint(0,len(times)-1)
        else:
            index=0
        timepoint=times[index]
        times=np.delete(times,index)
        if timepoint not in dates and timepoint not in exclude:
            sample={}
            if "Wind_Speed" in variables and fixwind:
                temp=met["Wind_Speed"].sel({"time":timepoint})[x-size:x+size,y-size:y+size].values
                temp2=met["Wind_Direction"].sel({"time":timepoint})
                xwind=np.cos(3*np.pi/2-2*np.pi/360.*temp2[x-size:x+size,y-size:y+size]).values
                ywind=np.sin(3*np.pi/2-2*np.pi/360.*temp2[x-size:x+size,y-size:y+size]).values
                sample["xWind"]=np.multiply(temp,xwind)
                sample["yWind"]=np.multiply(temp,ywind)
            for v in variables:
                if not fixwind or v not in ["Wind_Speed","Wind_Direction"]:
                    sample[v]=met[v].sel({"time":timepoint})[x-size:x+size,y-size:y+size].values
            # temp=fp_data.fp.sel({"time":timepoint})[x-size:x+size,y-size:y+size].values
            # # print(type(temp[0][0]))
            # if isinstance(temp[0,0], np.float32):
            #     # print("yas")
            #     temp[temp==0]=10^-9
            # elif isinstance(temp[0,0], np.float64):
            #     temp[temp==0]=10^-17
            t=np.nan_to_num(np.log10(fp_data.fp.sel({"time":timepoint})[x-size:x+size,y-size:y+size]).values,nan=0.0, posinf=1000, neginf=fpmin)
            #fig, ax = plt.subplots()
           # ax.contourf(t,levels=51)
            #plt.show()
            if not (np.isnan(list(sample.values())).any() or np.isnan(t).any()):
                dates.append(timepoint)
                if "xWind" not in list(sample.keys()) or not balancewind or not (sample["xWind"].mean()>0 and posxwind>=num/2 or sample["xWind"].mean()<=0 and negxwind>=num/2 or sample["yWind"].mean()>0 and posywind>=num/2 or sample["yWind"].mean()<=0 and negywind>=num/2):
                    if len(maxs.values())==0:
                        for k in list(sample.keys()):
                            maxs[k]=sample[k].max()
                            mins[k]=sample[k].min()
                    else:
                        for k in list(sample.keys()):
                            if maxs[k]<sample[k].max():
                                maxs[k]=sample[k].max()
                            if mins[k]>sample[k].min():
                                mins[k]=sample[k].min()
                    if "xWind" in list(sample.keys()) and sample["xWind"].mean()>0:
                        posxwind+=1
                    else:
                        negxwind+=1
                    if "yWind" in list(sample.keys()) and sample["yWind"].mean()>0:
                        posywind+=1
                    else:
                        negywind+=1
                    temp=[]
                    for k in list(sample.keys()):
                        temp.append(torch.Tensor(sample[k]))
                    if len(keys)==0:
                        keys=list(sample.keys())
                    sample=torch.stack(temp)
                    data.append(sample)
                    sampletruth=torch.Tensor(t)
                    truth.append(sampletruth)
    # print(len(data))
    # print(len(times))
    if randomise:
        order=list(range(num))
        random.shuffle(order)
        data=[data[i] for i in order]
    data=torch.stack(data)
    print(maxs)
    print(mins)
    for k in list(maxs.keys()):
        if k in list(hardmax.keys()):
            maxs[k]=hardmax[k]
        if k in list(hardmin.keys()):
            mins[k]=hardmin[k]
    print(maxs)
    with open('max.pkl', 'wb') as f:
        pickle.dump(maxs, f)
    with open('min.pkl', 'wb') as f:
        pickle.dump(mins, f)
    print(mins)
    print(posxwind)
    print(negxwind)
    print(posywind)
    print(negywind)
    for b in range(data.shape[0]):
        print(b)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):
                    if "xWind" in keys[i] or "yWind" in keys[i]  or "Temperature" in keys[i]:
                        # print(keys[i])
                        data[b][i][j][k]=2*(data[b][i][j][k]-mins[keys[i]])/(maxs[keys[i]]-mins[keys[i]])-1
                    else:
                        data[b][i][j][k]=(data[b][i][j][k]-mins[keys[i]])/(maxs[keys[i]]-mins[keys[i]])
    dataset = TensorDataset(data)
    # print(len(dataset))
    print(dataset[0][0].shape)
    if randomise:
        truth=[truth[i] for i in order]
    truth=torch.stack(truth)
    sz= truth[0].shape
    print(sz)
    variance=torch.empty(sz)
    for i in range(sz[0]):
        for j in range(sz[1]):
            variance[i,j]=torch.var(truth[:,i,j])
    print(variance)
    # plt.imshow(truth[0])
    # plt.show()
    print(truth[0])
    groundtruth= TensorDataset(truth)
    # loader = DataLoader(dataset, batch_size=len(dataset))
    # data = next(iter(loader))
    # mean,std=data[0].mean(), data[0].std()
    np.save('footprint.npy',variance)

    # opening the csv file in 'w+' mode
    file = open('timesused.csv', 'w+', newline ='')
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        print(dates)
        write.writerows(map(lambda x: [x], dates))
    return dataset, groundtruth, variance

# load(1)
