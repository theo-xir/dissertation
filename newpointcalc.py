from models import UNetSmall
import torch
from argparse import Namespace
import yaml
import pathlib
import os
import xarray as xr 
import numpy as np
import csv
import torch
import torch.nn as nn
import LoadData
from datetime import datetime
import pickle
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder', metavar='f', type=str, help='the name of the target folder')
parser.add_argument('--num', metavar='num', type=str, help='the number of points to load')

parserargs = parser.parse_args()
# print(parserargs)
with open(str(pathlib.Path(__file__).parent)+'/'+parserargs.f+'/args.yaml') as file:
    args=Namespace(**yaml.load(file,Loader=yaml.FullLoader))
model= torch.load(str(pathlib.Path(__file__).parent)+"/"+parserargs.f+"/state_dict_model.pt", map_location=torch.device('cpu'))
model.eval()
# input()
#6000
# mx={'xWind': 34.258717, 'yWind': 31.284176, 'PBLH': 4000, 'Pressure': 104466.3}
# mn={'xWind': -29.40597, 'yWind': -31.531086, 'PBLH': 100, 'Pressure': 89367.64}  
#1000
# max={'xWind': 34.258717, 'yWind': 25.762285, 'PBLH': 4000, 'Pressure': 104437.5}
# min={'xWind': -23.834251, 'yWind': -26.349575, 'PBLH': 100, 'Pressure': 89379.26}
# 6000 three years
# mx={'xWind': 34.258717, 'yWind': 27.624998, 'PBLH': 4000, 'Pressure': 105359.5}
# mn={'xWind': -25.324139, 'yWind': -31.531086, 'PBLH': 100, 'Pressure': 87646.14}
with open(str(pathlib.Path(__file__).parent)+'/'+parserargs.f+'/max.pkl', 'rb') as f:
    mx = pickle.load(f)
with open(str(pathlib.Path(__file__).parent)+'/'+parserargs.f+'/min.pkl', 'rb') as f:
    mn = pickle.load(f)
start=datetime.now()
dataset, groundtruth, variance = LoadData.load(int(parserargs.num),balancewind=args.balancewind, fixwind=args.fixwind, variables=["Wind","PBLH", "Pressure"],size=32, readymaxs=mx,readymins=mn,randomise=False)
loaded=datetime.now()
# print(dataset)

results={"groundtruth":[],"logits":[]}
print(len(dataset))
for i in range(len(dataset)):
    s=dataset[i][0]
    t=groundtruth[i][0]
    with torch.no_grad():
        logits = (model(s.unsqueeze(0)))
        logits=logits.cpu().numpy()
        results["groundtruth"].append(list(t.numpy()))
        results["logits"].append(list(logits))
passedthrough=datetime.now()
# print(type(results["groundtruth"]))
# results["groundtruth"]= [np.asarray(t) for t in results["groundtruth"]]
# results["logits"]= [np.asarray(t) for t in results["logits"]]
np.save(str(pathlib.Path(__file__).parent)+"/sanitypredictions.npy",results['logits'])
np.save(str(pathlib.Path(__file__).parent)+"/sanitygroundtruth.npy",results['groundtruth'])
done=datetime.now()

print(start)
print(loaded)
print(passedthrough)
print(done)