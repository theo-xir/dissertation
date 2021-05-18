import numpy as np
import math
from matplotlib import pyplot as plt
import yaml
from skimage import img_as_float
from skimage.metrics import structural_similarity
import matplotlib.lines as mlines
from statistics import mean
import pathlib
from scipy import interpolate
import sklearn.feature_selection
import xarray as xr
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder', metavar='folder', type=str, help='the name of the target folder')

parserargs = parser.parse_args()

f=parserargs.folder
predictions=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/predictions.npy',allow_pickle=True)
groundtruth=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/groundtruth.npy',allow_pickle=True)


minimum=np.nanmin(groundtruth)
predictions=predictions.flatten()
groundtruth=groundtruth.flatten()
print(stats.kstest(predictions.flatten(),groundtruth.flatten()))
bins=np.histogram(np.hstack((groundtruth.flatten(),predictions.flatten())), bins=40)[1]
plt.hist(groundtruth.flatten(),bins,alpha=1)
plt.hist(predictions.flatten(),bins,alpha=0.7)
plt.ylabel("Number of Occurences (e6)")
plt.xlabel("Value")
plt.legend(["Ground Truth","Predictions"])
plt.show()