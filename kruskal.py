import numpy as np
import pathlib
from scipy import stats
import argparse

def MSE(preds,truth):
    return ((preds - truth)**2).mean()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('files', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')
parserargs = parser.parse_args()

l=parserargs.files
metric={}
for f in l:
    metric[f]=[]
    p=np.load(str(pathlib.Path(__file__).parent)+'/'+str(f)+'/predictions.npy',allow_pickle=True)
    g=np.load(str(pathlib.Path(__file__).parent)+'/'+str(f)+'/groundtruth.npy',allow_pickle=True)
    for point in range(len(p)):
        metric[f].append(MSE(p[point].squeeze(), g[point]))
print(stats.kruskal(*[metric[f] for f in l]))
