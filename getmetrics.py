import numpy as np
import pathlib
from statistics import mean
from image_similarity_measures.quality_metrics import issm, psnr, fsim, ssim, sre, sam, uiq
from sewar.full_ref import uqi, mse
from skimage.metrics import structural_similarity
import sklearn.feature_selection
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder', metavar='folder', type=str, help='the name of the target folder')
parser.add_argument('--folder2', metavar='folder', type=str, help='the name of the second target folder')
parserargs = parser.parse_args()


# # p=np.load(str(pathlib.Path(__file__).parent)+'/3yroct17pred.npy',allow_pickle=True)
# # g=np.load(str(pathlib.Path(__file__).parent)+'/3yroct17truth.npy',allow_pickle=True)
f=parserargs.folder
p=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/predictions.npy',allow_pickle=True)
g=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/groundtruth.npy',allow_pickle=True)
if parserargs.folder2:
    f=parserargs.folder2
    p1=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/predictions.npy',allow_pickle=True)
    g1=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/groundtruth.npy',allow_pickle=True)
    
# p=np.load(str(pathlib.Path(__file__).parent)+'/sanitypredictions.npy',allow_pickle=True)
# g=np.load(str(pathlib.Path(__file__).parent)+'/sanitygroundtruth.npy',allow_pickle=True)

structsims=[]
mses=[]
mi=[]
uqis=[]
for point in range(len(p)):
    uqis.append(uqi(p[point].squeeze(), g[point]))
    structsims.append(structural_similarity(p[point].squeeze(), g[point]))
    mses.append(mse(p[point].squeeze(), g[point]))
    mi.append(sklearn.feature_selection.mutual_info_regression(p[point].squeeze().flatten().reshape(-1, 1),g[point].flatten())[0])
if parserargs.folder2:
    for point in range(len(p1)):
        uqis.append(uqi(p1[point].squeeze(), g1[point]))
        structsims.append(structural_similarity(p1[point].squeeze(), g1[point]))
        mses.append(mse(p1[point].squeeze(), g1[point]))
        mi.append(sklearn.feature_selection.mutual_info_regression(p1[point].squeeze().flatten().reshape(-1, 1),g1[point].flatten())[0])

if parserargs.folder2:
    print(mean(mses[:int(len(p))]))
    print(mean(mses[int(len(p)):]))
    print(mean(uqis[:int(len(p))]))
    print(mean(uqis[int(len(p)):]))
    print(mean(structsims[:int(len(p))]))
    print(mean(structsims[int(len(p)):]))
    print(mean(mi[:int(len(p))]))
    print(mean(mi[int(len(p)):]))
    print(stats.mannwhitneyu(mses[:int(len(p))], mses[int(len(p)):]))
    print(stats.mannwhitneyu(uqis[:int(len(p))], uqis[int(len(p)):]))
    print(stats.mannwhitneyu(structsims[:int(len(p))], structsims[int(len(p)):]))
    print(stats.mannwhitneyu(mi[:int(len(p))], mi[int(len(p)):]))
else:
    print(mean(mses))
    print(mean(uqis))
    print(mean(structsims))
    print(mean(mi))