import numpy as np
import math
from matplotlib import pyplot as plt
import yaml
from skimage import img_as_float
# from image_similarity_measures.quality_metrics import issm
# from skimage.metrics import structural_similarity
# preds=np.load('predictions.npy')
# truth=np.load('groundtruth.npy')

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

def weightedMSE(preds,truth,weights):
    return ((preds - truth)**2*weights).mean()

# a=np.array([[1,1,1],[1,1,1],[1,1,1]])
# b=np.array([[1,1,1],[1,2,1],[1,1,1]])
# print(np.expand_dims(np.array([[1,2,2],[2,3,1]]),axis=2).shape)
# a=img_as_float(np.expand_dims(np.array([[1,2,2],[2,3,1]]),axis=2))
a = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32)
# b=img_as_float(np.expand_dims(np.array([[1,2,3],[4,5,6]]),axis=2))
# print(issm(a,a))
# print(structural_similarity(a,a,multichannel=True))
# w=getWeightMatrix(3)
# print(w)
# print(MSE(a,b))
# print(weightedMSE(a,b,w))
# print(getWeightMatrix(2))

p=np.load('predictions.npy')
g=np.load('groundtruth.npy')
# print(p==g)

for point in range(len(p)):
    # print(issm(p[point], g[point]))
    ax1=plt.subplot(1, 2, 1)
    ax1.contourf(p[point],levels=51)
    ax2=plt.subplot(1, 2, 2)
    ax2.contourf(g[point],levels=51)
    plt.show()
