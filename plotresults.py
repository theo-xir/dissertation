import pathlib
from matplotlib import pyplot as plt
import numpy as np
import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--folder', metavar='folder', type=str, help='the name of the target folder')

# parserargs = parser.parse_args()

# f=parserargs.folder
# p=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/predictions.npy',allow_pickle=True)
# g=np.load(str(pathlib.Path(__file__).parent)+'/'+f+'/groundtruth.npy',allow_pickle=True)

p = np.load(str(pathlib.Path(__file__).parent)+ "/sanitypredictions.npy")
g = np.load(str(pathlib.Path(__file__).parent)+ "/sanitygroundtruth.npy")

flag=False
point=0
rows=4
columns=6
while point<len(p):
    for i in range(1,rows*columns,2):
        ax1=plt.subplot(rows,columns, i)
        ax1.set_box_aspect(1)
        ax1.contourf(p[point].squeeze(),levels=51)
        plt.axis('off')
        ax2=plt.subplot(rows,columns, i+1)
        ax2.set_box_aspect(1)
        ax2.contourf(g[point].squeeze(),levels=51)
        plt.axis('off')
        point+=1
    plt.tight_layout()
    plt.show()
