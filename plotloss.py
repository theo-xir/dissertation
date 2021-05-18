import numpy as np
import pathlib
from scipy import stats
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('files', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='test', action='store_true')
parserargs = parser.parse_args()

l=parserargs.files

box_pts=5
legend=[]
if parserargs.test:
    for i in l:
        test = np.genfromtxt(str(pathlib.Path(__file__).parent)+'/'+str(i)+'/test.csv', delimiter=',')[1:,:]
        y_smooth = np.convolve(test[:,2], np.ones(box_pts)/box_pts, 'valid')
        plt.plot(test[2:-2,1], y_smooth)
        legend.append(i+" test")
if parserargs.train:
    for i in l:
        test = np.genfromtxt(str(pathlib.Path(__file__).parent)+'/'+str(i)+'/train.csv', delimiter=',')[1:,:]
        y_smooth = np.convolve(test[:,2], np.ones(box_pts)/box_pts, 'valid')
        plt.plot(test[2:-2,1], y_smooth)
        legend.append(i+" train")
plt.legend(legend)
plt.ylim(0,12)
plt.ylabel("MSE Loss")
plt.xlabel("Number of Steps")
plt.show()