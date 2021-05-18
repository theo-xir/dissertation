import pickle
import numpy as np
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt

with open("myfile.pkl", "rb") as input_file:
    e = pickle.load(input_file)

metrics=["MSE", "UQI", "SSIM", "MI"]
variables=['xWind', 'yWind', 'PBLH', 'Pressure', 'Sea_level_pressure', 'Temperature']

table=np.empty((len(metrics),len(variables)))

nans=np.argwhere(np.isnan(e['xWind']))
for i in metrics:
    for n in nans:
        print(n)
        del e[i][n[0]]
for i in variables:
    for n in nans:
        del e[i][n[0]]

for i in range(len(metrics)):
    # print(np.isnan(e[metrics[i]]).any())
    for j in range(len(variables)):
        # print(i,j)
        # print(len(e[j]))
        # print(np.corrcoef(e[i],e[j]))
        # print(np.isnan(e[variables[j]]).any())
        # print(np.corrcoef(e[metrics[i]],e[variables[j]]))
        table[i][j]=np.corrcoef(e[metrics[i]],e[variables[j]])[0][1]


fig, ax = plt.subplots()
im = ax.imshow(table)


ax.set_yticks(np.arange(len(metrics)))
ax.set_xticks(np.arange(len(variables)))
ax.set_yticklabels(metrics)
ax.set_xticklabels(variables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
for i in range(len(metrics)):
    for j in range(len(variables)):
        text = ax.text(j,i, round(table[i, j],3),
                       ha="center", va="center", color="w")
cbarlabel="Correlation"
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

# ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()