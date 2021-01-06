import pyspedas
from pytplot import data_quants
import matplotlib.pyplot as plt
import numpy as np

probe = 4
data_rate = "brst"
trange = ["2015-10-07/11:44:24", "2015-10-07/11:44:59"]

print("Loading data.")
mms_fpi = pyspedas.mms.fpi(
    trange=trange,
    probe=probe,
    datatype="des-dist",
    data_rate=data_rate,
    time_clip=True,
)

raw_dist = data_quants["mms4_des_dist_brst"].values

data = raw_dist[0].mean(axis=0)
print(data.shape)
X, Y = np.meshgrid(np.arange(0, 360, 11.25), np.arange(0, 180, 11.25))
plt.contourf(X, Y, data)
plt.show()