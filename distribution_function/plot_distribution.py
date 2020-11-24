"""Average across all energies & plot angle-angle
pcolormesh for 16 evenly spaced time steps.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants, tplot

# Load data
trange = ["2015-10-07/11:44:15", "2015-10-07/11:45:10"]
# probe = ["1", "2", "3", "4"]
probe = "4"
data_rate = "brst"
# mms_fgm = pyspedas.mms.fgm(trange=trange,
#                            probe=probe,
#                            data_rate=data_rate,
#                            time_clip=True)

mms_fpi = pyspedas.mms.fpi(
    trange=trange, probe=probe, datatype="dis-dist", data_rate=data_rate, time_clip=True
)

dist = data_quants["mms4_dis_dist_brst"].values
print(np.shape(dist))

theta = np.linspace(0, 180, 16)
phi = np.linspace(0, 360, 32)

fig, axs = plt.subplots(4, 4)

TIME_INDEX = 30

for i, ax in enumerate(axs.reshape(-1)):
    data = np.mean(dist, axis=1)
    index_step = np.shape(dist)[0] // 16
    data = data[index_step * i, :, :]
    print(np.shape(data))
    ax.pcolormesh(phi, theta, data)
    # ax.axis('equal')

plt.show()
