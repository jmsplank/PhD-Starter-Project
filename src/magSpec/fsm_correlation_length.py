import pyspedas
import numpy as np
import matplotlib.pyplot as plt
from phdhelper.math.signal_processing import autocorrelate
from pytplot import data_quants

trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
# trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
probe = ["1", "2", "3", "4"]
# probe = "1"
data_rate = "brst"

pyspedas.mms.fsm(trange=trange, probe=probe[0], time_clip=True, level="l3")
fsm_B = data_quants["mms1_fsm_b_gse_brst_l3"].values
time_dist = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values
timeDelta = time_dist[1] - time_dist[0]

mag_fsm_B = np.linalg.norm(fsm_B, axis=1)
print(mag_fsm_B.shape)

meanv = np.load("src/magSpec/meanv.npy")
di = 50  # km
scale = timeDelta * meanv / di
steps = int(80 / scale)
print(steps)

corr = autocorrelate(mag_fsm_B, steps)
x = np.arange(len(corr)) * scale
plt.plot(x, corr)
plt.hlines(0, 0, 80, "k")

plt.show()