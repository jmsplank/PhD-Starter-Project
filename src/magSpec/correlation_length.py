import time
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants
from scipy.stats import pearsonr
import re
from phdhelper.math.signal_processing import autocorrelate

from magSpec import interp_correction

# trange = ["2016-12-09/09:01:36", "2016-12-09/09:07:00"]  # Interval 1
trange = ["2016-12-09/09:26:24", "2016-12-09/09:34:58"]  # Interval 2
probe = ["1", "2", "3", "4"]
# probe = "1"
data_rate = "brst"

# mms_fsm = pyspedas.mms.fsm(trange=trange, probe=probe, time_clip=True, level="l3")
fgm_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_brst_l2"
pyspedas.mms.fgm(trange=trange, probe=probe, data_rate="brst")
time_dist = data_quants[fgm_str(probe[0])].coords["time"].values
timeDelta = time_dist[1] - time_dist[0]


def autocorr(x):
    res = np.correlate(x, x, mode="full")
    return res[res.size // 2 :]


fgm_B = interp_correction(fgm_str(probe[0]), fgm_str(probe[0]), time_dist)
# fgm_B = data_quants[fgm_str(probe[0])].values
corr = np.empty((5000 // 2, 4))
for i in range(4):
    if i != 1:
        fgm_B = interp_correction(fgm_str(probe[0]), fgm_str(probe[i]), time_dist)
        # fgm_B = data_quants[fgm_str(probe[i])].values[:, :3]
    mag = (np.linalg.norm((fgm_B - fgm_B.mean(axis=0)), axis=1) ** 2).mean()
    corr[:, i] = np.array([autocorrelate(fgm_B[:, x], 5000) for x in range(3)]).mean(
        axis=0
    )
corr = corr.mean(axis=1)
np.save(
    "src/magSpec/autocorrelation_{}.npy".format(re.sub(r"[^\w]", "", trange[0])), corr
)

meanv = np.load("src/magSpec/meanv.npy")
# print(meanv)
# print(timeDelta)
print(corr)
x = np.arange(corr.shape[0]) * timeDelta * meanv / 50

plt.plot(x, corr)
plt.hlines(0, 0, 80, "k")
plt.xlim((0, 80))

plt.savefig(
    f'src/magSpec/img/correlationLength_{dt.strftime(dt.now(), "%H%M%S_%a%d%b")}.png'
)
plt.show()
