# import matplotlib.pyplot as plt
from datetime import datetime as dt

import numpy as np
import pyspedas
from pytplot import data_quants

# Load data
trange = ["2015-10-07/11:44:35", "2015-10-07/11:44:45"]
probe = ["1", "2", "3", "4"]
data_rate = "brst"
mms_fgm = pyspedas.mms.fgm(trange=trange,
                           probe=probe,
                           data_rate=data_rate,
                           time_clip=True)

# Interpolate b & r to match mms1_b
for i in range(1, 5):
    pyspedas.tinterpol(f"mms{i}_fgm_b_gse_brst_l2", "mms1_fgm_b_gse_brst_l2")
    pyspedas.tinterpol(f"mms{i}_fgm_r_gse_brst_l2", "mms1_fgm_b_gse_brst_l2")

# Get magnitudes for b_mms1-4
# Get position coords for r_mms1-4
data_b = []
data_r = []
for i in range(1, 5):
    data_b.append(data_quants[f"mms{i}_fgm_b_gse_brst_l2-itrp"].values[:, -1])
    data_r.append(data_quants[f"mms{i}_fgm_r_gse_brst_l2-itrp"].values[:, :3])

# Get cross correlation of btot using mms4 as reference
correls = []  # [chi_14, chi_24, chi_34]
for i in range(3):
    correlation = np.correlate(
        data_b[i] - np.mean(data_b[i]),
        data_b[3] - np.mean(data_b[3]),
        mode="full")  # need to center the data -> subtract mean
    # Peak correlation index from argmax, subtract len()-1 to
    # offset for 'full' mode
    correls.append(np.argmax(correlation) - (len(data_b[i]) - 1))
print(correls)

# choose a point as reference position
INDEX = 670
# There's a peak at index 670 in mms4, so use that??
positions = []
for i in range(3):
    offset = INDEX + correls[i]
    pos_a = data_r[i][offset, :]
    pos_b = data_r[3][INDEX, :]
    positions.append(pos_a - pos_b)

# list of relative positions
# [[r14x,r14y,r14z],[r24x,r24y,r24z],[r34x,r34y,r34z]]
rel_positions = np.array(positions)

# find relative times
times = data_quants["mms4_fgm_b_gse_brst_l2-itrp"].coords["time"].values
rel_times = np.array([times[np.array(correls) + INDEX]]) - times[INDEX]
T = rel_times.T

# Solve Equation D^{-1}T = m
D_inv = np.linalg.inv(rel_positions)
m = np.matmul(D_inv, T)

norm = np.linalg.norm(m)
nhat = m / norm
V = 1.0 / norm

print("n unit vector:\n", nhat)
print(f"Speed: {V:0.2f}")

time_index = dt.utcfromtimestamp(times[INDEX])
print(f'Time: {time_index.strftime("%d %b %y @ %H:%M:%S.%f")}')
