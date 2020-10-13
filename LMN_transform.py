import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from pytplot import tplot, data_quants
import datetime as dt
from operator import itemgetter

trange = ["2017-01-26/08:14:59.5", "2017-01-26/08:15:01.25"]
probe = "1"
data_rate = "brst"
fgm_all = pyspedas.mms.fgm(
    trange=trange, probe=["1", "2", "3", "4"], data_rate=data_rate, time_clip=True
)

Bxyz_indiv = []
for probe in range(1, 5):
    Bxyz_indiv.append(data_quants[f"mms{probe}_fgm_b_gse_brst_l2"].values[:, :3])

# Average over all 4 probes
Bxyz = np.mean(Bxyz_indiv, axis=0)

# Get time values
Bxyz_time = data_quants["mms1_fgm_b_gse_brst_l2"].coords["time"].values
# Convert time to datetime objects (for matplotlib)
time = [dt.datetime.utcfromtimestamp(i) for i in Bxyz_time]


M = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        M[i, j] = np.mean(Bxyz[:, i] * Bxyz[:, j]) - np.mean(Bxyz[:, i]) * np.mean(
            Bxyz[:, j]
        )
print(M)

bm = np.transpose(M)  # Does nothing?
w, v = np.linalg.eig(bm)  # Get Eigenvectors & eigenvalues


N = min(enumerate(w), key=itemgetter(1))[0]  # Find minimum eig.value
L = max(enumerate(w), key=itemgetter(1))[0]  # Find max eig.value
M = np.setdiff1d(range(3), [N, L])[0]  # Find the one that isn't min or max

Blmn = np.matmul(v.T, Bxyz.T)  # Transform to LMN coords
B_LMN = np.column_stack(
    [Blmn[L], Blmn[M], Blmn[N]]
)  # Reshape to (time, LMN) with ordered LMN


# Plot x,y,z
plt.plot(time, Bxyz[:, 0], label="x", alpha=0.2, color="r", ls="--")
plt.plot(time, Bxyz[:, 1], label="y", alpha=0.2, color="g", ls="--")
plt.plot(time, Bxyz[:, 2], label="z", alpha=0.2, color="b", ls="--")

# Plot L,M,N
plt.plot(time, B_LMN[:, 0], label="L", color="r")
plt.plot(time, B_LMN[:, 1], label="M", color="g")
plt.plot(time, B_LMN[:, 2], label="N", color="b")

plt.legend()
plt.show()
