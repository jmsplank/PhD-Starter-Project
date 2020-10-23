import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from pytplot import tplot, data_quants
import datetime as dt
from operator import itemgetter


def gse_to_lmn(param: np.ndarray) -> np.ndarray:
    """Transforms data from GSE coordinates to LMN using minimum variance analysis.

    IN:
        param       :   np.ndarray
                    :   Shape (time, xyz)
    OUT:
        param_LMN   :   np.ndarray
                    :   Shape (time, LMN)
    """

    M = np.zeros((3, 3))  # Initalise magnetic variance martix
    for i in range(3):
        for j in range(3):
            # Populate with <B_i*B_j> - <B_i>*<B_j>
            M[i, j] = np.mean(param[:, i] * param[:, j]) - np.mean(
                param[:, i]
            ) * np.mean(param[:, j])

    eig_w, eig_v = np.linalg.eig(M)  # Obtain eig.values & eig.vectors
    N = min(enumerate(eig_w), key=itemgetter(1))[0]  # N = min variance = min eig.value
    L = max(enumerate(eig_w), key=itemgetter(1))[0]  # L = max variance = max eig.value
    M = np.setdiff1d(range(3), [L, N])[0]  # M = intermediate variance = not N or L

    param_lmn = np.matmul(eig_v.T, param.T)  # Rotate coords of param into LMN frame
    param_LMN = np.column_stack(
        [param_lmn[L], param_lmn[M], param_lmn[N]]
    )  # Stack components in LMN order

    return eig_v, param_LMN


if __name__ == "__main__":
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

    B_LMN = gse_to_lmn(Bxyz)[1]

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
