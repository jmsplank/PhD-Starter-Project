import pyspedas
from pytplot import data_quants
import numpy as np
from trange import trange as tr


def meanv(data):
    trange = tr(data)
    return _calc(trange)


def _calc(trange):
    probe = "1"
    data_rate = "brst"
    level = "l2"

    pyspedas.mms.fpi(
        trange=trange,
        probe=probe,
        data_rate=data_rate,
        level=level,
    )

    v = data_quants["mms1_dis_bulkv_gse_brst"].values

    meanv = np.linalg.norm(v, axis=1).mean()
    return meanv


if __name__ == "__main__":
    # Load Search coil
    trange = ["2020-03-18/02:25:30", "2020-03-18/02:44:00"]
    meanv = _calc(trange)

    np.save("src/Event_2/stats/meanv.npy", meanv)
    print(f"meanv: {meanv:03.3f} km/s")
