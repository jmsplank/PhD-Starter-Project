import pyspedas
from pytplot import data_quants
import numpy as np
from phdhelper.helpers.CONSTANTS import c, m_i, q, epsilon_0
from trange import trange as tr


def d_i(data):
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

    number_density = data_quants["mms1_dis_numberdensity_brst"].values
    number_density = number_density.mean()
    number_density *= 1e6

    plasma_freq = np.sqrt((number_density * q ** 2) / (m_i * epsilon_0))

    d_i = c / plasma_freq
    d_i /= 1e3

    return d_i


if __name__ == "__main__":
    # Load Search coil
    trange = ["2020-03-18/02:48:00", "2020-03-18/03:09:00"]
    d_i = _calc(trange=trange)
    np.save("src/Event_2/stats/d_i.npy", d_i)
    print(f"d_i | ion inertial length: {d_i:03.3f} km")