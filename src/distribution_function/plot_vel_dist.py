"""1st attempt at converting velocities. Deprecated.
"""
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants
from numba import jit


@jit(nopython=True)
def polar_to_GSE(E, theta, phi):
    """Converts MMS angles to energy components in GSE XYZ.
    phi=0 -> GSE +X
    theta = 0 -> GSE +Z

    equations:
    x = r sin(theta) cos(phi)
    y = r sin(theta) sin(phi)
    z = r cos(theta)
    """
    # Convert to radians
    theta = np.radians(theta)
    phi = np.radians(phi)

    x = E * np.sin(theta) * np.cos(phi)
    y = E * np.sin(theta) * np.sin(phi)
    z = E * np.cos(theta)

    return x, y, z


@jit(nopython=True)
def energy_to_velocity(vec, species="i"):
    """Converts energy components in GSE to velocity.

    E = gamma mc**2
    Let Er = m/c**2 <- Rest Mass energy
    => v = c * sqrt(1-1/((E/Er + 1)^2))
    """
    if species == "i":
        mass = 940.815e6  # In eV
    elif species == "e":
        mass = 0.5117094e6  # In eV
    else:
        print("Species " + species + "not recognised")
        return 0

    parity = 1 if vec >= 0 else -1
    c = 299792458  # in m/s
    return (c * np.sqrt(1 - 1.0 / ((np.abs(vec) / mass + 1) ** 2)) / 1000) * parity


if __name__ == "__main__":
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
        trange=trange,
        probe=probe,
        datatype="dis-dist",
        data_rate=data_rate,
        time_clip=True,
    )

    dist = data_quants["mms4_dis_dist_brst"].values
    dist_time = data_quants["mms4_dis_dist_brst"].coords["time"].values

    theta = np.linspace(0, 180, 16).astype(float)
    phi = np.arange(0, 360, 360.0 / 32).astype(float)

    energy_bin_starts = [
        1.29,
        3.04,
        4.79,
        9.32,
        12.48,
        15.92,
        21.08,
        27.13,
        35.67,
        46.15,
        60.43,
        78.4,
        102.7,
        132.7,
        173.9,
        225.7,
        294.7,
        382.9,
        498.7,
        649.6,
        846.0,
        1103.0,
        1433.0,
        1867.0,
        2432.0,
        3171.0,
        4126.0,
        5377.0,
        7000.0,
        9120.0,
        11910.0,
        15460.0,
        20140.0,
    ]
    energy_bin_starts = np.array(energy_bin_starts)

    vel_bins = np.array(list(map(energy_to_velocity, energy_bin_starts)))

    vel_bins = np.append(np.flip(vel_bins) * -1, vel_bins)
    vel_bins = np.insert(vel_bins, len(vel_bins) // 2, float(0))
    # vel_bins = np.column_stack([vel_bins] * 3)

    dist_size = np.shape(dist)

    points = []
    values = []

    test_time = 100
    data = np.mean(dist[:, :, :, :], axis=0)
    print(np.shape(data))
    for E in range(dist_size[1]):
        for el in range(dist_size[2]):
            for az in range(dist_size[3]):
                energy = energy_to_velocity(energy_bin_starts[E])
                vector = polar_to_GSE(energy, theta[el], phi[az])
                points.append(np.array(vector[:2]))
                values.append(data[E, el, az])

    vx, vy = np.meshgrid(vel_bins, vel_bins)
    points = np.array(points)
    values = np.array(values)
    grid = griddata(points, values, (vx, vy), method="cubic")

    print(vel_bins[0], vel_bins[-1], vel_bins[0], vel_bins[-1])

    plt.pcolormesh(vel_bins, vel_bins, grid)
    # values *= 1. / values.max()
    # values *= 1. / (values.mean() + values.std())
    # values[values > 1] = 1

    # print(np.shape(points[:, 0]))

    # plt.scatter(points[:, 0], points[:, 1], c=values, s=500)

    # data = np.mean(dist, axis=0)
    # data = np.mean(data, axis=1)
    # data *= 1.0e3 / data.max()
    # # data = scale(data, axis=0, with_mean=True, with_std=True)
    # plt.pcolormesh(phi, energy_bin_starts, data)

    # grid_bins = np.append(np.flip(energy_bin_starts) * -1, energy_bin_starts[:-1])
    # grid_bins = np.insert(grid_bins, len(grid_bins) // 2, float(0))
    # for i in range(len(grid_bins[:-1])):
    #     x = grid_bins[:-1] * np.cos(np.radians(45))
    #     y = grid_bins[i]
    #     r = np.sqrt(x**2 + y**2)
    #     p = np.degrees(np.arctan2(y, x) + np.pi)
    #     p2 = np.degrees(np.arctan2(x, y) + np.pi)
    #     plt.plot(p, r, c='k')
    #     plt.plot(p2, r, c='k')

    # plt.yscale('log')
    plt.rc({"figsize": (10, 10)})
    plt.show()
