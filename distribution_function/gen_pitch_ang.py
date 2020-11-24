# import os
from datetime import datetime as dt

# import cpickle as pickle
import numpy as np
import pyspedas
from pytplot import data_quants
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

import animate_velocity as pa
import plot_vel_dist as pvd


if __name__ == "__main__":
    print("Loading data...")
    # Load data
    trange = ["2015-10-07/11:44:41", "2015-10-07/11:44:50"]
    # probe = ["1", "2", "3", "4"]
    probe = "4"
    data_rate = "brst"

    mms_fpi = pyspedas.mms.fpi(
        trange=trange,
        probe=probe,
        datatype="dis-dist",
        data_rate=data_rate,
        time_clip=True,
    )

    mms_fgm = pyspedas.mms.fgm(
        trange=trange, probe=probe, data_rate=data_rate, time_clip=True
    )

    raw_dist = data_quants["mms4_dis_dist_brst"].values
    time_dist = data_quants["mms4_dis_dist_brst"].coords["time"].values
    time_dist = np.array([dt.utcfromtimestamp(x) for x in time_dist])
    B = data_quants["mms4_fgm_b_gse_brst_l2"].values[:, :3]
    B = np.mean(B, axis=0)

    print(
        f"""Data loaded. {raw_dist.shape[0]} time steps
    Start:  {dt.strftime(time_dist[0], '%H:%M:%S.%f')}
    End:    {dt.strftime(time_dist[-1], '%H:%M:%S.%f')}"""
    )

    e_bins = pa.get_e_bins()
    phi = np.arange(0, 360, 11.25)
    theta = np.linspace(0, 180, 16)

    v_bins = np.array(list(map(pvd.energy_to_velocity, e_bins)))
    v_bins = np.append(np.flip(v_bins) * -1, v_bins)
    v_bins = np.insert(v_bins, len(v_bins) // 2, float(0))

    x, y, z = np.meshgrid(v_bins, v_bins, v_bins)
    xyz = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    dist = np.mean(raw_dist, axis=0)
    print("Getting points and values")
    points, values = pa.convert_space(dist, e_bins, theta, phi, rotate=B)

    print("Triangulating")
    tri = Delaunay(points)
    print("Creating interpolation function")
    itrp = LinearNDInterpolator(tri, values)
    print("Interpolating")
    grid = itrp(xyz).reshape([len(v_bins)] * 3)

    print(grid.shape)
