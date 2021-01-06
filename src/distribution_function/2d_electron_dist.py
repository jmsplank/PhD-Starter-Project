"""Main script. Animates velocity distribution in GSE
coordinates, as observed by MMS in spherical energy coords.

Run as python animate_velocity.py -h to see the help on running the script.
"""
print(
    f"""   _____ __             __ 
  / ___// /_____ ______/ /_
  \__ \/ __/ __ `/ ___/ __/
 ___/ / /_/ /_/ / /  / /_  
/____/\__/\__,_/_/   \__/  
"""
)

print("Loading Imports")
# Default imports
import argparse as ap
import glob
import os
from datetime import datetime as dt

# 3rd Party
import numpy as np
import pyspedas
from tqdm import trange as tqdmRange

# import pyvista
# import vtk
from pytplot import data_quants
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# Local
import interpol
import plot_vel_dist as pvd

# import plotlyApplied as pA
from rotate import rot_to_b
from get_e_bins import get_e_bins


def convert_space(dist, e_bins, theta, phi, rotate=None):
    """Maps distribution in spherical coordinates to cartesian.

    IN:
        dist:   np.array(32,16,32)
                    - The ion energy distribution
        e_bins: np.array(32)
                    - Energy bins (centres)
        theta:  np.array(16)
                    - Elevation angle (0, 180) inclusive
        phi:    np.array(32)
                    - Azimuthal angle (0, 360]
        rotate: np.array(3)
                    - (x,y,z) coords of vector pointing in
                        to rotate +Z

    OUT:
        points: np.array(32*16*32, 3)
                    - Coordinates of each value
        values: np.array(32*16*32)
                    - Values of distribution
    """
    # Shape of distribution (time, energy, theta, phi) => (t, 32, 16, 32)
    ds = np.shape(dist)
    # print(e_bins.shape, theta.shape, phi.shape)

    points = np.zeros((ds[0] * ds[1] * ds[2], 3), dtype=np.float64)
    values = np.zeros(ds[0] * ds[1] * ds[2], dtype=np.float64)
    i = 0
    # loop over energies
    for E in range(ds[0]):
        # Loop over elevation angles
        for el in range(ds[1]):
            # Loop over azimuthal angles
            for az in range(ds[2]):
                # Get velocity from energy
                velocity = pvd.energy_to_velocity(e_bins[E], species="e")
                # transform polar vel to cartesian GSE
                vector = pvd.polar_to_GSE(velocity, theta[el], phi[az])
                # Rotate data so +x points to B parallel
                if rotate is not None:
                    vector = rot_to_b(rotate, vector)
                # Add coordinate to list
                points[i] = np.array(vector)
                # Add coordinate value to list
                values[i] = dist[E, el, az]
                i += 1

    return points, values


def generate_frame(raw_dist, new_bins, e_bins, theta, phi, i=0, frange=None):
    """Ties together all steps to convert SC data into vel dist in XYZ.

    IN:
        raw_dist:   np.array(time, e_bins, theta, phi)
                    - The raw, unprocessed data from SC
        new_bins:   np.array(n)
                    - New velocity bins across one axis
                    - Generates cubic grid from bins[0] -> bins[-1]
                    - Total number of bins in output: n**3
        e_bins: np.array(32)
                    - Energy bins
        theta:  np.array(16)
                    - Elevation angle (0, 180) inclusive
        phi:    np.array(32)
                    - Azimuthal angle (0, 360]
        i:      int
                    - Default 0
                    - Index of time step to calculate transformation
        frange: array_like[first,last] OR str
                    - Default None
                    - Average together all frames from first:last, by index
                    - Accepts kw 'all'. Averages all frames

    OUT:
        grid:   np.array(n,n,n)
                    - 3D grid of distribution, corresponding to coordinates from bins
    """
    if frange is not None:
        if frange == "all":
            dist = np.mean(raw_dist, axis=0)
        else:
            dist = np.mean(raw_dist[frange[0] : frange[1], :, :, :], axis=0)
    else:
        if i == 0:
            dist = np.mean(raw_dist[:3, :, :, :], axis=0)
        elif i == raw_dist.shape[0] - 1:
            dist = np.mean(raw_dist[-3:, :, :, :], axis=0)
        else:
            dist = np.mean(raw_dist[i - 1 : i + 2, :, :, :], axis=0)

    vx, vy, vz = np.meshgrid(new_bins, new_bins, new_bins)
    # coordinates of each point on output
    uvw = np.array([vx.flatten(), vy.flatten(), vz.flatten()]).T

    # Generate input points & values
    points, values = convert_space(dist, e_bins, theta, phi)

    cwd = os.getcwd()
    # Check if running from inside distribution_function or project dir
    if cwd.split("/")[-1] != "distribution_function":
        cwd += "/src/distribution_function"
    # Check if vertices and weights have already been generated
    if cwd + "/vtx_e.npy" and cwd + "/wts_e.npy" in glob.glob(cwd + "/*.npy"):
        # print(f"IT{i:02d}: Saved vtx_e and wts_e found. Loading...")
        vtx = np.load(cwd + "/vtx_e.npy")
        wts = np.load(cwd + "/wts_e.npy")
    else:
        # print(f"IT{i:02d}: No saved vtx_e &/or wts_e found. Generating...")
        vtx, wts = interpol.interp_weights(points, uvw)
        # print(f"IT{i:02d}: Generated vtx_e & wts_e. Saving...")
        np.save(cwd + "/vtx_e.npy", vtx)
        np.save(cwd + "/wts_e.npy", wts)

    # print(f"IT{i:02d}: Interpolating values onto grid...")
    grid = interpol.interpolate(values, vtx, wts, fill_value=0.0).reshape(vx.shape)

    # print(f"IT{i:02d}: Applying Gaussian smoothing filter...")
    # for mean and std
    grid = gaussian_filter(grid, sigma=1.2, mode="constant")

    # Scale data in range [0,1). get max dynamic range by accounting
    # print(f"IT{i:02d}: Rescaling Grid...")
    grid = grid.flatten()
    grid = scale(grid, axis=0, with_mean=True, with_std=True)
    grid = np.reshape(grid, vx.shape)
    # print(f"IT{i:02d}: Done.")
    return grid


def inputs():
    # start_time, end_time, date=None, probe=4
    parser = ap.ArgumentParser()
    parser.add_argument(
        "start_time",
        type=str,
        help="The start time. Can be formatted as YYYY-MM-DD/HH:MM:SS or HH:MM:SS with -d keyword",
    )
    parser.add_argument(
        "end_time",
        type=str,
        help="The end time. Can be formatted as YYYY-MM-DD/HH:MM:SS or HH:MM:SS with -d keyword",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="The output file to save the animation in. use .mp4 format. default is ./vid_YYYYMMDDHHMMSS_LENGTHs.mp4",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        help="(optional) date if start_time/end_time contains only times, formatted as YYYY-MM-DD",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--probe",
        type=int,
        help="(optional) MMS probe to use [1-4]",
        default=4,
        choices=[1, 2, 3, 4],
    )
    args = parser.parse_args()
    if isinstance(args.start_time, str) and isinstance(args.end_time, str):
        # Is correct type
        if args.date is not None:
            try:
                args.start_time = f"{args.date}/{args.start_time}"
                args.end_time = f"{args.date}/{args.end_time}"
            except (TypeError, AttributeError) as e:
                print("date needs to be a string.", e)
            # Try parsing them as dates.
            # If this works then they are the correct format
        try:
            st = dt.strptime(args.start_time, "%Y-%m-%d/%H:%M:%S")
            et = dt.strptime(args.end_time, "%Y-%m-%d/%H:%M:%S")
        except Exception as e:
            print(
                """date should be formatted as YYYY-MM-DD
start_time and end_time should be formatted as:
with date: YYYY-MM-DD/HH:MM:SS
without date (Note: Must supply -d keyword): HH:MM:SS""",
                e,
            )
    else:
        raise TypeError("Both start_time and end_time need to be of type str.")

    if args.output == "":
        args.output = f"vid_{dt.strftime(st, '%Y%m%d%H%M%S')}_{str(int((et-st).total_seconds()))}s.mp4"
    if ".mp4" not in args.output:
        args.output += ".mp4"

    return [args.start_time, args.end_time], args.probe, args.output


if __name__ == "__main__":
    # Get CL args
    print("Processing input args")
    trange, probe, fname = inputs()
    # trange = ["2015-10-07/11:44:34", "2015-10-07/11:44:49"]
    # probe = ["1", "2", "3", "4"]
    # probe = "4"
    data_rate = "brst"

    # Load data
    print("Loading data.")
    mms_fpi = pyspedas.mms.fpi(
        trange=trange,
        probe=probe,
        datatype="des-dist",
        data_rate=data_rate,
        time_clip=True,
    )

    # Get raw distribution data
    raw_dist = data_quants["mms4_des_dist_brst"].values
    # Get time of each step
    time_dist = data_quants["mms4_des_dist_brst"].coords["time"].values
    # Convert to datetime
    time_dist = np.array([dt.utcfromtimestamp(x) for x in time_dist])

    print(
        f"""Data loaded. {raw_dist.shape[0]} time steps
    Start:  {dt.strftime(time_dist[0], '%H:%M:%S.%f')}
    End:    {dt.strftime(time_dist[-1], '%H:%M:%S.%f')}"""
    )

    # Bin centres for energy
    e_bins = get_e_bins(species="e")
    # Elevation angles, theta
    theta = np.linspace(0, 180, 16)
    # Azimuthal angles phi
    phi = np.arange(0, 360, 11.25)

    max_v = pvd.energy_to_velocity(e_bins[-1], species="e")
    # Generate velocity bins
    # Goes from -max energy -> +max energy
    # Number of bins determines size of ouput array
    # Pretty sure it's O(n^3) so be careful
    v_bins = np.linspace(-max_v, max_v, 64)

    # ============== ANIMATE =============

    def gnfrm(i):
        """Helper to avoid typing too much.
        Abbrev. of generate_frame. Reformats into shape expected
        by pyvista's routines.
        """
        return np.swapaxes(
            generate_frame(raw_dist, v_bins, e_bins, theta, phi, i=i).T, 0, 1
        ).mean(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    X, Y = np.meshgrid(v_bins, v_bins)
    for i in tqdmRange(len(time_dist)):
        plt.contourf(X, Y, gnfrm(i), levels=np.arange(-1, 5, 0.2))
        if i == 0:
            plt.colorbar()
        plt.savefig(f"FRAMES_2d_electron_dist/frame{i}.png")
