"""Main script. Animates velocity distribution in GSE
coordinates, as observed by MMS in spherical energy coords.

Looking at a different time period should be as easy as 
changing the trange parameter in the main script. (untested)
"""
# Default imports
import glob
import os
from datetime import datetime as dt

# 3rd Party
import numpy as np
import pyspedas
import pyvista
import vtk
from pytplot import data_quants
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import scale

# Local
import interpol
import plot_vel_dist as pvd
import plotlyApplied as pA
from rotate import rot_to_b


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
                velocity = pvd.energy_to_velocity(e_bins[E])
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


def generate_frame(bins, e_bins, theta, phi, i=0, frange=None):
    """Ties together all steps to convert SC data into vel dist in XYZ.

    IN:
        bins:   np.array(n)
                    - New velocity bins across one axis
                    - Generates square grid from bins[0] -> bins[-1]
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

    vx, vy, vz = np.meshgrid(bins, bins, bins)
    # coordinates of each point on output
    uvw = np.array([vx.flatten(), vy.flatten(), vz.flatten()]).T

    # Generate input points & values
    points, values = convert_space(dist, e_bins, theta, phi)

    cwd = os.getcwd()
    # Check if running from inside distribution_function or project dir
    if cwd.split("/")[-1] != "distribution_function":
        cwd += "/distribution_function"
    # Check if vertices and weights have already been generated
    if cwd + "/vtx.npy" and cwd + "/wts.npy" in glob.glob(cwd + "/*.npy"):
        print(f"IT{i:02d}: Saved vtx and wts found. Loading...")
        vtx = np.load(cwd + "/vtx.npy")
        wts = np.load(cwd + "/wts.npy")
    else:
        print(f"IT{i:02d}: No saved vtx &/or wts found. Generating...")
        vtx, wts = interpol.interp_weights(points, uvw)
        print(f"IT{i:02d}: Generated vtx & wts. Saving...")
        np.save(cwd + "/vtx.npy", vtx)
        np.save(cwd + "/wts.npy", wts)

    print(f"IT{i:02d}: Interpolating values onto grid...")
    grid = interpol.interpolate(values, vtx, wts, fill_value=0.0).reshape(vx.shape)

    print(f"IT{i:02d}: Applying Gaussian smoothing filter...")
    # Scale data in range [0,1). get max dynamic range by accounting
    # for mean and std
    grid = gaussian_filter(grid, sigma=1.2, mode="constant")

    print(f"IT{i:02d}: Rescaling Grid...")
    grid = grid.flatten()
    grid = scale(grid, axis=0, with_mean=True, with_std=True)
    grid = np.reshape(grid, vx.shape)

    print(f"IT{i:02d}: Done.")
    return grid


def get_e_bins(loc="centre"):
    """Get an array of energy bins.

    IN:
        loc:    str
                    - 'centre': Returns bin centres.

    OUT:
        bins:   np.array(32)
                    - Energy bins
    """
    if loc == "centre":
        bins = np.array(
            [
                2.16,
                3.91,
                7.07,
                10.9,
                14.2,
                18.5,
                24.1,
                31.4,
                40.9,
                53.3,
                69.4,
                90.4,
                118.0,
                153.0,
                200.0,
                260.0,
                339.0,
                441.0,
                574.0,
                748.0,
                974.0,
                1270.0,
                1650.0,
                2150.0,
                2800.0,
                3650.0,
                4750.0,
                6190.0,
                8060.0,
                10500.0,
                13700.0,
                17800.0,
            ]
        )
    else:
        print("loc not recognised: Please choose from ['left', 'centre', 'right']")
        print("Note 17/11/20 only centre implemented.")
        bins = 0

    return bins


if __name__ == "__main__":
    print("Loading data.")
    # Load data
    trange = ["2015-10-07/11:44:34", "2015-10-07/11:44:49"]
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

    # Get raw distribution data
    raw_dist = data_quants["mms4_dis_dist_brst"].values
    # Get time of each step
    time_dist = data_quants["mms4_dis_dist_brst"].coords["time"].values
    # Convert to datetime
    time_dist = np.array([dt.utcfromtimestamp(x) for x in time_dist])

    print(
        f"""Data loaded. {raw_dist.shape[0]} time steps
    Start:  {dt.strftime(time_dist[0], '%H:%M:%S.%f')}
    End:    {dt.strftime(time_dist[-1], '%H:%M:%S.%f')}"""
    )

    # Bin centres for energy
    e_bins = get_e_bins()
    # Elevation angles, theta
    theta = np.linspace(0, 180, 16)
    # Azimuthal angles phi
    phi = np.arange(0, 360, 11.25)

    max_v = pvd.energy_to_velocity(e_bins[-1])
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
        return np.swapaxes(generate_frame(v_bins, e_bins, theta, phi, i=i).T, 0, 1)

    # Output file name
    fname = "animate.mp4"
    # Min isosurface, max ..., and num of surfaces
    isomin, isomax, isosurf = 0, 1, 10
    # Initalise plot with first frame
    meshData = gnfrm(0)
    # Generate pyvista object from np array
    mesh = pyvista.wrap(meshData)
    # Modify assumed dimensions (should do nothing)
    mesh.dimensions = meshData.shape
    # pyvista origin is min value on each axis (bottom left)
    mesh.origin = (v_bins[0], v_bins[0], v_bins[0])
    # Coordinates determined by origin+spacing
    mesh.spacing = tuple([np.diff(v_bins)[0]] * 3)

    # Generate isosurfaces from distribution
    contour = mesh.contour(np.linspace(isomin, isomax, isosurf))

    # White BG, bigger labels etc...
    pyvista.set_plot_theme("document")
    # Create 2x2 plot
    plotter = pyvista.Plotter(shape=(2, 2))
    # Tell pyvista to use imageio-ffmpeg to generate movie
    # Frame rate = inverse of time step
    plotter.open_movie(
        fname, framerate=1.0 / (time_dist[1] - time_dist[0]).total_seconds()
    )

    def plot_init():
        """Commands that need to be run to initialise each subplot."""
        plotter.show_grid()
        plotter.add_axes()
        # Uncomment below to change method of rendering opacity
        # plotter.enable_depth_peeling()

        # Add contour mesh to plot
        # cmap keywords requires colorcet (pip install colorcet)
        plotter.add_mesh(contour, opacity="linear", cmap="fire")
        # Add bounding box
        plotter.add_mesh(mesh.outline_corners())

    # Camera position
    # [[pos_x, pos_y, pos_z], [look_at_x, ..y, ..z], [up_direction_x, ..y, ..z]]
    cam = np.array(
        [
            [7125.103749987386, 7125.103749987386, 7125.103749987386],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    replace_arr = np.ones(cam.shape)  # helps move camera to all 4 top-corners
    plot_init()
    plotter.camera_position = cam * replace_arr
    plotter.subplot(0, 1)
    # Add the time
    timeText = plotter.add_text(time_dist[0].strftime(r"%Y/%m/%d-%H:%M:%S.%f"))
    # Move camera to -y
    replace_arr[0, 1] = -1
    plotter.camera_position = cam * replace_arr
    plot_init()
    # Move cam to +y
    replace_arr[0, 1] = 1
    # Move cam to -x
    replace_arr[0, 0] = -1
    plotter.subplot(1, 0)
    plotter.camera_position = cam * replace_arr
    plot_init()
    # Move cam to -y
    replace_arr[0, 1] = -1
    plotter.subplot(1, 1)
    plotter.camera_position = cam * replace_arr
    plot_init()

    # Opens an interactive window. Fine-tune camera angles before generating movie
    # Controls: https://docs.pyvista.org/plotting/plotting.html

    print('Orient the view, then press "q" to close window and produce movie')
    camera = plotter.show(auto_close=False)
    print(camera)
    plotter.write_frame()

    frames = raw_dist.shape[0]
    for i in range(1, frames):
        # Replace mesh values with new ones from next frame
        mesh.point_arrays["values"] = gnfrm(i).flatten(order="F")
        # Overwrite contours by regnerating with new data
        contour.overwrite(mesh.contour(np.linspace(isomin, isomax, isosurf)))
        plotter.subplot(0, 0)
        plotter.update()
        plotter.subplot(0, 1)
        # Update time
        timeText.ClearAllTexts()
        timeText.SetText(
            vtk.vtkCornerAnnotation.UpperLeft,
            time_dist[i].strftime(r"%Y/%m/%d-%H:%M:%S.%f"),
        )
        plotter.update()
        plotter.subplot(1, 0)
        plotter.update()
        plotter.subplot(1, 1)
        plotter.update()
        plotter.write_frame()

    # Check output against original (using plotly) method by
    # commenting out all above animation code and uncommenting
    # below
    # data = generate_frame(v_bins, e_bins, theta, phi, i=0).T
    # pA.plotVol(data, v_bins)
