"""First attempt at creating a magnetic spectra.
SRC:
10.3847/2041-8213/ab21c8
Properties of the Turbulence Associated with Electron-only Magnetic Reconnection in Earth's Magnetosheath
J. E. Stawarz et al.
"""
import math
from datetime import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import data_quants
from scipy.optimize import curve_fit


def shift_with_nan(xs, n):
    """Shift array elements in xs by n indices.
    SRC: https://stackoverflow.com/q/30399534/4465614
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


# Load mms data
print("Loading data.")
# Load data
trange = ["2016-12-09/09:03:00", "2016-12-09/09:10:00"]
probe = ["1", "2", "3", "4"]
# probe = "1"
data_rate = "brst"

mms_fpi = pyspedas.mms.fpi(
    trange=trange,
    probe=probe,
    # datatype="dis-dist",
    data_rate=data_rate,
    time_clip=True,
)
mms_fgm = pyspedas.mms.fgm(
    trange=trange, probe=probe, data_rate=data_rate, time_clip=True
)

# mms_scm = pyspedas.mms.scm(
#     trange=trange, probe=probe, data_rate=data_rate, time_clip=True
# )

# Get raw distribution data
get_b_str = lambda probe_num: f"mms{probe_num}_fgm_b_gse_{data_rate}_l2"
time_dist = data_quants[get_b_str(1)].coords["time"].values
b_data = {}
for probe_num in range(1, 5):
    b_string = get_b_str(probe_num)
    pyspedas.tinterpol(b_string, get_b_str(1), newname=b_string)
    data = data_quants[b_string].values[:, 3]
    finiteMask = np.isfinite(data)
    b_data[f"mms{probe_num}"] = np.interp(
        time_dist, time_dist[finiteMask], data[finiteMask]
    )
# Get time of each step
# Convert to datetime
time_dist = np.array([dt.utcfromtimestamp(x) for x in time_dist])

print(
    f"""Data loaded. {b_data['mms1'].shape[0]} time steps
Start:  {dt.strftime(time_dist[0], '%H:%M:%S.%f')}
End:    {dt.strftime(time_dist[-1], '%H:%M:%S.%f')}"""
)

print("Averaging |B| from mms1-4")
temp = np.empty((b_data["mms1"].shape[0], 4))
shifts = []
for i in range(1, 5):
    print(f"MMS{i}: Assigning data")
    temp[:, i - 1] = b_data[f"mms{i}"]
    if i > 1:
        print(f"MMS{i}: Correlating with MMS1")
        # See src/shock_normal/timing_analysis.py
        corr = np.correlate(
            temp[:, i - 1] - temp[:, i - 1].mean(),
            temp[:, 0] - temp[:, 0].mean(),
            mode="full",
        )
        shift = np.argmax(corr) - (len(temp[:, i - 1]) - 1)
        shifts.append(shift)
        # Shift SC data to same time point
        print(f"MMS{i}: Shifting {shift} indices")
        temp[:, i - 1] = shift_with_nan(temp[:, i - 1], shift)

avg_B = np.mean(temp, axis=1)
shifts = np.array(shifts)
# Generate slice to trim nan's created by aligning SC's
slice_B = slice(
    max(shifts[shifts > 0]) if len(shifts[shifts > 0]) > 0 else None,
    -1 + min(shifts[shifts < 0]) if len(shifts[shifts < 0]) > 0 else None,
)
avg_B = avg_B[slice_B]
time_dist = time_dist[slice_B]
print("Cleaning Memory")
del temp
del b_data
#       l=nu*Deltat+exp(2**x)
# plt.subplot(2, 1, 1)
# plt.plot(time_dist, avg_B)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
# plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))

# plt.subplot(2, 1, 2)
plt.subplot(2, 1, 1)
res = avg_B - np.mean(avg_B)
YHann = np.hanning(len(res)) * res
Y = np.fft.fft(YHann)
freq = np.fft.fftfreq(len(res), (time_dist[1] - time_dist[0]).total_seconds())

# Scale using Taylor Hypothesis
# Get average ion velocity
meanv = 0
for iprobe in probe:
    bulkv = data_quants[f"mms{iprobe}_dis_bulkv_gse_brst"].values
    bulkv = np.linalg.norm(bulkv, axis=1)
    meanv += bulkv.mean()
meanv /= len(probe)

ν_0 = meanv  # * np.sin(np.radians(47))
print(ν_0)
freq2 = 2 * np.pi * freq / ν_0

plt.loglog(freq2[freq > 0], abs(Y)[freq > 0])
plt.xlabel("f [Hz]")
plt.ylabel("Magnetic spectrum [nT²Hz⁻¹]")

plt.subplot(2, 1, 2)

freq2 = freq2[freq > 0]
data = abs(Y)[freq > 0]
data2 = data * freq2 ** (2.7)
plt.loglog(freq2, data2)
plt.xlabel("k[km$^{-1}$]")
plt.ylabel(r"Magnetic Spectrum $\times k^{2.7}$")

ρ_i = 180.0
ρ_e = 1.0

x = np.array([1.0 / ρ_i, 1.0 / ρ_e])
# indX = [find_nearest(freq2, x[0]), find_nearest(freq2, x[1])]
indX = np.searchsorted(freq2, x, side="left")
print(indX)
y = np.array([data[indX[0]], data[indX[1]]])
print(x, y)
print(np.diff(np.log(y)) / np.diff(np.log(x)))
plt.loglog(x, y * x ** 2.7)


plt.show()
