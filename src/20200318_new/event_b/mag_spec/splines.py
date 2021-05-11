import numpy as np
import os
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
import pandas as pd
from scipy import interpolate
import subprocess

override_mpl.override()

path = os.path.dirname(os.path.realpath(__file__))

x = np.load(f"{path}/total_k.npy")
instrument_mask = x < 10
x = x[instrument_mask]
y = np.load(f"{path}/total_y.npy")
y = y[instrument_mask]

# y *= x ** -2

x = np.log10(x)
y = np.log10(y)

xx = np.log10(np.logspace(x[0], x[-1], num=1000))
f = interpolate.interp1d(x, y)
yy = f(xx)

data = {"x": xx, "y": yy}
df = pd.DataFrame(data)
# df.to_csv(f"{path}/raw_r.csv", index=False)

# Run R code
subprocess.call(f"{path}/mars.r")

df2 = pd.read_csv(f"{path}/mars.csv")
YY = np.array(df2.y)

fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(x, y, alpha=0.5)
ax[0].plot(xx, yy, alpha=0.5)
ax[0].plot(xx, YY)

slopes = np.gradient(YY, np.diff(xx)[0])
ax[1].plot(xx, slopes)

slopes, slope_index, slope_counts = np.unique(
    np.round(slopes, 2),
    return_index=True,
    return_counts=True,
)

slope_counts = slope_counts > 2
slopes = slopes[slope_counts]
slope_index = 10 ** xx[slope_index[slope_counts]]

print(f"slopes: {slopes}, break_points: {slope_index}")

plt.show()