import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from datetime import datetime as dt

override_mpl.override("", "inter", serif=False)

pp = new_path(get_path(__file__, ".."))
p = new_path(get_path(__file__))

B = np.load(pp("data/fgm/data.npy"))[::250]
B_time = np.load(pp("data/fgm/time.npy"))[::250]

time = np.load(pp("data/fsm/time.npy"))

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(B_time, B)
shock_time = 1584500551
ax.axvspan(B_time[0], shock_time, fc="k", alpha=0.15, label="STR", ec="k")
ax.axvspan(shock_time, B_time[-1], fc="white", label="SW", ec="k", alpha=0.15)
ax.set_xlim((B_time[0], B_time[-1]))

firstplot, secondplot = 2488567, 3705200
fts = lambda x: dt.strftime(dt.utcfromtimestamp(x), "%H:%M:%S")
ax.axvline(time[firstplot], color="k", ls="--", label=fts(time[firstplot]))
ax.axvline(time[secondplot], color="k", ls="-.", label=fts(time[secondplot]))
ax.legend()

ax.set_ylabel("B [nT]")
ax.set_xlabel("Time UTC 2020/03/18 (hh:mm:ss)")
ax.set_xticklabels(map(fts, ax.get_xticks()))

plt.tight_layout()
plt.savefig(p("one.png"), dpi=300)
