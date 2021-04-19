import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

data = np.load(
    "/Users/jamesplank/Documents/PHD/PhD-Starter-Project/src/20200318_new/event_b/data/fsm/data.npy"
)
time = np.load(
    "/Users/jamesplank/Documents/PHD/PhD-Starter-Project/src/20200318_new/event_b/data/fsm/time.npy"
)
time2 = [dt.utcfromtimestamp(t) for t in time[:: len(time) // 1000]]
plt.plot(time2, data[:: len(time) // 1000])
plt.show()