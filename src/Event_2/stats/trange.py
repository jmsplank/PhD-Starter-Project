from datetime import datetime as dt


# trange = ["2020-03-18/02:25:30", "2020-03-18/02:44:00"]
def trange(data):
    start = dt.utcfromtimestamp(trange[0])
    stop = dt.utcfromtimestamp(trange[-1])

    start = start.strftime("%Y-%m-%d/%H:%M:%S")
    stop = stop.strftime("%Y-%m-%d/%H:%M:%S")
    return [start, stop]
