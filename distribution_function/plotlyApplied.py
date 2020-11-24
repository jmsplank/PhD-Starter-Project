"""Helper functions for using plotly to create
html based plots of distribution function as either
volume-based isosurfaces (plotVol) or as an animated 
step-through of z-axis, showing each x-y plane as
an image (plotSlices)
"""
import plotly.graph_objects as go
import numpy as np


def plotVol(data, drange):
    """
    data:   np.array(n,n,n)
                3D grid. Can be unequally spaced.
    drange: np.array(d)
                axis coordinates. Can be unequally spaced.
    """
    X, Y, Z = np.meshgrid(drange, drange, drange)
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,
            surface_count=17,
        )
    )
    fig.show()


def plotSlices(data):
    """
    data:   np.array(n,n,n)
                3D grid. Can be unequally spaced.
    """
    ds = data.shape
    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=((ds[0] - 1) / 10 - k * 0.1) * np.ones((ds[1], ds[2])),
                    surfacecolor=np.flipud(data[ds[0] - 1 - k]),
                    cmin=0,
                    cmax=1,
                ),
                name=str(k),
            )
            for k in range(ds[0])
        ]
    )

    fig.add_trace(
        go.Surface(
            z=(ds[0] - 1) / 10 * np.ones((ds[1], ds[2])),
            surfacecolor=np.flipud(data[ds[0] - 1]),
            colorscale="Gray",
            cmin=0,
            cmax=1,
            colorbar=dict(thickness=20, ticklen=4),
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title="Slices in volumetric data",
        # width=600,
        # height=600,
        scene=dict(
            zaxis=dict(range=[-0.1, ds[0] / 10 + 0.1], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.show()
