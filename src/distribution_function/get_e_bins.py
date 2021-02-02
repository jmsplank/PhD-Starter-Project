import numpy as np


def get_e_bins(loc="centre", species="i"):
    """Get an array of MMS energy bins.

    IN:
        loc:        str
                    - 'centre': Returns bin centres.
        species:    str
                    - 'i' returns ion energies
                    - 'e' returns electron energies

    OUT:
        bins:   np.array(32)
                    - Energy bins
    """
    if species == "i":
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
    elif species == "e":
        bins = np.array(
            [
                6.52,
                8.54,
                11.2,
                14.6,
                19.1,
                25.1,
                32.8,
                43.0,
                56.2,
                73.6,
                96.3,
                126.0,
                165.0,
                216.0,
                283.0,
                370.0,
                485.0,
                635.0,
                831.0,
                1090.0,
                1420.0,
                1860.0,
                2440.0,
                3190.0,
                4180.0,
                5470.0,
                7160.0,
                9370.0,
                12300.0,
                16100.0,
                21000.0,
                27500.0,
            ]
        )
    else:
        print(
            "Species not recognised, please enter either 'i' for ion or 'e' for electron."
        )
        bins = 0

    return bins