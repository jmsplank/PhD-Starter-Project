import os
import pyspedas
from pytplot import data_quants
import numpy as np
import logging as log
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))

log.basicConfig(
    filename=f"{path}/cdf2npy.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


class Product:
    NUMBERDENSITY = "numberdensity"
    TEMPPERP = "tempperp"
    TEMPPARA = "temppara"
    BULKV = "bulkv"


class Instrument:
    FPI = "fpi"
    FSM = "fsm"
    FGM = "fgm"


def download_data(EVENT, INSTRUMENT, SPECIES="", PRODUCT=""):
    dir_name = f"{path}/event_{EVENT}/data/{INSTRUMENT}"
    log.info(f"dir_name: {dir_name}")

    if EVENT == "a":
        trange = [
            "2020-03-18/02:05:23",
            "2020-03-18/02:44:52",
        ]
    elif EVENT == "b":
        # trange = [
        #     "2020-03-18/02:47:18",
        #     "2020-03-18/03:08:41",
        # ]
        trange = [
            "2020-03-18/02:57:00",
            "2020-03-18/03:08:41",
        ]

    probe = "1"
    data_rate = "brst"

    if INSTRUMENT == Instrument.FGM:
        pyspedas.mms.fgm(
            trange=trange,
            probe=probe,
            data_rate=data_rate,
            level="l2",
        )
        data = data_quants["mms1_fgm_b_gse_brst_l2"].values
        time = data_quants["mms1_fgm_b_gse_brst_l2"].coords["time"].values
    elif INSTRUMENT == Instrument.FSM:
        pyspedas.mms.fsm(
            trange=trange,
            probe=probe,
            data_rate=data_rate,
            level="l3",
        )
        data = data_quants["mms1_fsm_b_gse_brst_l3"].values
        time = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values
    elif INSTRUMENT == Instrument.FPI:
        pyspedas.mms.fpi(
            trange=trange,
            probe=probe,
            data_rate=data_rate,
            level="l2",
        )
        if PRODUCT == Product.NUMBERDENSITY:
            data = data_quants[f"mms1_d{SPECIES}s_numberdensity_brst"].values
            time = (
                data_quants[f"mms1_d{SPECIES}s_numberdensity_brst"]
                .coords["time"]
                .values
            )
        elif PRODUCT == Product.TEMPPERP:
            data = data_quants[f"mms1_d{SPECIES}s_tempperp_brst"].values
            time = data_quants[f"mms1_d{SPECIES}s_tempperp_brst"].coords["time"].values
        elif PRODUCT == Product.TEMPPARA:
            data = data_quants[f"mms1_d{SPECIES}s_temppara_brst"].values
            time = data_quants[f"mms1_d{SPECIES}s_temppara_brst"].coords["time"].values
        elif PRODUCT == Product.BULKV:
            data = data_quants[f"mms1_d{SPECIES}s_bulkv_gse_brst"].values
            time = data_quants[f"mms1_d{SPECIES}s_bulkv_gse_brst"].coords["time"].values
        else:
            raise NotImplementedError(
                f"No definition found for PRODUCT == Product.{PRODUCT.upper()}"
            )

    def interp(dat, finite_mask):
        log.warning(
            f"MISSING DATA {np.size(finite_mask) - np.count_nonzero(finite_mask)} non-finite"
        )
        log.info("Correcting missing through interpolation")
        return np.interp(time, time[finite_mask], dat[finite_mask])

    if len(np.shape(data)) > 1:
        log.info("Multi-dimensional data")
        for i in range(np.shape(data)[1]):
            log.info(f"looking at dimension {i}")
            dat = data[:, i]
            finite_mask = np.isfinite(dat)
            if np.size(dat) - np.sum(finite_mask) > 0:
                log.info("nans present")
                data[:, i] = interp(dat, finite_mask)
    else:
        log.info("single-dimension data")
        finite_mask = np.isfinite(data)
        if (np.size(data) - np.sum(finite_mask)) > 0:
            log.info("nans present")
            data = interp(data, finite_mask)

    log.info("Saving arrays")
    np.save(
        f"{dir_name}/data{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy",
        data,
    )
    log.info(
        f"Saved {dir_name}/data{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy"
    )
    np.save(
        f"{dir_name}/time{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy",
        time,
    )
    log.info(
        f"Saved {dir_name}/time{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy"
    )

    del data, time


if __name__ == "__main__":
    download_data(
        EVENT="b",
        INSTRUMENT=Instrument.FSM,
        SPECIES="",
        PRODUCT="",
    )
    # for EVENT in ["a", "b"]:
    #     for INSTRUMENT in [Instrument.FGM, Instrument.FPI, Instrument.FSM]:
    #         for SPECIES in ["i", "e"]:
    #             for PRODUCT in [Product.NUMBERDENSITY, Product.TEMPPERP, Product.BULKV]:
    #                 download_data(EVENT, INSTRUMENT, SPECIES, PRODUCT)
    # download_data("b", Instrument.FPI, "e", Product.BULKV)
    for PRODUCT in [Product.NUMBERDENSITY]:
        download_data("b", Instrument.FPI, "e", PRODUCT)
