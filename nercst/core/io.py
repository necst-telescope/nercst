import necstdb
import numpy as np
import pandas as pd
import nercst
import os
import neclib
from glob import glob

from pathlib import Path
from datetime import datetime
from typing import Union, Literal, get_args
from nercst.core.coords_converter import add_celestial_coords

PathLike = Union[str, os.PathLike]
timestamp2datetime = np.vectorize(datetime.utcfromtimestamp)
TypeBoards = Literal[
    "xffts_board01",
    "xffts_board02",
    "xffts_board03",
    "xffts_board04",
    "xffts_board05",
    "xffts_board06",
    "xffts_board07",
    "xffts_board08",
    "xffts_board09",
    "xffts_board10",
    "xffts_board11",
    "xffts_board12",
    "xffts_board13",
    "xffts_board14",
    "xffts_board15",
    "xffts_board16",
]


def get_timelabel(structure: np.ndarray):
    timestamps = []
    for name in structure.dtype.names:
        if "time" in name:
            timestamps.append(name)

    if len(timestamps) > 1:
        for time in timestamps:
            if "stamp" in time:
                tlabel = time
            elif time == "time":
                tlabel = time
            else:
                pass
    else:
        tlabel = timestamps[0]

    return tlabel


def get_time_indexed_df(structure: np.ndarray, tlabel):
    df = pd.DataFrame()
    for label in structure.dtype.names:
        df.index = structure[tlabel]
        if "time" not in label:
            df[label] = structure[label]
        else:
            pass
    return df


def loaddb(
    dbname: PathLike,
    spec_topicname: TypeBoards,
    telescop: Literal["NANTEN2", "OPU1.85", "Common"] = "Common",
    pe_cor=True,
):
    """Data loader for the necst telescopes

    Parameters
    ----------
    dbname : PathLike
        File path for the data to be loaded
    spec_topicname : TypeBoards
        Topic name to specify the spectroscopic data. You can subtract
        the topic key for the spectroscopic data using
        the `topic_getter` function.
    telescop : Literal["NANTEN2", "OPU1.85", "Common"]
        Use default parameter ``Common`` if you are using the
        NECST v4 system. ``NANTEN2``, ``OPU-1.85`` are for the NECST v2
        and v3,respectively.

    Examples
    --------
    >>> from nercst.core import io
    >>> array = io.loaddb("path/to/necstdb")

    """

    if telescop == "NANTEN2":
        db = necstdb.opendb(dbname)
        data = db.open_table(spec_topicname).read(astype="array")
        encoder = db.open_table("status_encoder").read(astype="array")
        weather = db.open_table("status_weather").read(astype="array")
        spec_label = "spec"
        data_tlabel = get_timelabel(data)

        # backward compatibillity
        obsmode = db.open_table("obsmode").read(astype="array")
        fields = []
        for field in obsmode.dtype.names:
            if field != "obs_mode":
                fields.append(field)
            else:
                fields.append("position")
        obsmode.dtype.names = tuple(fields)

    elif telescop == "Common":
        db = necstdb.opendb(dbname)
        data = db.open_table(spec_topicname).read(astype="array")
        obsmode = db.open_table(spec_topicname).read(
            astype="array", cols=["time", "position"]
        )
        scan_num = db.open_table(spec_topicname).read(
            astype="array", cols=["time", "id"]
        )
        encoder = db.open_table("necst-OMU1P85M-ctrl-antenna-encoder").read(
            astype="array"
        )
        weather = db.open_table("necst-OMU1P85M-weather-ambient").read(astype="array")
        spec_label = "data"

    data_tlabel = get_timelabel(data)

    enc_tlabel = get_timelabel(encoder)
    df_enc = get_time_indexed_df(encoder, enc_tlabel)
    df_enc = df_enc.sort_index().reindex(index=data[data_tlabel], method="bfill")

    obs_tlabel = get_timelabel(obsmode)
    df_obsmode = get_time_indexed_df(obsmode, obs_tlabel)
    df_obsmode = df_obsmode.sort_index().reindex(
        index=data[data_tlabel], method="bfill"
    )

    df_scan_num = get_time_indexed_df(scan_num, data_tlabel)
    df_scan_num = df_scan_num.sort_index().reindex(
        index=data[data_tlabel], method="bfill"
    )

    weather_tlabel = get_timelabel(weather)
    df_weather = get_time_indexed_df(weather, weather_tlabel)
    df_weather = df_weather.sort_index().reindex(
        index=data[data_tlabel], method="bfill"
    )

    time_coords = pd.concat(
        [df_enc, df_weather, df_obsmode, df_scan_num], axis=1
    ).to_dict(orient="list")
    channel_coords = {"channel": np.arange(len(data[spec_label][0]))}
    loaded = nercst.core.struct.make_time_series_array(
        data[spec_label],
        time_coords=time_coords,
        channel_coords=channel_coords,
    )

    loaded["t"] = data[data_tlabel]
    loaded["ch"] = pd.Index(np.arange(32768))

    pointing_parampath = Path(dbname + "/pointing_param.toml")
    obs_filepath = Path(glob(dbname + "/*.obs")[0])
    obs_filepath = Path(glob(dbname + "/config.toml")[0])
    loaded = loaded.assign_attrs(pointing_params_path=pointing_parampath)
    loaded = loaded.assign_attrs(obs_filepath=obs_filepath)

    if pe_cor:
        frame = neclib.core.files.toml.read(obs_filepath)["coordinate"]["coord_sys"]
        return add_celestial_coords(loaded, frame, pointing_parampath)
    else:
        return loaded


def topic_getter(dbname: PathLike):
    db = necstdb.opendb(dbname)
    config = neclib.config

    prefix = f"necst-{config.observatory}-"
    spectral_data = [
        tablename
        for tablename in db.list_tables()
        if tablename.startswith(prefix + "data-spectral")
    ]

    args = get_args(TypeBoards)
    topics = db.list_tables()
    args_set = set(args)
    topic_set = set(topics)
    spectral_data = set(spectral_data)

    if len(list(args_set & topic_set)) == 0:
        return spectral_data
    else:
        return list(args_set & topic_set)


"""
    def convert(
        structured_array: np.ndarray, dims: List[str]
    ) -> Dict[str, xr.DataArray]:
        if not structured_array.dtype.names:  # when not structured
            return {"total_power": xr.DataArray(structured_array, dims=dims)}
        ret = {}
        for field in structured_array.dtype.names:
            dim = dims[: structured_array[field].ndim]
            ret[field] = xr.DataArray(structured_array[field], dims=dim)
        return ret

    def interp(dict_of_xarrays: Dict, tref: np.ndarray) -> Dict[str, xr.DataArray]:
        ret = {}
        for k, v in dict_of_xarrays.items():
            ret[k] = v.interp(t=tref)
        return k

    def reindex(
        dict_of_xarrays: Dict, ref: Union[xr.DataArray, np.ndarray]
    ) -> Dict[str, xr.DataArray]:
        ret = {}
        for k, v in dict_of_xarrays.items():
            ret[k] = v.reindex_like(ref, method="bfill")
        return k

    data = convert(data, ["t", "spectra"])
    obsmode = convert(obsmode, ["t"])
    encoder = convert(encoder, ["t"])
    weather = convert(weather, ["t"])

    if debug:
        return data, obsmode, encoder, weather

    tref = data["timestamp"]

    encoder = interp(encoder, tref)
    weather = interp(weather, tref)
    obsmode = reindex(obsmode, data["spec"])

    time_coords = xr.concat([encoder, weather, obsmode], dims="t").to_dict()
    channel_coords = {"channel": np.arange(len(data["spec"][0]))}
"""
