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
from .multidimensional_coordinates import add_celestial_coords, add_radial_velocity

PathLike = Union[str, os.PathLike]
timestamp2datetime = np.vectorize(datetime.utcfromtimestamp)


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
    board: str,
    telescop: Literal["NANTEN2", "OMU1P85M", "previous"],
    pe_cor=True,
    dop_cor=False,
):
    """Data loader for the necst telescopes

    Parameters
    ----------
    dbname : PathLike
        File path for the data to be loaded
    board : str
        For NECST v4 system, the ``necst-{telescop}-data-spectral-{board}`` is loaded if you use parameter such as ``xffts-board1`` or ``ac240_1-board1` in {board}.
        Use parameter such as ``xffts_board01`` for NECST v2 or v3.
    telescop : Literal["NANTEN2", "OMU1P85M", "previous"]
        Use parameter ``NANTEN2`` and ``OMU1P85M`` if you are using the
        NECST v4 system. ``previous`` is for the NECST v2 or v3.

    Examples
    --------
    >>> from nercst.core import io
    >>> array = io.loaddb("path/to/necstdb", "spec-topic-name", "NANTEN2")

    """

    if telescop == "previous":
        db = necstdb.opendb(dbname)
        data = db.open_table(board).read(astype="array")
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

    else:
        db = necstdb.opendb(dbname)
        spec_topicname = f"necst-{telescop}-data-spectral-{board}"
        data = db.open_table(spec_topicname).read(astype="array")
        obsmode = db.open_table(spec_topicname).read(
            astype="array", cols=["time", "position"]
        )
        scan_num = db.open_table(spec_topicname).read(
            astype="array", cols=["time", "id"]
        )
        encoder = db.open_table("necst-{telescop}-ctrl-antenna-encoder").read(
            astype="array"
        )
        weather = db.open_table("necst-{telescop}-weather-ambient").read(astype="array")
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

    pointing_parampath = Path(str(dbname) + "/pointing_param.toml")
    obs_filepath = Path(glob(str(dbname) + "/*.obs")[0])
    config_filepath = Path(glob(str(dbname) + f"/{telescop}_config.toml")[0])
    device_setting_path = Path(str(dbname) + "/device_setting.toml")
    loaded = loaded.assign_attrs(pointing_params_path=pointing_parampath)
    loaded = loaded.assign_attrs(obs_filepath=obs_filepath)
    loaded = loaded.assign_attrs(config_filepath=config_filepath)
    loaded = loaded.assign_attrs(device_setting_path=device_setting_path)

    if pe_cor:
        loaded = add_celestial_coords(loaded)
        if dop_cor:
            loaded = add_radial_velocity(
                spec_array=loaded, dbname=dbname, topic_name=spec_topicname
            )
    else:
        if dop_cor:
            raise ValueError("Not apply doppler correction without pointing correction")

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
