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
    telescop: Literal["NANTEN2", "OMU1p85m", "previous"],
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
    telescop : Literal["NANTEN2", "OMU1p85m", "previous"]
        Use parameter ``NANTEN2`` and ``OMU1p85m`` if you are using the
        NECST v4 system. ``previous`` is for the NECST v2 or v3.

    Examples
    --------
    >>> from nercst.core import io
    >>> array_n2 = io.loaddb("path/to/necstdb", "xffts-board1", "NANTEN2")
    >>> array_1p85 = io.loaddb("path/to/necstdb", "xffts-board1", "OMU1p85m")
    >>> array_1p85_old = io.loaddb("path/to/necstdb", "board1", "OMU1p85m")
    >>> array_v2 = io.loaddb("path/to/necstdb", "xffts_board01", "previous")

    """

    array_list = []
    if telescop == "previous":
        db = necstdb.opendb(dbname)
        data = db.open_table(board).read(astype="array")
        encoder = db.open_table("status_encoder").read(astype="array")
        array_list.append(encoder)
        try:
            weather = db.open_table("status_weather").read(astype="array")
            array_list.append(weather)
        except Exception as e:
            print(e)

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
        array_list.append(obsmode)

    else:
        db = necstdb.opendb(dbname)
        spec_topicname = f"necst-{telescop.upper()}-data-spectral-{board}"
        data = db.open_table(spec_topicname).read(astype="array")
        obsmode = db.open_table(spec_topicname).read(
            astype="array", cols=["time", "position"]
        )
        array_list.append(obsmode)
        scan_num = db.open_table(spec_topicname).read(
            astype="array", cols=["time", "id"]
        )
        array_list.append(scan_num)
        encoder = db.open_table(f"necst-{telescop.upper()}-ctrl-antenna-encoder").read(
            astype="array"
        )
        array_list.append(encoder)
        try:
            weather = db.open_table(f"necst-{telescop.upper()}-weather-ambient").read(
                astype="array"
            )
            array_list.append(weather)
        except Exception as e:
            print(e)
        spec_label = "data"

    data_tlabel = get_timelabel(data)

    df_reindex_list = []
    for array in array_list:
        array_tlabel = get_timelabel(array)
        _df = get_time_indexed_df(array, array_tlabel)
        _df = _df.sort_index().reindex(index=data[data_tlabel], method="bfill")
        df_reindex_list.append(_df)

    time_coords = pd.concat(df_reindex_list, axis=1).to_dict(orient="list")
    channel_coords = {"channel": np.arange(len(data[spec_label][0]))}
    loaded = nercst.core.struct.make_time_series_array(
        data[spec_label],
        time_coords=time_coords,
        channel_coords=channel_coords,
    )

    loaded["t"] = data[data_tlabel]
    loaded["ch"] = pd.Index(np.arange(data[spec_label].shape[1]))

    config_filepath_list = [
        Path(file_path) for file_path in glob(str(dbname) + f"/*config.toml")
    ]
    if len(config_filepath_list) == 1:
        loaded = loaded.assign_attrs(config_filepath=config_filepath_list[0])
    else:
        for file_path in config_filepath_list:
            if telescop in file_path.name:
                loaded = loaded.assign_attrs(config_filepath=file_path)

    try:
        obs_filepath = Path(glob(str(dbname) + "/*.obs")[0])
        loaded = loaded.assign_attrs(obs_filepath=obs_filepath)
    except IndexError:
        pass

    if pe_cor:
        pointing_parampath = Path(str(dbname) + "/pointing_param.toml")
        loaded = loaded.assign_attrs(pointing_params_path=pointing_parampath)
        loaded = add_celestial_coords(loaded)
        if dop_cor:
            device_setting_path = Path(str(dbname) + "/device_setting.toml")
            loaded = loaded.assign_attrs(device_setting_path=device_setting_path)
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
