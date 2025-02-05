import necstdb
import numpy as np
import pandas as pd

import os
from glob import glob
import logging

from pathlib import Path
from datetime import datetime
from typing import Union, Literal, get_args
from astropy.coordinates import EarthLocation
from .multidimensional_coordinates import add_celestial_coords, add_radial_velocity
from .struct import make_time_series_array
from neclib.core import RichParameters

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


logger = logging.getLogger("necst")
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.DEBUG)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)


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
    obs_line=None,
    pe_cor=True,
    dop_cor=False,
):
    """Data loader for the necst telescopes

    Parameters
    ----------
    dbname : PathLike
        File path for the data to be loaded
    board : str
        For NECST v4 system, the ``necst-{telescop}-data-spectral-{board}``
        is loaded if you use parameter such as ``xffts-board1`` or
        ``ac240_1-board1` in {board}.
        Use parameter such as ``xffts_board01`` for NECST v2 or v3.
    telescop : Literal["NANTEN2", "OMU1p85m", "previous"]
        Use parameter ``NANTEN2`` and ``OMU1p85m`` if you are using the
        NECST v4 system. ``previous`` is for the NECST v2 or v3.
    obs_line : str or astropy.Quantity
        Observed line name listed in analysis_params or frequency. For example, "12CO(1-0)" or 115.27120*u.GHz.

    Examples
    --------
    >>> from nercst.core import io
    >>> array_n2 = io.loaddb("path/to/necstdb", "xffts-board1", "NANTEN2", obs_line="12CO(1-0)", pe_cor=True, dop_cor=True)
    >>> array_1p85 = io.loaddb("path/to/necstdb", "xffts-board1", "OMU1p85m", obs_line=230.538000*u.GHz, pe_cor=True, dop_cor=True)
    >>> array_1p85_old = io.loaddb("path/to/necstdb", "board1", "OMU1p85m", "12CO(2-1)")
    >>> array_v2 = io.loaddb("path/to/necstdb", "xffts_board01", "previous", "12CO(2-1)")

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
            logger.warning(e)

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
            logger.warning(e)
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
    loaded = make_time_series_array(
        data[spec_label],
        time_coords=time_coords,
        channel_coords=channel_coords,
    )

    loaded["t"] = data[data_tlabel]
    loaded["ch"] = pd.Index(np.arange(data[spec_label].shape[1]))

    config_filepath_list = [
        Path(file_path) for file_path in glob(str(dbname) + "/*config.toml")
    ]
    if len(config_filepath_list) == 1:
        loaded = loaded.assign_attrs(config_filepath=config_filepath_list[0])
    else:
        for file_path in config_filepath_list:
            if telescop in file_path.name:
                loaded = loaded.assign_attrs(config_filepath=file_path)

    device_setting_filepath_list = [
        Path(file_path) for file_path in glob(str(dbname) + "/device_setting.toml")
    ]
    if len(device_setting_filepath_list) == 1:
        loaded = loaded.assign_attrs(
            device_setting_path=device_setting_filepath_list[0]
        )
    elif (len(device_setting_filepath_list) == 0) & dop_cor:
        raise FileNotFoundError(f"device_setting.toml dose not exist in {dbname}.")
    else:
        pass

    try:
        obs_filepath = Path(glob(str(dbname) + "/*.obs")[0])
        loaded = loaded.assign_attrs(obs_filepath=obs_filepath)
    except IndexError:
        pass

    if pe_cor:
        pointing_parampath_list = glob(str(dbname) + "/pointing_param.toml")
        if len(pointing_parampath_list) == 0:
            logger.warning(
                f"File of pointing_params dose not exist in {dbname}."
                " Assign pointing_parampath manually;"
                " `loaded.assign_attrs(pointing_params_path=``pointing_parampath'')`"
                " and then execute `add_celestial_coords(loaded)`."
            )
        else:
            loaded = loaded.assign_attrs(
                pointing_params_path=Path(pointing_parampath_list[0])
            )
            loaded = add_celestial_coords(loaded)
        if dop_cor:
            loaded = add_radial_velocity(
                spec_array=loaded,
                dbname=dbname,
                board=board,
                telescop=telescop,
                obs_line=obs_line,
            )
    else:
        if dop_cor:
            raise ValueError("Not apply doppler correction without pointing correction")

    return loaded


def topic_getter(dbname: PathLike):
    db = necstdb.opendb(dbname)
    spectral_data = [
        tablename for tablename in db.list_tables() if "data-spectral" in tablename
    ]

    if len(spectral_data) == 0:
        args = get_args(TypeBoards)
        topics = db.list_tables()
        args_set = set(args)
        topic_set = set(topics)
        return list(args_set & topic_set)
    else:
        return spectral_data


def read_tomlfile(file_path: PathLike):
    config = RichParameters.from_file(file_path)
    try:
        config.attach_parsers(location=lambda x: EarthLocation(**x))
    except:
        logger.info("No location info")
    return config
