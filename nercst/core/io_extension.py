from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import necstdb

import io


def list_all_topics(path):
    """List file names which necstdb can read."""
    p = Path(path).glob("**/*")
    return sorted(set([a.stem for a in p if a.is_file()]))


def get_tp_data(path, topic_name, kisa_path=None, weather=False):
    """TotalPower data with numerous metadata.

    This function supports data from AC240 and `concatenate` without
    *kisa* file. Weather informations will be with the TP data if you
    set the option `weather=True`.

    Examples
    --------
    >>> path = '/mnt/usbdisk9/data/observation/otf_planet2018/n20200320131924_12CO_2-1_otfplanet_sun'
    >>> topic_name = ['xffts_power_board01', 'ac240_tp_data_1']
    >>> get_tp_data(path, topic_name[0])
    "<xarray.DataArray (t: 1191)>
    array([2.25653151e+11, ..."

    Notes
    -----
    [TODO] Support multiple topics, maybe by wrapping this function.

    """  # noqa: E501
    data = io.Initial_array(path, kisa_path, topic_name)
    dat, obsmode, *_ = data.get_tp_array()  # `_` is for value(s) to be ignored
    # also cf. extended unpacking

    # convert structured array to xarray
    coords = {"t": dat.t.astype(np.float64)}
    for name in dat.data.dtype.names:
        coords[name] = ("t", dat.data[name].astype(np.float64))
    # extract main data
    if "total_power" in dat.data.dtype.names:
        tp_data = coords.pop("total_power")[1]
    elif "POWER_BE1" in dat.data.dtype.names:
        tp_data = coords.pop("POWER_BE1")[1]
    else:
        raise ValueError("Total power field not found")
    # timestamp is not needed anymore
    coords.pop("timestamp")
    # make DataArray
    data_array = xr.DataArray(
        tp_data,
        dims=["t"],
        coords=coords,
    )

    # perform or escape `apply_kisa`
    if kisa_path:
        data.apply_kisa()
    else:
        data.kisa_applyed_az = data.az_array  # fooling instance attribute(s)
        data.kisa_applyed_el = data.el_array  # is not a good way...

    # create weather informations coordinates
    if weather:
        db = necstdb.opendb(path)
        weather = db.open_table("status_weather").read(astype="array")
        contents = {"t": weather["timestamp"].astype(np.float64)}
        for name in weather.dtype.names:
            contents[name] = ("t", weather[name].astype(np.float64))
        contents.pop("timestamp")
        contents.pop("received_time")  # due to conflict with `received_time` of TP data
        # apply the coordinates
        weather = xr.Dataset(contents)
        reindexed_weather = weather.interp_like(data_array)
        coords = {}
        for key in reindexed_weather.variables.keys():
            coords[key] = ("t", reindexed_weather[key])
        data_array = data_array.assign_coords(**coords)

    # list coordinates of constant values (possibly incorrect values) or `nan` only
    unique = []
    for name in data_array.coords.keys():
        elem = np.unique(data_array[name])  # `nan`s are not compressed
        if len(elem[~np.isnan(elem)]) <= 1:  # len(`nan`s removed), tilde is bitwise NOT
            unique.append(name)
    data.data_array = data_array.assign_attrs({"constant_or_nodata": unique})

    # convert timestamp to datetime obj, making `utcfromtimestamp` accept iterable
    ret = data.concatenate()
    # set descriptive string instead of digits
    ret = ret.assign_coords(
        {
            "t": np.vectorize(datetime.utcfromtimestamp)(ret.t),
            "obsmode": ("t", obsmode.reindex_like(ret, "nearest")),
        }
    )
    return ret


def get_spec_data(path, topic_name, kisa_path=None, weather=False):
    """Spectral data with numerous metadata.

    This function supports data from AC240 and `concatenate` without
    *kisa* file. Weather informations will be with the data if you
    set the option `weather=True`.

    Examples
    --------
    >>> path = '/mnt/usbdisk9/data/observation/otf/otf_2019/n20200320193251_12CO_2-1_otf_OriKL'
    >>> topic_name = ['xffts_board01', 'ac240_spactra_data_1']
    >>> kisa_path = '/home/amigos/seigyo/backup/hosei/hosei_230.txt'
    >>> get_spec_data(path, topic_name[0], kisa_path, True)
    "<xarray.DataArray (t: 20934, spec: 32768)>\narray([[1.28365281e+10, ..."

    Notes
    -----
    [TODO] Support multiple topics, maybe by wrapping this function.
    """  # noqa: E501
    data = io.Initial_array(path, kisa_path, topic_name)
    dat, obsmode, *_ = data.get_tp_array()  # not a bug, this is a WORKAROUND

    # convert structured array to xarray
    coords = {"t": dat.t.astype(np.float64)}
    for name in dat.data.dtype.names:
        coords[name] = ("t", dat.data[name].astype(np.float64))
    # extract main data
    if "spec" in dat.data.dtype.names:
        tp_data = coords.pop("spec")[1]
    else:
        raise ValueError("Spectral data field not found")
    # timestamp is not needed anymore
    coords.pop("timestamp")
    # make DataArray
    data_array = xr.DataArray(
        tp_data,
        dims=["t", "spec"],
        coords=coords,
    )

    # perform or escape `apply_kisa`
    if kisa_path:
        data.apply_kisa()
    else:
        data.kisa_applyed_az = data.az_array  # fooling instance attribute(s)
        data.kisa_applyed_el = data.el_array  # is not a good way...

    # create weather informations coordinates
    if weather:
        db = necstdb.opendb(path)
        weather = db.open_table("status_weather").read(astype="array")
        contents = {"t": weather["timestamp"].astype(np.float64)}
        for name in weather.dtype.names:
            contents[name] = ("t", weather[name].astype(np.float64))
        contents.pop("timestamp")
        contents.pop("received_time")  # due to conflict with `received_time` of TP data
        # apply the coordinates
        weather = xr.Dataset(contents)
        reindexed_weather = weather.interp_like(data_array)
        coords = {}
        for key in reindexed_weather.variables.keys():
            coords[key] = ("t", reindexed_weather[key])
        data_array = data_array.assign_coords(**coords)

    # list coordinates of constant values (possibly incorrect values) or `nan` only
    unique = []
    for name in data_array.coords.keys():
        elem = np.unique(data_array[name])  # `nan`s are not compressed
        if len(elem[~np.isnan(elem)]) <= 1:  # len(`nan`s removed), tilde is bitwise NOT
            unique.append(name)
    data.data_array = data_array.assign_attrs({"constant_or_nodata": unique})

    # convert timestamp to datetime obj, making `utcfromtimestamp` accept iterable
    ret = data.concatenate()
    # set descriptive string instead of digits
    ret = ret.assign_coords(
        {
            "t": np.vectorize(datetime.utcfromtimestamp)(ret.t),
            "obsmode": ("t", obsmode.reindex_like(ret, "nearest")),
        }
    )
    return ret
