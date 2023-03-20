import xarray as xr
import numpy as np


def mean(array):
    mean = array.mean("t", keepdims=True)
    mean["t"] = array.t.mean("t", keepdims=True)
    return mean


def scanmask(time_series_array: xr.DataArray) -> xr.DataArray:
    try:
        ids = np.unique(time_series_array["id"])
        scan_key = "id"
    except KeyError:
        ids = np.unique(time_series_array["scan_num"])
        scan_key = "scan_num"

    mean_array_list = []

    for scan in ids:
        scanmask = time_series_array[scan_key] == scan
        scanmasked_array = time_series_array[scanmask]
        scanmasked_array_mean = mean(scanmasked_array)
        mean_array_list.append(scanmasked_array_mean)
    mean_array = xr.concat(mean_array_list, dim="t")
    return mean_array


def obsmode_sep(time_series_array: xr.DataArray) -> xr.DataArray:
    on_array = time_series_array[time_series_array["position"] == b"ON      "]
    off_array = time_series_array[time_series_array["position"] == b"OFF     "]
    hot_array = time_series_array[time_series_array["position"] == b"HOT     "]
    return on_array, off_array, hot_array


def chopper_wheel(time_series_array: xr.DataArray) -> xr.DataArray:

    on_array, off_array, hot_array = obsmode_sep(time_series_array)

    def get_scan_key(array: xr.DataArray):
        scan_key = "id" if list(on_array.coords.keys()).count("id") >= 1 else "scan_num"
        return scan_key

    scan_key = get_scan_key(on_array)
    if len(np.unique(off_array[scan_key])) >= 2:
        print("OFF array interpolated")
        off_array = scanmask(off_array)
        off_array = off_array.interp_like(on_array)
    else:
        off_array = off_array.mean(axis=0)
    if len(np.unique(hot_array[scan_key])) >= 2:
        print("HOT array interpolated")
        hot_array = scanmask(hot_array)
        hot_array = hot_array.interp_like(on_array)
    else:
        hot_array = hot_array.mean(axis=0)

    calib_array = ((on_array - off_array) / (hot_array - off_array)) * 300

    return calib_array
