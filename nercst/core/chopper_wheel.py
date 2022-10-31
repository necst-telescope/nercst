import xarray as xr
import numpy as np


def scanmask(time_series_array: xr.DataArray) -> xr.DataArray:
    scan_num_list = np.unique(time_series_array["scan_num"])

    mean_array_list = []

    for scan_num in scan_num_list:
        scanmask = time_series_array["scan_num"] == scan_num
        scanmasked_array = time_series_array[scanmask].mean(axis=1)

        mean_array_list.append(scanmasked_array)
    mean_array = xr.concat(mean_array_list, dim="t")
    return mean_array


def chopper_wheel(
    on_array: xr.DataArray, off_array: xr.DataArray, hot_array: xr.DataArray
) -> xr.DataArray:
    if len(np.unique(off_array["scan_num"])) >= 2:
        print()
        print("OFF array interpolated")
        off_array = scanmask(off_array)
        off_array = off_array.interp_like(on_array)
    else:
        off_array = off_array.mean(axis=0)
    if len(np.unique(hot_array["scan_num"])) >= 2:
        print("HOT array interpolated")
        hot_array = scanmask(hot_array)
        hot_array = hot_array.interp_like(on_array)
    else:
        hot_array = hot_array.mean(axis=0)

    calib_array = ((on_array - off_array) / (hot_array - off_array)) * 300

    return calib_array
