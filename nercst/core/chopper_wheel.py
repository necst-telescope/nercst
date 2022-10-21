import xarray as xr
import numpy as np


def scanmask(time_series_array: xr.DataArray) -> xr.DataArray:
    scan_num_list = np.unique(time_series_array["scan_num"])

    mean_array_list = []

    for scan_num in scan_num_list:
        scanmask = time_series_array["scan_num"] == scan_num
        scanmasked_array = time_series_array[scanmask].mean()

        mean_array_list.append(scanmasked_array)
    mean_array = xr.concat(mean_array_list, dim="t")
    return mean_array


def chopper_wheel(
    on_array: xr.DataArray, off_array: xr.DataArray, hot_array: xr.DataArray
) -> xr.DataArray:
    fixed_off_array = scanmask(off_array)
    fixed_hot_array = scanmask(hot_array)
    reindexed_off_array = fixed_off_array.interp_like(on_array)
    reindexed_hot_array = fixed_hot_array.interp_like(on_array)

    calib_array = (
        (on_array - reindexed_off_array) / (reindexed_hot_array - reindexed_off_array)
    ) * 300

    return calib_array
