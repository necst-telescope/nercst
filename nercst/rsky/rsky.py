import xarray as xr


def rsky(hot_array: xr.DataArray, sky_array: xr.DataArray) -> xr.DataArray:
    y_factor = hot_array / sky_array
    tsys_array = 300.0 / (y_factor - 1.0)
    return tsys_array
