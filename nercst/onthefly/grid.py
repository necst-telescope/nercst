import numpy as np
import xarray as xr

from scipy.interpolate import griddata
from astropy import units as u


def make_grid(array, grid_size, map_center, grid_number_x, grid_number_y):
    if not isinstance(grid_size, u.Quantity):
        raise (TypeError("grid units must be specified"))
    else:
        pass

    if not isinstance(map_center, tuple):
        raise (TypeError("map center must be given as a tuple of coordinate"))
    if (not isinstance(map_center[0], u.Quantity)) & (
        not isinstance(map_center[1], u.Quantity)
    ):
        raise (TypeError("map center units must be specified"))
    else:
        pass

    grid_size_deg = grid_size.to(u.deg).value
    map_center = np.array(
        [map_center[0].to(u.deg).value, map_center[1].to(u.deg).value]
    )

    lon_coords = np.linspace(
        map_center[0] - grid_size_deg * grid_number_x / 2,
        map_center[0] + grid_size_deg * grid_number_x / 2,
        grid_number_x,
    )
    lat_coords = np.linspace(
        map_center[1] - grid_size_deg * grid_number_y / 2,
        map_center[1] + grid_size_deg * grid_number_y / 2,
        grid_number_y,
    )
    vcoords = array.ch.values
    grid = np.meshgrid(lon_coords, lat_coords, vcoords)
    # grid = np.meshgrid(lon_coords, lat_coords)

    return grid


def gridding(array: xr.DataArray, grid: list, method="nearest"):
    Xarr = np.ones_like(array.data) * array.lon_cor.data[:, None]
    Yarr = np.ones_like(array.data) * array.lat_cor.data[:, None]
    Zarr = array.channel.data * np.ones_like(array.data)
    cube = griddata(
        points=(Xarr.ravel(), Yarr.ravel(), Zarr.ravel()),
        values=array.data.ravel(),
        xi=(grid[0], grid[1], grid[2]),
        method=method,
    )
    return cube
