import necstdb
import numpy as np
import xarray as xr

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Union, Tuple, Literal, List, Dict, Any
from xarray_dataclasses import AsDataArray, Coord, Data


T = Literal["t"]
Ch = Literal["ch"]
_ = Tuple[()]


@dataclass(frozen=True)
class TimeSeriesArray(AsDataArray):

    """Specification for NERCST basic array"""

    # TODO: automatic generation from neclib.config and/or ROS topic list
    data: Data[Tuple[T, Ch], Any]
    t: Coord[T, datetime]
    ch: Coord[Ch, int]
    vrad: Coord[T, float]
    time: Coord[T, float]
    in_temp: Coord[T, float]
    out_temp: Coord[T, float]
    in_humi: Coord[T, float]
    out_humi: Coord[T, float]
    wind_sp: Coord[T, float]
    wind_dir: Coord[T, float]
    press: Coord[T, float]
    rain: Coord[T, float]
    cabin_temp1: Coord[T, float]
    cabin_temp2: Coord[T, float]
    dome_temp1: Coord[T, float]
    dome_temp2: Coord[T, float]
    gen_temp1: Coord[T, float]
    gen_temp2: Coord[T, float]


@xr.register_dataarray_accessor("dcc")
@dataclass(frozen=True)
class TimeSeriesArrayAccessor:
    time_series_array: xr.DataArray

    @property
    def time_coords(self):
        """Dictionary of arrays that label time axis."""
        return {
            k: v.values
            for k, v in self.time_series_array.coords.items()
            if v.dims == ("t",)
        }

    @property
    def channel_coords(self):
        """Dictionary of arrays that label channel axis."""
        return {
            k: v.values
            for k, v in self.time_series_array.coords.items()
            if v.dims == ("ch",)
        }

    @property
    def data_coords(self):
        """Dictionary of arrays that label time and channel axis."""
        return {
            k: v.values
            for k, v in self.time_series_array.coords.items()
            if v.dims == ("t", "ch")
        }

    @property
    def scalar_coords(self):
        """Dictionary of values that don't label any axes (point-like)."""
        return {
            k: v.values
            for k, v in self.time_series_array.coords.items()
            if v.dims == ()
        }


def make_xarray(
    data,
    time_coords=None,
    channel_coords=None,
    scalar_coords=None,
    data_coords=None,
):
    tsarray = TimeSeriesArray.new(data)
    # update coords with input values (if any)
    if time_coords is not None:
        tsarray.coords.update({k: ("t", v) for k, v in time_coords.items()})

    if channel_coords is not None:
        tsarray.coords.update({k: ("ch", v) for k, v in channel_coords.items()})

    if data_coords is not None:
        tsarray.coords.update({k: (("t", "ch"), v) for k, v in data_coords.items()})

    if scalar_coords is not None:
        tsarray.coords.update({k: ((), v) for k, v in scalar_coords.items()})

    return tsarray
