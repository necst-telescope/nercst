import xarray as xr
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Literal, Any
from xarray_dataclasses import AsDataArray, Coord, Data

T = Literal["t"]
Ch = Literal["ch"]


@dataclass(frozen=True)
class TimeSeriesArray(AsDataArray):

    """Specification for NERCST basic array"""

    # TODO: automatic generation from neclib.config and/or ROS topic list
    data: Data[Tuple[T, Ch], Any]
    t: Coord[T, datetime] = 0
    ch: Coord[Ch, int] = 0
    vrad: Coord[T, float] = 0
    time: Coord[T, float] = 0
    in_temp: Coord[T, float] = 0
    out_temp: Coord[T, float] = 0
    in_humi: Coord[T, float] = 0
    out_humi: Coord[T, float] = 0
    wind_sp: Coord[T, float] = 0
    wind_dir: Coord[T, float] = 0
    press: Coord[T, float] = 0
    rain: Coord[T, float] = 0
    cabin_temp1: Coord[T, float] = 0
    cabin_temp2: Coord[T, float] = 0
    dome_temp1: Coord[T, float] = 0
    dome_temp2: Coord[T, float] = 0
    gen_temp1: Coord[T, float] = 0
    gen_temp2: Coord[T, float] = 0


@xr.register_dataarray_accessor("ts")
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


def make_time_series_array(
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
