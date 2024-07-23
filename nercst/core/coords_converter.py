import astropy.units as u
import xarray as xr

from astropy.time import Time
from astropy.coordinates import EarthLocation
from neclib.coordinates import parse_frame
from astropy.coordinates import SkyCoord
from neclib.coordinates import PointingError
from neclib.core import RichParameters


def read_location(array: xr.DataArray):
    config_filepath = array.attrs["config_filepath"]
    config = RichParameters.from_file(config_filepath)
    config.attach_parsers(location=lambda x: EarthLocation(**x))
    return config.location


def add_celestial_coords(array: xr.DataArray) -> xr.DataArray:
    pepath = array.attrs["pointing_params_path"]
    pe = PointingError.from_file(pepath)
    lon_list = []
    lat_list = []
    for _lon, _lat in zip(array["lon"].values, array["lat"].values):
        lon, lat = pe.apparent_to_refracted(
            _lon * u.deg,
            _lat * u.deg,
        )
        lon_list.append(lon)
        lat_list.append(lat)
    obstime = Time(array.t, format="unix")
    location = read_location(array)
    lon_lat = SkyCoord(
        lon_list, lat_list, frame="altaz", obstime=obstime, location=location
    ).transform_to("icrs")
    array = array.assign_coords({"ra_cor": ("t", lon_lat.ra.value)})
    array = array.assign_coords({"dec_cor": ("t", lon_lat.dec.value)})
    return array


def convert_frame(array: xr.DataArray, frame: str) -> xr.DataArray:
    target_frame = parse_frame(frame)
    location = read_location(array)
    coords = SkyCoord(
        ra=array["ra_cor"].data * u.deg,
        dec=array["dec_cor"].data * u.deg,
        frame="icrs",
        obstime=Time(array.t, format="unix"),
        location=location,
    ).transform_to(target_frame)
    if target_frame.name in ["fk4", "fk5"]:
        array = array.assign_coords(
            {f"ra_cor_{target_frame.name}": ("t", coords.ra.value)}
        )
        array = array.assign_coords(
            {f"dec_cor_{target_frame.name}": ("t", coords.dec.value)}
        )
    if "galactic" in target_frame.name:
        array = array.assign_coords({"l_cor": ("t", coords.l.value)})
        array = array.assign_coords({"b_cor": ("t", coords.b.value)})
    if "altaz" in target_frame.name:
        array = array.assign_coords({"lon_cor": ("t", coords.az.value)})
        array = array.assign_coords({"lat_cor": ("t", coords.alt.value)})
    return array
