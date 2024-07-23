import astropy.units as u
import xarray as xr

from astropy.time import Time
from astropy.coordinates import EarthLocation
from neclib.coordinates import parse_frame
from astropy.coordinates import SkyCoord
from neclib.coordinates import PointingError
from neclib.core import RichParameters


def add_celestial_coords(array: xr.DataArray, frame: str) -> xr.DataArray:
    config_filepath = array.attrs["config_filepath"]
    config = RichParameters.from_file(config_filepath)
    config.attach_parsers(location=lambda x: EarthLocation(**x))

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
    target_frame = parse_frame(frame)
    obstime = Time(array.t, format="unix")
    lon_lat = SkyCoord(
        lon_list, lat_list, frame="altaz", obstime=obstime, location=config.location
    ).transform_to(target_frame)
    if "fk5" in target_frame.name:
        array = array.assign_coords({"lon_cor": ("t", lon_lat.ra.value)})
        array = array.assign_coords({"lat_cor": ("t", lon_lat.dec.value)})
    if "Galactic" in target_frame.name:
        array = array.assign_coords({"lon_cor": ("t", lon_lat.lon.value)})
        array = array.assign_coords({"lat_cor": ("t", lon_lat.lat.value)})
    if "altaz" in target_frame.name:
        array = array.assign_coords({"lon_cor": ("t", lon.value)})
        array = array.assign_coords({"lat_cor": ("t", lat.value)})
    return array
