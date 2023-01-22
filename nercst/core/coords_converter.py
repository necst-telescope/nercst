import astropy.units as u
import xarray as xr

from neclib import config
from astropy.time import Time
from neclib.coordinates import parse_frame
from astropy.coordinates import SkyCoord
from neclib.parameters import PointingError


def add_celestial_coords(array: xr.DataArray, frame: str) -> xr.DataArray:

    pe = PointingError("OMU1P85M")
    lon, lat = pe.apparent2refracted(
        array["lon"].values * u.deg, array["lat"].values * u.deg
    )
    target_frame = parse_frame(frame)
    obstime = Time(array.t, format="unix")
    lon_lat = SkyCoord(
        lon, lat, frame="altaz", obstime=obstime, location=config.location
    ).transform_to(target_frame)
    if "fk5" in frame:
        array = array.assign_coords({"lon_cor": ("t", lon_lat.ra.value)})
        array = array.assign_coords({"lat_cor": ("t", lon_lat.dec.value)})
    if "Galactic" in frame:
        array = array.assign_coords({"lon_cor": ("t", lon_lat.lon.value)})
        array = array.assign_coords({"lat_cor": ("t", lon_lat.lat.value)})
    return array
