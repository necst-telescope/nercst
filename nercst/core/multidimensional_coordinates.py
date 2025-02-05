import astropy.units as u
import numpy as np
import xarray as xr
import re
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import logging
from typing import Union, Literal
import os

from neclib.coordinates import Observer, parse_frame, PointingError
from neclib.core import RichParameters
import necstdb

PathLike = Union[str, os.PathLike]

logger = logging.getLogger("necst")
logger.setLevel(logging.DEBUG)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)

rest_frequency = {
    "12CO(1-0)": 115.271202 * u.GHz,
    "12CO(2-1)": 230.538000 * u.GHz,
    "13CO(1-0)": 110.201353 * u.GHz,
    "13CO(2-1)": 220.398681 * u.GHz,
    "C18O(1-0)": 109.782173 * u.GHz,
    "C18O(2-1)": 219.560354 * u.GHz,
    "SO": 219.940000 * u.GHz,
}


def calc_vobs(ra_array, dec_array, obstime_array, location):

    observer = Observer(location)
    logger.info("Calculating Vobs...")
    v_obs = observer.v_obs(
        lon=ra_array, lat=dec_array, time=obstime_array, frame="icrs", unit="deg"
    )
    return v_obs


def convert_to_velocity(
    channel_integer_numbers,
    freq_resolution,
    factor_1st_lo,
    freq_1st_lo,
    freq_2nd_lo,
    side_band,
    observation_frequency,
):

    logger.info("Converting channel into velocity...")
    obsfreq_array = channel_integer_numbers * freq_resolution
    if side_band == "usb":
        pre_hd_array = (obsfreq_array + freq_2nd_lo) + factor_1st_lo * freq_1st_lo
    elif side_band == "lsb":
        pre_hd_array = (-obsfreq_array + freq_2nd_lo) + factor_1st_lo * freq_1st_lo
    logger.info(f"rest frequency: {observation_frequency}")
    freq_to_velocity_equiv = u.doppler_radio(observation_frequency)
    observed_v_array = pre_hd_array.to(
        u.km * u.s ** (-1), equivalencies=freq_to_velocity_equiv
    )

    return observed_v_array


def get_vlsr(
    spec_array: xr.DataArray,
    freq_resolution: u.Quantity,
    factor_1st_lo: int,
    freq_1st_lo: u.Quantity,
    freq_2nd_lo: u.Quantity,
    side_band: Literal["usb", "lsb"],
    observation_frequency: u.Quantity,
    location: EarthLocation,
):
    """Calc radial velocity

    Examples
    --------
    >>> from nercst.core import io, multidimensional_coordinates
    >>> array_1p85 = io.loaddb("path/to/necstdb", "xffts-board1", "OMU1p85m")
    >>> config = io.read_tomlfile(array_1p85.attrs["config_filepath"])
    >>> freq_resolution = config.spectrometer.xffts.bw_MHz["1"]*u.MHz / config.spectrometer.xffts.max_ch
    >>> da_vrad = multidimensional_coordinates.get_vlsr(array_1p85,freq_resolution,factor_1st_lo=12,freq_1st_lo=18.75*u.GHz,freq_2nd_lo=4*u.GHz,"usb",230.538*u.GHz, config.location)
    """

    channel_integer_numbers = spec_array.channel.data
    observed_v_array = convert_to_velocity(
        channel_integer_numbers,
        freq_resolution,
        factor_1st_lo,
        freq_1st_lo,
        freq_2nd_lo,
        side_band,
        observation_frequency,
    )
    v_obs_array = calc_vobs(
        spec_array.ra_cor.data, spec_array.dec_cor.data, spec_array.t.data, location
    )
    observed_v_matrix = observed_v_array * np.ones(shape=spec_array.shape)
    velocity_array = observed_v_matrix + v_obs_array.reshape(-1, 1)
    return velocity_array


def make_dataset(data_array, velocity_array):
    ds = xr.Dataset()
    ds["spectral_data"] = data_array
    da = xr.DataArray(
        velocity_array.value, coords=data_array.coords, dims=data_array.dims
    )
    ds["radial_velocity"] = da
    return ds


def get_lo(dbname, band_name, side_band):
    db = necstdb.opendb(dbname)
    topic_list = list(
        filter(lambda x: re.search(f"lo_signal.{band_name}_1st", x), db.list_tables())
    )
    data_lo_1st = db.open_table(topic_list[0]).read(astype="array")
    freq_1st_lo = data_lo_1st["freq"][0] * u.GHz
    topic_signal = topic_list[0].replace("1st", "")
    data_lo_2nd = db.open_table(f"{topic_signal}{side_band}_2nd").read(astype="array")
    freq_2nd_lo = data_lo_2nd["freq"][0] * u.GHz
    return freq_1st_lo, freq_2nd_lo


def add_radial_velocity(
    spec_array: xr.DataArray,
    dbname: PathLike,
    board: str,
    telescop: Literal["NANTEN2", "OMU1p85m", "previous"],
    obs_line,
):
    """Add radial velocity array to spectral data array

    Parameters
    ----------
    spec_array : xr.DataArray
        xarray dataarray of spectral data
    dbname : PathLike
        File path for the data to be loaded
    board : str
        For NECST v4 system, the ``necst-{telescop}-data-spectral-{board}``
        is loaded if you use parameter such as ``xffts-board1`` or
        ``ac240_1-board1` in {board}.
        Use parameter such as ``xffts_board01`` for NECST v2 or v3.
    telescop : Literal["NANTEN2", "OMU1p85m", "previous"]
        Use parameter ``NANTEN2`` and ``OMU1p85m`` if you are using the
        NECST v4 system. ``previous`` is for the NECST v2 or v3.
    obs_line : str or astropy.Quantity
        Observed line name listed in analysis_params or frequency. For example, "12CO(1-0)" or 115.27120*u.GHz.

    Examples
    --------
    >>> from nercst.core import io, multidimensional_coordinates
    >>> array_1p85 = io.loaddb("path/to/necstdb", "xffts-board1", "OMU1p85m")
    >>> ds = multidimensional_coordinates.add_radial_velocity(array_1p85,"path/to/necstdb","xffts-board1", "OMU1p85m", "12CO(2-1)")
    """

    config_filepath = spec_array.attrs["config_filepath"]
    device_setting_filepath = spec_array.attrs["device_setting_path"]
    logger.info(f"read config file from {config_filepath}.")
    config = RichParameters.from_file(config_filepath)
    config.attach_parsers(location=lambda x: EarthLocation(**x))
    logger.info(f"read device setting file from {device_setting_filepath}.")
    setting = RichParameters.from_file(device_setting_filepath)
    board_split = board.split("-")
    board_id = re.sub(r"\D", "", board_split[-1])
    if len(board_split) == 1:
        bw = config.spectrometer["bw_MHz"][board_id] * u.MHz
        max_ch = config.spectrometer["max_ch"]
    else:
        board_name = board_split[0]
        bw = config.spectrometer[board_name]["bw_MHz"][board_id] * u.MHz
        max_ch = config.spectrometer[board_name]["max_ch"]
    freq_resolution = bw / max_ch
    if_name = setting.spectrometer.if_name[board]
    if_name_split = if_name.split("-")
    band_name = if_name_split[0]
    side_band = if_name_split[1]
    freq_1st_lo, freq_2nd_lo = get_lo(dbname, band_name, side_band)
    if (telescop == "NANTEN2") & (side_band == "lsb"):
        side_band = "usb"
    observation_frequency = rest_frequency.get(obs_line, obs_line)
    if not isinstance(observation_frequency, u.Quantity):
        raise ValueError(
            f"Invalid obs_line: {obs_line}. Expected one of {list(rest_frequency.keys())}. \n Enter the frequency (u.Quantity) directry."
        )

    velocity_array = get_vlsr(
        spec_array,
        freq_resolution,
        setting.multiplier[f"factor_{band_name}_lo"],
        freq_1st_lo,
        freq_2nd_lo,
        side_band,
        observation_frequency,
        config.location,
    )
    ds = make_dataset(spec_array, velocity_array)
    return ds


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
    )
    array = array.assign_coords({"lon_cor": ("t", lon_lat.az.value)})
    array = array.assign_coords({"lat_cor": ("t", lon_lat.alt.value)})
    radec_lat = SkyCoord(
        lon_list, lat_list, frame="altaz", obstime=obstime, location=location
    ).transform_to("icrs")
    array = array.assign_coords({"ra_cor": ("t", radec_lat.ra.value)})
    array = array.assign_coords({"dec_cor": ("t", radec_lat.dec.value)})
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
        array = array.assign_coords({"lon_recor": ("t", coords.az.value)})
        array = array.assign_coords({"lat_recor": ("t", coords.alt.value)})
    return array
