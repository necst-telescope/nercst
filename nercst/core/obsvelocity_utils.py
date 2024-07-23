import astropy.units as u
import numpy as np
import xarray as xr
import re
from astropy.coordinates import EarthLocation
import logging

from neclib.coordinates import Observer
from neclib.core import RichParameters
import necstdb

logger = logging.getLogger("necst")
logger.setLevel(logging.DEBUG)
st_handler = logging.StreamHandler()
logger.addHandler(st_handler)


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
    freq_to_velocity_equiv = u.doppler_radio(observation_frequency)
    observed_v_array = pre_hd_array.to(
        u.km * u.s ** (-1), equivalencies=freq_to_velocity_equiv
    )

    return observed_v_array


def get_vlsr(
    spec_array,
    freq_resolution,
    factor_1st_lo,
    freq_1st_lo,
    freq_2nd_lo,
    side_bnad,
    observation_frequency,
    location,
):
    channel_integer_numbers = spec_array.channel.data
    observed_v_array = convert_to_velocity(
        channel_integer_numbers,
        freq_resolution,
        factor_1st_lo,
        freq_1st_lo,
        freq_2nd_lo,
        side_bnad,
        observation_frequency,
    )
    v_obs_array = calc_vobs(
        spec_array.lon_cor.data, spec_array.lat_cor.data, spec_array.t.data, location
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


def get_lo(dbname, side_band):
    db = necstdb.opendb(dbname)
    topic_list = list(
        filter(lambda x: re.search("lo_signal.*1st", x), db.list_tables())
    )
    data_lo_1st = db.open_table(topic_list[0]).read(astype="array")
    freq_1st_lo = data_lo_1st["freq"][0] * u.GHz
    topic_signal = topic_list[0].replace("1st", "")
    data_lo_2nd = db.open_table(f"{topic_signal}{side_band}_2nd").read(astype="array")
    freq_2nd_lo = data_lo_2nd["freq"][0] * u.GHz
    return freq_1st_lo, freq_2nd_lo


def add_radial_velocity(spec_array, dbname, topic_name):
    board_id = int(topic_name[-1])
    config_filepath = spec_array.attrs["config_filepath"]
    device_setting_filepath = spec_array.attrs["device_setting_path"]
    logger.info(f"read config file from {config_filepath}.")
    config = RichParameters.from_file(config_filepath)
    logger.info(f"read device setting file from {device_setting_filepath}.")
    setting = RichParameters.from_file(device_setting_filepath)
    config.attach_parsers(location=lambda x: EarthLocation(**x))
    freq_resolution = (
        config.spectrometer.bw_MHz[str(board_id)] * u.MHz / config.spectrometer.max_ch
    )
    freq_1st_lo, freq_2nd_lo = get_lo(
        dbname, setting.spectrometer.side_band[str(board_id)]
    )
    velocity_array = get_vlsr(
        spec_array,
        freq_resolution,
        setting.multiplier.factor_1st_lo,
        freq_1st_lo,
        freq_2nd_lo,
        setting.spectrometer.side_band[str(board_id)],
        config.observation_frequency,
        config.location,
    )
    ds = make_dataset(spec_array, velocity_array)
    return ds
