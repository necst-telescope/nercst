import astropy.units as u
import numpy as np
import xarray as xr
import re
from astropy.coordinates import EarthLocation

from neclib.coordinates import Observer, CoordCalculator
from neclib.core.files import toml
from neclib.core import RichParameters
import necstdb


def calc_vobs(obstime_array, location, target):

    az_list = []
    el_list = []
    obs = Observer(location)
    calc = CoordCalculator(location)
    name_coord = calc.name_coordinate(target)
    # print("Calculating Vobs...")
    for time in obstime_array:
        ret = name_coord.realize(time)
        az_list.append(ret.lon)
        el_list.append(ret.lat)

    print("Calculating Vobs...")
    v_obs = obs.v_obs(
        lon=az_list, lat=el_list, time=obstime_array, frame="altaz", unit="deg"
    )
    return v_obs


def convert_to_velocity(
    channel_array,
    freq_resolution,
    factor_1st_lo,
    freq_1st_lo,
    freq_2nd_lo,
    side_band,
    observation_frequency,
):

    print("Converting channel into velocity...")
    observed_GHz_array = channel_array * freq_resolution
    if side_band == "usb":
        pre_hd_array = (observed_GHz_array + freq_2nd_lo) + factor_1st_lo * freq_1st_lo
    elif side_band == "lsb":
        pre_hd_array = (-observed_GHz_array + freq_2nd_lo) + factor_1st_lo * freq_1st_lo
    freq_to_velocity_equiv = u.doppler_radio(observation_frequency)
    observed_v_array = pre_hd_array.to(
        u.km * u.s ** (-1), equivalencies=freq_to_velocity_equiv
    )

    return observed_v_array


def get_vlsr(
    data_array,
    freq_resolution,
    factor_1st_lo,
    freq_1st_lo,
    freq_2nd_lo,
    side_bnad,
    observation_frequency,
    target,
    location,
):
    channel_array = data_array.channel.data
    obstime_array = data_array.t.data
    observed_v_array = convert_to_velocity(
        channel_array,
        freq_resolution,
        factor_1st_lo,
        freq_1st_lo,
        freq_2nd_lo,
        side_bnad,
        observation_frequency,
    )
    v_obs_array = calc_vobs(obstime_array, location, target)
    observed_v_matrix = observed_v_array * np.ones(shape=data_array.shape)
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


def add_radial_velocity(data_array, dbname, topic_name):
    board_id = int(topic_name[-1])
    obs_filepath = data_array.attrs["obs_filepath"]
    config_filepath = data_array.attrs["config_filepath"]
    print(f"read obsfile from {obs_filepath}.")
    params = toml.read(obs_filepath)
    print(f"read config file from {config_filepath}.")
    config = RichParameters.from_file(config_filepath)
    config.attach_parsers(location=lambda x: EarthLocation(**x))
    target = " ".join(params["observation_property"]["target"].split("_"))
    freq_resolution = (
        config.spectrometer.bw_MHz[str(board_id)] * u.MHz / config.spectrometer.max_ch
    )
    sis_channel = config.sis_bias_setter.channel
    sis_channel_inv = {v: k for k, v in sis_channel.items()}
    side_band = sis_channel_inv[board_id].lower()
    freq_1st_lo, freq_2nd_lo = get_lo(dbname, side_band)
    velocity_array = get_vlsr(
        data_array,
        freq_resolution,
        config.multiplier.factor_1st_lo,
        freq_1st_lo,
        freq_2nd_lo,
        sis_channel_inv[board_id].lower(),
        config.observation_frequency,
        target,
        config.location,
    )
    ds = make_dataset(data_array, velocity_array)
    return ds
