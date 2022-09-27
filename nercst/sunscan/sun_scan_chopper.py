#!/usr/bin/env python3

from pathlib import Path
import pickle

from astropy.coordinates import get_body, AltAz
from astropy.time import Time
import astropy.constants as const
import astropy.units as u
import xarray as xr
import numpy as np

# import constants as n2const
import n_const.constants as n2const
from io_extension import get_tp_data


class ChopperSun:
    TOPIC_PREFIX = {
        "100G": np.array(["xffts_power_board"]),
        "200G": np.array(["ac240_tp_data_"]),
    }

    def __init__(
        self, path_to_data, path_to_kisa, topic_name, body="sun", obs_freq=None
    ):
        if not obs_freq:
            self.obs_freq = self.board2beam(topic_name)
        else:
            self.obs_freq = obs_freq
        self.body = body
        self.data = get_tp_data(
            path_to_data, topic_name, path_to_kisa, weather=True
        ).assign_attrs({"obs_freq": self.obs_freq})
        return

    def get_target_coord(self):
        target = get_body(body=self.body, time=Time(self.data.t))
        args = {
            "obstime": Time(self.data.t),
            "pressure": self.data["press"].mean().data * u.hPa,
            # "temperature": (self.data["out_temp"].mean().data * u.K).to(
            #     u.deg_C, equivalencies=u.equivalencies.temperature()
            # ),
            # "relative_humidity": self.data["out_humi"].mean().data * u.percent,
            "temperature": 15 * u.deg_C,  # TODO: give as kwargs?
            "relative_humidity": 20 * u.percent,  # TODO: give as kwargs?
            "location": n2const.LOC_NANTEN2,
        }
        if self.obs_freq == "100G":
            obswl = (const.c / (n2const.REST_FREQ.j10_12co)).to("micron")
        elif self.obs_freq == "200G":
            obswl = (const.c / (n2const.REST_FREQ.j21_12co)).to("micron")
        else:
            raise ValueError("`obs_freq` must be '100G' or '200G'")
        self.target = target.transform_to(
            AltAz(
                obswl=obswl,
                **args,
            )
        )
        return self.target

    def get_relative_position(self):
        """
        Relative position from center of the target.

        Notes
        -----
        Needed in ``sun_scan_beampattern``.
        """
        self.get_target_coord()
        # relative coordinate from center of the target
        res = {
            "resAz": self.target.az.deg - self.data["azlist"],
            "resEl": self.target.alt.deg - self.data["ellist"],
        }
        self.data = self.data.assign_coords(res)
        return

    @classmethod
    def board2beam(cls, topic_name):
        """
        Guess observation frequency from topic name.
        """
        for beam, spectrometer in cls.TOPIC_PREFIX.items():
            if topic_name.startswith(spectrometer[0]):
                return beam
        raise ValueError("Unknown topic name")

    def chopper_wheel(self):
        self.get_relative_position()
        tp_hot_mean, tp_off_mean, *_ = self.data.groupby("obsmode").mean(
            keep_attrs=True
        )
        tp_on = self.data.where(self.data.obsmode == 2, drop=True)
        with xr.set_options(keep_attrs=True):
            # Ta* = [P(ON) - P(OFF)] / [P(HOT) - P(OFF)] * Tamb
            t_a = (tp_on - tp_off_mean) / (tp_hot_mean - tp_off_mean) * 300
        return t_a


IF_LIST = {
    "100G": np.array(
        [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
        ]
    ),
    "200G": np.array(["1", "2", "3", "4"]),
}

TP_TOPICS = np.concatenate(
    (
        np.char.add(ChopperSun.TOPIC_PREFIX["100G"], IF_LIST["100G"]),
        np.char.add(ChopperSun.TOPIC_PREFIX["200G"], IF_LIST["200G"]),
    )
)  # ac240_tp_data_1, ac240_tp_data_2, xffts_power_board01, ..., xffts_power_board16


def create_chopperwheeled_data(
    path_to_data,
    path_to_kisa,
    save_dir="./",
    topic_name=TP_TOPICS,
    body="sun",
    obs_freq=None,
):
    """
    Notes
    -----
    Wall time: 30s / 18IF
    """
    # path_to_data = Path(path_to_data)
    save_dir = Path(save_dir).absolute()
    if path_to_data.name.find(body) == -1:  # filename doesn't contain target name
        return
    # loop through IFs
    t_a = {}
    if isinstance(topic_name, str):
        topic_name = [topic_name]
    for topic in topic_name:
        try:
            cs = ChopperSun(path_to_data, path_to_kisa, topic, body, obs_freq)
            t_a[topic] = cs.chopper_wheel()
        except KeyError:  # catch error from specifying all 20IF
            t_a[topic] = None
    # save
    filename = (save_dir / path_to_data.name).with_suffix(".pickle")
    print(f"Data file is at {filename}")
    with open(filename, "wb") as f:
        pickle.dump(t_a, f)
    return filename


if __name__ == "__main__":
    pass
