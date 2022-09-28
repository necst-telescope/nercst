from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Union, List, Dict

import necstdb
import numpy as np
import xarray as xr


PathLike = Union[str, Path]
timestamp2datetime = np.vectorize(datetime.utcfromtimestamp)
AC240_TP_FIELD = "POWER_BE1"


@dataclass
class InitialArray(object):
    """Read raw data and make DataArray.

    Parameters
    ----------
    data_path: str or Path
        Path to data directory where raw data (.header and .data files)
        are saved.
    kisa_path: str or Path
        Path to *kisa* file.
    topic_name: str
        Name of ROS topic through which the data taken with specific
        spectrometer board is sent.
    Notes
    -----
    ROS topic for each board is named as follows:
    - Spectral data taken by XFFTS board xx : "xffts_boardxx" (0-padded)
    - Total power data taken by XFFTS board xx: "xffts_power_boardxx" (0-padded)
    - Spectral data taken by AC240 board x: "ac240_spactra_data_x"
    - Total power data taken by AC240 board x: "ac240_tp_data_x"
    Examples
    --------
    >>> data_path = "/path/to/data/directory"
    >>> kisa_path = "path/to/kisafile.txt"
    >>> telescope = "NANTEN2" or "OPU-1.85"
    >>> ia = InitialArray(data_path, "xffts_board01", kisa_path, telescope)
    """

    data_path: PathLike
    topic_name: str
    #     kisa_path: Optional[PathLike] = None
    telescope: str = "NANTEN2"

    def create_data_array(self) -> xr.Dataset:
        """Get spectral data."""
        if self.telescope == "NANTEN2":
            # read data
            db = necstdb.opendb(self.data_path)
            data = db.open_table(self.topic_name).read(astype="array")
            obsmode = db.open_table("obsmode").read(astype="array")
            encoder = db.open_table("status_encoder").read(astype="array")
            weather = db.open_table("status_weather").read(astype="array")

            # convert structured array into dict of DataArrays
            def convert(
                structured_array: np.ndarray, dims: List[str]
            ) -> Dict[str, xr.DataArray]:
                if not structured_array.dtype.names:  # when not structured
                    return {"total_power": xr.DataArray(structured_array, dims=dims)}
                ret = {}
                for field in structured_array.dtype.names:
                    dim = dims[: structured_array[field].ndim]
                    ret[field] = xr.DataArray(structured_array[field], dims=dim)
                return ret

            data = convert(data, ["t", "spectra"])
            obsmode = convert(obsmode, ["t"])
            encoder = convert(encoder, ["t"])
            weather = convert(weather, ["t"])

            # change time format
            data["timestamp"] = timestamp2datetime(data["timestamp"].astype(float))
            obsmode["received_time"] = timestamp2datetime(
                obsmode["received_time"].astype(float)
            )
            encoder["timestamp"] = timestamp2datetime(
                encoder["timestamp"].astype(float)
            )
            weather["timestamp"] = timestamp2datetime(
                weather["timestamp"].astype(float)
            )

            # create Dataset
            self.data_set = (
                xr.Dataset(data)
                .set_index({"t": "timestamp"})
                .drop_vars("received_time")
            )
            self.obsmode_set = xr.Dataset(obsmode).set_index({"t": "received_time"})
            self.encoder_set = (
                xr.Dataset(encoder)
                .set_index({"t": "timestamp"})
                .drop_vars("received_time")
            ) / 3600
            self.weather_set = (
                xr.Dataset(weather)
                .set_index({"t": "timestamp"})
                .drop_vars("received_time")
            )
            return self.data_set, self.obsmode_set, self.encoder_set, self.weather_set

        elif self.telescope == "OPU-1.85":
            # read data
            db = necstdb.opendb(self.data_path)
            data = db.open_table(self.topic_name).read(astype="array")
            obsmode = db.open_table("otf-status").read(astype="array")
            encoder_az = db.open_table("dev-ND287-ttyUSB0-az").read(astype="array")
            encoder_el = db.open_table("dev-ND287-ttyUSB1-el").read(astype="array")

            encoder_az = xr.DataArray(
                data=encoder_az["data"],
                dims=("t"),
                coords={"t": encoder_az["timestamp"]},
                name="enc_az",
            )
            encoder_el = xr.DataArray(
                data=encoder_el["data"],
                dims=("t"),
                coords={"t": encoder_el["timestamp"]},
                name="enc_el",
            )
            encoder_az = encoder_az.interp(t=data["timestamp"])
            encoder_el = encoder_el.interp(t=data["timestamp"])
            default_weather = [
                (0, 0, 0, 0, 0, 0, 850, 0, 290, 290, 290, 290, 0, 0)
            ] * len(data["timestamp"])
            weather = np.array(
                default_weather,
                dtype=[
                    ("in_temp", "<f8"),
                    ("out_temp", "<f8"),
                    ("in_humi", "<f8"),
                    ("out_humi", "<f8"),
                    ("wind_sp", "<f8"),
                    ("wind_dir", "<f8"),
                    ("press", "<f8"),
                    ("rain", "<f8"),
                    ("cabin_temp1", "<f8"),
                    ("cabin_temp2", "<f8"),
                    ("dome_temp1", "<f8"),
                    ("dome_temp2", "<f8"),
                    ("gen_temp1", "<f8"),
                    ("gen_temp2", "<f8"),
                ],
            )

            # convert structured array into dict of DataArrays
            def convert(
                structured_array: np.ndarray, dims: List[str]
            ) -> Dict[str, xr.DataArray]:
                if not structured_array.dtype.names:  # when not structured
                    return {"total_power": xr.DataArray(structured_array, dims=dims)}
                ret = {}
                for field in structured_array.dtype.names:
                    dim = dims[: structured_array[field].ndim]
                    ret[field] = xr.DataArray(structured_array[field], dims=dim)
                return ret

            data = convert(data, ["t", "spectra"])
            obsmode = convert(obsmode, ["t"])
            weather = convert(weather, ["t"])

            # create Dataset
            self.data_set = xr.Dataset(data).set_index({"t": "timestamp"})
            self.obsmode_set = xr.Dataset(obsmode).set_index({"t": "timestamp"})
            self.encoder_set = xr.merge(
                [
                    encoder_az.to_dataset(name="enc_az"),
                    encoder_el.to_dataset(name="enc_el"),
                ],
                compat="equals",
            )
            self.weather_set = xr.Dataset(weather, coords={"t": data["timestamp"]})

            # change time format

            self.data_set["t"] = timestamp2datetime(self.data_set["t"].astype(float))
            self.obsmode_set["t"] = timestamp2datetime(
                self.obsmode_set["t"].astype(float)
            )
            self.encoder_set["t"] = timestamp2datetime(
                self.encoder_set["t"].astype(float)
            )
            self.weather_set["t"] = timestamp2datetime(
                self.weather_set["t"].astype(float)
            )

            return self.data_set, self.obsmode_set, self.encoder_set, self.weather_set
