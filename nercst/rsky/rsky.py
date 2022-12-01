import xarray as xr
import matplotlib.pyplot as plt


class Rsky:
    """
    Analysis of R-sky to get Tsys.

    Parameters
    ----------
    data_array: xr.DataArray
        data array of one topic

    Examples
    --------
    >>> Rsky = rsky.Rsky(data array)
    >>> tsys_array = Rsky.tsys()
    >>> Rsky.plot()
    (Plot Hot, Sky and Tsys)
    """

    def __init__(self, data_array: xr.DataArray):
        self.hot_array = data_array.where(
            data_array["obs_mode"] == b"HOT       ", drop=True
        )
        self.sky_array = data_array.where(
            data_array["obs_mode"] == b"SKY       ", drop=True
        )

    def tsys(self) -> xr.DataArray:
        self.ave_hot_array = self.hot_array.mean(dim="t")
        self.ave_sky_array = self.sky_array.mean(dim="t")
        self.y_factor = self.ave_hot_array / self.ave_sky_array
        tsys_array = 300.0 / (self.y_factor - 1.0)
        self.tsys_array = tsys_array
        self.tsys_mean = str(round(self.tsys_array.data.mean(), 2))
        return tsys_array

    def plot(
        self,
        fig: plt.figure = None,
        ax: plt.axes = None,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax2 = ax.twinx()
        ax.plot(
            self.ave_hot_array["channel"].data,
            self.ave_hot_array.data,
            ".",
            color="red",
        )
        ax.plot(
            self.ave_sky_array["channel"].data,
            self.ave_sky_array.data,
            ".",
            color="blue",
        )
        ax2.plot(
            self.tsys_array["channel"].data,
            self.tsys_array.data,
            ".",
            color="green",
        )
        ax.text(
            self.ave_sky_array["channel"].data[-1] / 30,
            self.ave_sky_array.data.min(),
            f"Tsys = {self.tsys_mean}K",
            size=25,
        )
        ax.set_xlabel("channel", size=20)
        ax.set_ylabel("count", size=20)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_yscale("log")
        ax2.set_ylabel("Tsys [K]", size=20)
        ax2.grid()
        ax2.tick_params(axis="y", labelsize=16)
        ax2.set_ylim(0, 1400)
        ax.legend(["HOT", "SKY"], fontsize=16)
