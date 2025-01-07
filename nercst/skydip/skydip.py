import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


class Skydip:
    """
    Analysis of skydip to get optical thickness.

    Parameters
    ----------
    data_array: xr.DataArray
        data array of one topic

    Examples
    --------
    >>> Skydip = skydip.Skydip(data array)
    >>> Skydip.plot()
    (Show result)
    >>> tau, _ = Skydip.line_fit
    (get tau)
    """

    def __init__(self, data_array: xr.DataArray):
        self.position_dtype = data_array.position.dtype.itemsize
        self.data_array = data_array

    def classify_data(self):
        pos = "".ljust(self.position_dtype)
        data_list = []
        mean_list = []
        std_list = []
        el_mean_list = []
        el_list = []
        position_list = []
        for i, one_data in enumerate(self.data_array):
            if pos == one_data.position:
                data_list.append(one_data.median(dim="ch").values)
                el_list.append(one_data.lat)
            else:
                position_list.append(pos)
                if len(data_list) > 0:
                    mean_list.append(np.mean(data_list))
                    std_list.append(np.nanstd(data_list))
                    el_mean_list.append(np.mean(el_list))
                else:
                    mean_list.append(np.nan)
                    std_list.append(np.nan)
                    el_mean_list.append(np.nan)
                pos = one_data.position.values.item()
                data_list = [one_data.median(dim="ch").values]
                el_list = [one_data.lat]
        return position_list, mean_list, std_list, el_mean_list

    def calc_plot(self):
        position_list, mean_list, std_list, el_mean_list = self.classify_data()
        secz_list = []
        term_list = []
        y_err_list = []
        for i, pos in enumerate(position_list):
            if pos == b"HOT".ljust(self.position_dtype):
                hot = mean_list[i]
                hot_std = std_list[i]
            elif pos == b"SKY".ljust(self.position_dtype):
                sky = mean_list[i]
                sky_std = std_list[i]
                if hot - sky > 0:
                    term_list.append(np.log(hot - sky))
                    y_err_list.append(np.sqrt(hot_std**2 + sky_std**2) / (hot - sky))
                else:
                    term_list.append(np.nan)
                    y_err_list.append(np.nan)
                z = (90 - el_mean_list[i]) * np.pi / 180.0
                secz_list.append(1 / np.cos(z))
        return secz_list, term_list, y_err_list

    @property
    def line_fit(self):
        secz_list, term_list, y_err_list = self.calc_plot()
        tau, intercept = np.polyfit(
            secz_list, term_list, 1, w=[1 / e**2 for e in y_err_list]
        )
        return tau, intercept

    def plot(self, ax: plt.axes = None, title: str = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        secz_list, term_list, y_err_list = self.calc_plot()
        ax.scatter(secz_list, term_list, marker=".")
        tau, intercept = self.line_fit
        xlims = ax.get_xlim()
        ax.plot(list(xlims), [tau * xlims[0] + intercept, tau * xlims[1] + intercept])
        ax.errorbar(
            secz_list,
            term_list,
            yerr=y_err_list,
            capsize=4,
            fmt="o",
            ecolor="k",
            markeredgecolor="k",
            color="w",
        )
        ax.set_xlabel("sec Z", size=20)
        ax.set_ylabel("log(hot-sky)", size=20)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid()
        tau_str = str(round(abs(tau), 3))
        ax.text(
            xlims[0],
            np.nanmin(term_list),
            r"$\tau = $" + f"{tau_str}",
            size=25,
        )
        if title is not None:
            ax.set_title(title, fontsize=20)
        return ax
