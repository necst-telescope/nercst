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
    """

    def __init__(self, data_array: xr.DataArray):
        self.position_dtype = data_array.position.dtype.itemsize
        self.data_array = data_array

    def classify_data(self):
        pos = "".ljust(self.position_dtype)
        data_list = []
        mean_list = []
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
                    el_mean_list.append(np.mean(el_list))
                else:
                    mean_list.append(np.nan)
                    el_mean_list.append(np.nan)
                pos = one_data.position.values.item()
                data_list = [one_data.median(dim="ch").values]
                el_list = [one_data.lat]
        return position_list, mean_list, el_mean_list

    def calc_plot(self):
        position_list, mean_list, el_mean_list = self.classify_data()
        secz_list = []
        term_list = []
        for i, pos in enumerate(position_list):
            if pos == b"HOT".ljust(self.position_dtype):
                hot = mean_list[i]
            elif pos == b"SKY".ljust(self.position_dtype):
                sky = mean_list[i]
                if sky / hot < 1:
                    term_list.append(np.log(1 - sky / hot))
                else:
                    term_list.append(np.nan)
                z = (90 - el_mean_list[i]) * np.pi / 180.0
                secz_list.append(1 / np.cos(z))
        return secz_list, term_list

    def line_fit(self):
        secz_list, term_list = self.calc_plot()
        tau, intercept = np.polyfit(secz_list, term_list, 1)
        return tau, intercept

    def plot(self, fig: plt.figure = None, ax: plt.axes = None, topicname: str = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        secz_list, term_list = self.calc_plot()
        ax.scatter(secz_list, term_list, marker=".")
        tau, intercept = self.line_fit()
        xlims = ax.get_xlim()
        ax.plot(list(xlims), [tau * xlims[0] + intercept, tau * xlims[1] + intercept])
        ax.set_xlabel("sec Z", size=20)
        ax.set_ylabel("log(1-sky/hot)", size=20)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid()
        tau_str = str(round(abs(tau), 3))
        ax.text(
            xlims[0],
            np.nanmin(term_list),
            f"tau = {tau_str}",
            size=25,
        )
        if topicname is not None:
            ax.set_title(topicname, fontsize=20)
