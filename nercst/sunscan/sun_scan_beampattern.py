#!/usr/bin/env python3

import pickle
from pathlib import Path

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class BeampatternSun:
    """
    Parameters
    ----------
    data: ``xr.DataArray``
    scan_direction: ``str``
        <'az'|'el'>, [OPTIONAL]
    """

    def __init__(self, data=None, scan_direction=None):
        if scan_direction:
            scan_direction = scan_direction.lower()
        self.scan_direction = scan_direction
        if data is not None:
            self.data = data
            self.get_beam_size()
        return

    @staticmethod
    def gaussian(x, sigma, mu, height=1):
        return height * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def split_data(self):
        # determine which coordinate the scan is done along
        if not self.scan_direction:
            self.scan_direction = self.__guess_scan_direction()
        # split data as initial guess
        data_fit = {}
        data_fit["in"] = self.data_derivative.where(
            self.data_derivative[f"res_{self.scan_direction}"] < 0, drop=True
        )
        data_fit["out"] = self.data_derivative.where(
            self.data_derivative[f"res_{self.scan_direction}"] > 0, drop=True
        )
        # fix scan order from chronological consistency
        if data_fit["in"].received_time.mean() > data_fit["out"].received_time.mean():
            data_fit["in"], data_fit["out"] = data_fit["out"], data_fit["in"]
        return data_fit

    def get_beam_size(self):
        """Beamsize (HPBW) as FWHM of gaussian."""
        el = self.data.ellist
        self.data = self.data.assign_coords(
            {
                "res_x": self.data.resAz * np.cos(np.deg2rad(el)),
                "res_y": self.data.resEl,
            }
        )
        self.data_derivative = self.data.differentiate("t").assign_attrs(
            self.data.attrs
        )
        # method ``differentiate`` has no option ``keep_attr``
        # ``xr.set_options(keep_attrs=True)`` doesn't work for this

        # split the data into into-sun-scan and out-of-sun-scan
        self.data_fit = self.split_data()

        # fitting
        popt, pcov = {}, {}
        fwhm = {}
        # EMPIRICAL
        p0 = {
            "in": [1, 0, 1],
            "out": [1, -0, -1],
        }  # best-estimate won't work (reason unknown)

        for edge in ["in", "out"]:
            popt[edge], pcov[edge] = self.curve_fitting(
                self.gaussian,
                self.data_fit[edge][f"res_{self.scan_direction}"],
                self.data_fit[edge],
                p0=p0[edge],
            )
            fwhm[edge] = abs(
                (2 * np.sqrt(2 * np.log(2)) * popt[edge][0] * u.deg).to(u.arcsec)
            )

        self.popt, self.pcov = popt, pcov
        self.fwhm = fwhm
        return fwhm

    def __guess_scan_direction(self):
        # scan speed in (x, y) direction
        # ideally, v_x = 0 or v_y = 0
        scan_v_x = abs(self.data_derivative.res_x.differentiate("t").mean())
        scan_v_y = abs(self.data_derivative.res_y.differentiate("t").mean())

        if scan_v_x > scan_v_y:
            return "x"
        elif scan_v_x < scan_v_y:
            return "y"
        raise ValueError(
            "Couldn't determine scan direction." "Specify ``scan_direction``."
        )

    @staticmethod
    def curve_fitting(func, x, y, p0, **kwargs):
        # EMPIRICAL
        threshold = 1e5  # since the data have quite large variance
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, **kwargs)
            if np.any(pcov > threshold):
                popt = [np.nan] * 3
        except RuntimeError:
            popt = [np.nan] * 3
            pcov = None
        return popt, pcov

    def draw_beampattern(self, data_fit, edges=["in", "out"], ax=None, **kwargs):
        """
        Parameters
        ----------
        edges: ``list`` of ``str``
            <'in'|'out'>
        **kwargs: Parameters to ``plt.lines.Line2D``
        """
        color = {
            "fit_in": "#F79",
            "fit_out": "#F46",
            "data_in": "#6AF",
            "data_out": "#37F",
        }
        for edge in edges:
            x = data_fit[edge][f"res_{data_fit[edge].scan_direction}"]
            fit = self.gaussian(x, *data_fit[edge].popt)
            # plot fit curve
            if ax:
                ax.plot(x, fit, c=color[f"fit_{edge}"], lw=4)
            else:
                plt.plot(x, fit, c=color[f"fit_{edge}"], lw=4)
            # plot actual data
            data_fit[edge].plot(
                x=f"res_{data_fit[edge].scan_direction}",
                ax=ax,
                c=color[f"data_{edge}"],
                **kwargs,
            )

        if ax:
            ax.set_ylabel("count")
        else:
            plt.ylabel("count")

        plt.grid()
        return

    def get_center(self):
        center = (self.popt["in"][1] + self.popt["out"][1]) / 2 * u.deg
        return (f"res_{self.scan_direction}", center.to(u.arcmin))

    def get_sun_size(self):
        size = abs(self.popt["in"][1] - self.popt["out"][1]) * u.deg
        return size.to(u.arcmin)

    def get_fitting_data(self):
        parameters = {
            "sun_size": self.get_sun_size(),
            "beam_squint": self.get_center(),
            "scan_direction": self.scan_direction,
        }
        for edge in ["in", "out"]:
            self.data_fit[edge] = self.data_fit[edge].assign_attrs(
                {
                    "HPBW": self.fwhm[edge],
                    "popt": self.popt[edge],
                    "pcov": self.pcov[edge],
                    **parameters,
                }
            )
        return self.data_fit


BOARD2BEAM = {
    "ac240_tp_data_1": "Beam1 - LL",
    "ac240_tp_data_2": "Beam1 - LU",
    "ac240_tp_data_3": "Beam1 - RL",
    "ac240_tp_data_4": "Beam1 - RU",
    "xffts_power_board01": "Beam2 - LU",
    "xffts_power_board02": "Beam2 - LL",
    "xffts_power_board03": "Beam2 - RU",
    "xffts_power_board04": "Beam2 - RL",
    "xffts_power_board05": "Beam3 - LU",
    "xffts_power_board06": "Beam3 - LL",
    "xffts_power_board07": "Beam3 - RU",
    "xffts_power_board08": "Beam3 - RL",
    "xffts_power_board09": "Beam4 - LU",
    "xffts_power_board10": "Beam4 - LL",
    "xffts_power_board11": "Beam4 - RU",
    "xffts_power_board12": "Beam4 - RL",
    "xffts_power_board13": "Beam5 - LU",
    "xffts_power_board14": "Beam5 - LL",
    "xffts_power_board15": "Beam5 - RU",
    "xffts_power_board16": "Beam5 - RL",
}


def create_fitting_data(pickle_path, save_dir="./"):
    """
    Notes
    -----
    Wall time: 7s / 18IF
    """
    with open(pickle_path, "rb") as f:
        chopperwheeled_data = pickle.load(f)
    observation = Path(pickle_path).stem
    fitting_data = {}
    for topic, data in chopperwheeled_data.items():
        bs = BeampatternSun(data)
        try:
            fitting_data[topic] = bs.get_fitting_data()
            scan_direction = bs.scan_direction
        except AttributeError:  # catch error from specifying all 20IF
            fitting_data[topic] = None

    save_dir = Path(save_dir).absolute()
    filename = (
        save_dir / f"{observation}_fitting_{scan_direction.upper()}scan"
    ).with_suffix(".pickle")
    print(f"Fitting data is at {filename}")
    with open(filename, "wb") as f:
        pickle.dump(fitting_data, f)
    return filename


def multi_axes_style(axes, nrows, ncols):
    """
    Remove duplicate ticks and align all subplots.

    Parameters
    ----------
    axes: ``np.ndarray`` of ``matplotlib.axes._subplots.AxesSubplot``
        from ``fig, axes = plt.subplots(nrows, ncols)``
    nrows: ``int``
    ncols: ``int``
    xlim: 2-``array-like`` of ``float``
    ylim: 2-``array-like`` of ``float``
    """
    for i, ax in enumerate(axes.flat):
        if not i % ncols == 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel(None)
        if not i // ncols == nrows - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel(None)
        ax.grid()
    return


def no_signal(axes, num_data, xlabel=None, reverse_grid=True):
    for i, ax in enumerate(axes.flat):
        if i >= num_data:
            ax.text(
                0.5,
                0.5,
                "No signal",
                fontsize=25,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(xlabel)
            if reverse_grid:
                ax.grid()  # to erase the grid out
    return


def draw_figure(pickle_path, save=False, save_dir="./"):
    """
    Notes
    -----
    Wall time: 3s / 18IF
    """
    observation = Path(pickle_path).stem
    with open(pickle_path, "rb") as f:
        fitting_data = pickle.load(f)
    if_num = len(fitting_data)
    nrows = int(np.ceil(if_num / 4))
    ncols = 4
    fig_width = 4 * ncols
    fig_height = 3 * nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey="row"
    )
    for ax, (topic, data) in zip(axes.flat, fitting_data.items()):
        bs = BeampatternSun()
        try:
            bs.draw_beampattern(data, ax=ax, lw=0, marker=".", ms=5)
            ax.text(
                0.98,
                0.04,
                f"HPBW = {data['in'].HPBW:.1f} (in)\n" f"{data['out'].HPBW:.1f} (out)",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="w", alpha=0.5, edgecolor="w"),
                transform=ax.transAxes,
            )
        except TypeError:  # catch error from specifying all 20IF
            if_num -= 1

        if topic in BOARD2BEAM:
            ax.set_title(BOARD2BEAM[topic])
        else:
            ax.set_title(topic)

    multi_axes_style(axes, nrows, ncols)
    no_signal(axes, if_num, xlabel=axes.flat[ncols * (nrows - 1)].get_xlabel())

    plt.suptitle(f"{observation}", y=0.99, size=20)
    plt.tight_layout()
    if save:
        save_dir = Path(save_dir).absolute()
        filename = save_dir / f"{observation}.pdf"
        plt.savefig(filename)
        print(f"Figure is at {filename}")
    return


if __name__ == "__main__":
    pass
