import xarray as xr
import numpy as np
from typing import List


def basefit(calibrated_array, fitrange, deg):
    fit_array_list = [calibrated_array[:, s] for s in fitrange]
    fit_array = xr.concat(fit_array_list, dim="ch")
    fit = fit_array.polyfit(dim="ch", deg=deg, skipna=True)
    return fit


def base_subtraction(calibrated_array, fit):
    polyfit_coefficients = fit["polyfit_coefficients"]
    fit_order = len(polyfit_coefficients) - 1
    baseline = np.zeros_like(calibrated_array.values)
    for coefficient in polyfit_coefficients:
        if fit_order == 0:
            baseline = baseline + coefficient.values[..., None] * np.ones(32768)
            print("liner")
        else:
            baseline = baseline + coefficient.values[..., None] * (
                np.arange(32768) ** fit_order
            )
            print("nonzero")
        fit_order = fit_order - 1

    return calibrated_array - baseline


def apply_baseline_fitting(calibrated_array: xr.DataArray, fitrange: List[slice], deg):
    fit = basefit(calibrated_array, fitrange, deg)
    return base_subtraction(calibrated_array, fit)
