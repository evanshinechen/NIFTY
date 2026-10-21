"""NIFTY-style plots."""

import corner
import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .model import IndexMap, linear_teff
from .sample import PosteriorSamples

__all__ = ["plot_corner", "plot_photometry", "plot_spectrum"]

LABELS = {
    "teff": r"$T_{\mathrm{eff}}/\mathrm{K}$",
    "logg": r"$\mathrm{log}_{10} \left( g \right)$",
    "kzz": r"$\mathrm{log}_{10} \left( K_{\mathrm{zz}} \right)$",
    "mh": r"[M/H]",
    "co": r"C/O",
    "d": r"$d/\mathrm{pc}$",
}


def _get_axis_label(key: str) -> str:
    """Get the plot label for an axis from a key.

    If the key is not known, the key itself is used as the label.
    """
    return LABELS.get(key, key)


def plot_corner(
    fig: Figure, samples: PosteriorSamples, *, axes_map: IndexMap
) -> Figure:
    """Create a corner plot of samples.

    Parameters
    ----------
    fig : Figure
        Figure to create corner plot.
    samples : PosteriorSamples
        Samples of the posterior. Can be weighted or unweighted.
    axes_map : IndexMap
        Map of key to index for axis labels.

    Returns
    ------
    fig : Figure
        Figure with corner plot.
    """
    labels = [_get_axis_label(key) for key in axes_map.keys()]
    corner.corner(
        data=linear_teff(samples.samples, axes_map),
        weights=samples.weights,
        labels=labels,
        show_titles=True,
        bins=40,
        levels=[0.68, 0.95, 0.99],
        quantiles=[0.16, 0.50, 0.84],
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 15},
        plot_datapoints=False,
        scale_hist=False,
        smooth1d=False,
        smooth=True,
        labelpad=0,
        lw=3,
        hist_kwargs={"lw": 2},
        contourf_kwargs={"lw": 2},
        fig=fig,
    )
    for ax in fig.axes:
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            bottom=True,
            top=False,
            left=True,
            right=False,
            length=4.5,
            width=2,
            labelsize=12,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            bottom=True,
            top=False,
            left=True,
            right=False,
            length=3.0,
            width=2,
            labelsize=12,
        )
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
    return fig


def plot_photometry(
    ax: Axes,
    *,
    central_wave: npt.NDArray[np.float32],
    obj_flux: npt.NDArray[np.float32],
    obj_error: npt.NDArray[np.float32],
    valid_flux: npt.NDArray[np.bool_],
    obj_id: int,
    model_phot_flux: npt.NDArray[np.float32],
    model_spec_wave: npt.NDArray[np.float32],
    model_spec_flux: npt.NDArray[np.float32],
    model_spec_flux_lower: npt.NDArray[np.float32],
    model_spec_flux_upper: npt.NDArray[np.float32],
    chi_sq_red: float,
) -> Axes:
    """Plot object photometry, model photometry, and model SED on axes.

    Parameters
    ----------
    ax : Axes
        Axes on which to plot.
    central_wave : ndarray
        Central wavelengths of filters in object photometry in microns.
    obj_flux : ndarray
        Object photometric flux.
    obj_error : ndarray
        Object photometric flux error.
    valid_flux : ndarray
        Boolean mask of valid object fluxes.
    obj_id : int
        Object ID number.
    model_phot_flux : ndarray
        Model photometric flux.
    model_spec_wave : ndarray
        Wavelength of model spectroscopy in Angstroms.
    model_spec_flux : ndarray
        Model spectroscopic flux.
    model_spec_flux_lower : ndarray
        Lower bound of 68% confidence interval for model spectroscopy.
    model_spec_flux_upper : ndarray
        Upper bound of 68% confidence interval for model spectroscopy.
    chi_sq_red : float
        Chi-square reduced value between observed and model photometry.

    Returns
    -------
    ax : Axes
        Axes with photometry plotted.
    """
    model_spec_wave = model_spec_wave / 1e4  # um
    valid_obj_flux = obj_flux[valid_flux]
    valid_central_wave = central_wave[valid_flux]
    ax.scatter(
        valid_central_wave,
        valid_obj_flux,
        s=40,
        color="black",
        alpha=1.0,
        zorder=15,
        label="Observed Photometry",
    )
    ax.errorbar(
        valid_central_wave,
        valid_obj_flux,
        yerr=obj_error[valid_flux],
        color="black",
        ls="None",
        alpha=0.8,
        zorder=14,
    )
    ax.scatter(
        central_wave,
        model_phot_flux,
        color="None",
        edgecolor="red",
        linewidth=3.0,
        alpha=0.9,
        marker="s",
        s=80,
        zorder=13,
        label=(
            r"Model Photometry, $\chi_{\mathrm{red}}^2$ = "
            + str(round(chi_sq_red, 2))
        ),
    )
    ax.fill_between(
        model_spec_wave,
        model_spec_flux_lower,
        model_spec_flux_upper,
        step="mid",
        color="red",
        alpha=0.1,
        label=r"68\% Confidence"
        if matplotlib.rcParams["text.usetex"]
        else "68% Confidence",
        zorder=11,
    )
    ax.step(
        model_spec_wave,
        model_spec_flux,
        color="red",
        alpha=0.4,
        label="Model Flux",
        zorder=12,
    )
    ax.set_title(f"Source ID {obj_id}", fontsize=15)
    # Give some extra margin.
    xlim_min = 0.9 * np.min(central_wave)
    xlim_max = 1.1 * np.max(central_wave)
    ax.set_xlim(xlim_min, xlim_max)
    if valid_obj_flux.size == 0:
        ymin = 1
        ymax = 11
    else:
        ymin = max(np.min(valid_obj_flux) / 10.0, 1e-2)
        ymax = 11 * np.max(valid_obj_flux)
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel(r"F$_{\nu}$ / (nJy) ")
    ax.legend()
    return ax


def plot_spectrum(
    ax: Axes,
    *,
    wave: npt.NDArray[np.float32],
    obj_flux: npt.NDArray[np.float32],
    obj_error: npt.NDArray[np.float32],
    valid_flux: npt.NDArray[np.bool_],
    obj_id: int,
    model_wave: npt.NDArray[np.float32],
    model_flux: npt.NDArray[np.float32],
    model_flux_lower: npt.NDArray[np.float32],
    model_flux_upper: npt.NDArray[np.float32],
    chi_sq_red: float,
) -> Axes:
    """Plot object spectroscopy and model SED on axes.

    Parameters
    ----------
    ax : Axes
        Axes on which to plot.
    wave: ndarray
        Wavelength of object spectroscopy in Angstroms.
    obj_flux : ndarray
        Object spectroscopic flux.
    obj_error : ndarray
        Object spectroscopic flux error.
    valid_flux : ndarray
        Boolean mask of object fluxes to display.
    obj_id : int
        Object ID number.
    model_wave : ndarray
        Wavelength of model spectroscopy in Angstroms.
    model_flux : ndarray
        Model spectroscopic flux.
    model_flux_lower : ndarray
        Lower bound of 68% confidence interval for model spectroscopy.
    model_flux_upper : ndarray
        Upper bound of 68% confidence interval for model spectroscopy.
    chi_sq_red : float
        Chi-square reduced value between observed and model spectroscopy.

    Returns
    -------
    ax : Axes
        Axes with spectroscopy plotted.
    """
    wave = wave / 1e4  # um
    model_wave = model_wave / 1e4  # um
    valid_error = valid_flux & np.isfinite(obj_error) & (obj_error > 0)
    ax.fill_between(
        wave,
        np.where(valid_error, obj_flux - obj_error, np.nan),
        np.where(valid_error, obj_flux + obj_error, np.nan),
        where=valid_error,
        step="mid",
        color="black",
        alpha=0.2,
        label=r"Observed 68\% Confidence"
        if matplotlib.rcParams["text.usetex"]
        else "Observed 68% Confidence",
        zorder=14,
    )
    ax.step(
        wave,
        np.where(valid_flux, obj_flux, np.nan),
        where="mid",
        color="black",
        alpha=0.5,
        label="Observed Spectrum",
        zorder=15,
    )
    ax.fill_between(
        model_wave,
        model_flux_lower,
        model_flux_upper,
        step="mid",
        color="red",
        alpha=0.5,
        label=r"Model 68\% Confidence"
        if matplotlib.rcParams["text.usetex"]
        else "Model 68% Confidence",
        zorder=17,
    )
    ax.step(
        model_wave,
        model_flux,
        where="mid",
        color="red",
        alpha=1.0,
        label=(
            "Model Flux, $\\chi_{\\mathrm{red}}^2$ = "
            + str(round(chi_sq_red, 2))
        ),
        zorder=18,
    )

    ax.set_title(f"Source ID {obj_id}", fontsize=15)
    xlim_min, xlim_max = 0.8, 5.2
    ax.set_xlim(xlim_min, xlim_max)
    observed_in_view = (wave >= xlim_min) & (wave <= xlim_max)
    model_in_view = (model_wave >= xlim_min) & (model_wave <= xlim_max)
    visible_flux = valid_flux & observed_in_view
    visible_error = valid_error & observed_in_view
    shown_flux = np.concatenate(
        (
            obj_flux[visible_flux],
            (obj_flux - obj_error)[visible_error],
            (obj_flux + obj_error)[visible_error],
            model_flux_lower[model_in_view],
            model_flux_upper[model_in_view],
            np.array([0.0]),
        )
    )
    ylim_min = float(np.min(shown_flux))
    ylim_max = float(np.max(shown_flux))
    padding = 0.1 * (ylim_max - ylim_min) or 0.1
    ylim_min -= padding
    ylim_max += padding
    ax.plot(
        [xlim_min, xlim_max],
        [0, 0],
        ls="--",
        color="black",
        alpha=0.3,
        zorder=1,
    )
    ax.set_ylim(float(ylim_min), float(ylim_max))
    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel(r"F$_{\nu}$ / (nJy) ")
    ax.legend()
    return ax
