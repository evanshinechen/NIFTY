"""Convert and resample spectral flux densities."""

import numpy as np
import numpy.typing as npt
from spectres import spectres

__all__ = ["flambda_to_fnu", "fnu_to_flambda", "resample_spectrum"]

_C_ANGSTROM_PER_SECOND = 2.99792458e18


def flambda_to_fnu(
    wave: npt.ArrayLike, flux: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Convert F_lambda to F_nu.

    Parameters
    ----------
    wave : ndarray of shape (..., N)
        Wavelength in Angstroms. Broadcasts against flux.
    flux : ndarray of shape (..., N)
        F_lambda in spectral flux density per Angstrom.

    Returns
    -------
    flux_fnu : ndarray of shape (..., N)
        F_nu in spectral flux density per Hz.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    return flux * np.square(wave) / _C_ANGSTROM_PER_SECOND


def fnu_to_flambda(
    wave: npt.ArrayLike, flux: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Convert F_nu to F_lambda.

    Parameters
    ----------
    wave : ndarray of shape (..., N)
        Wavelength in Angstroms. Broadcasts against flux.
    flux : ndarray of shape (..., N)
        F_nu in spectral flux density per Hz.

    Returns
    -------
    flux_flambda : ndarray of shape (..., N)
        F_lambda in spectral flux density per Angstrom.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    return flux * _C_ANGSTROM_PER_SECOND / np.square(wave)


def resample_spectrum(
    new_wave: npt.ArrayLike,
    wave: npt.ArrayLike,
    flux: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Resample F_lambda while conserving integrated flux.

    Parameters
    ----------
    new_wave : ndarray of shape (M,)
        Target wavelength grid in Angstroms.
    wave : ndarray of shape (N,)
        Source wavelength grid in Angstroms.
    flux : ndarray of shape (..., N)
        F_lambda in spectral flux density per Angstrom.

    Returns
    -------
    resampled : ndarray of shape (..., M)
        Resampled F_lambda in spectral flux density per Angstrom. Bins outside
        the source wavelength coverage contain NaN.

    Notes
    -----
    Both wavelength grids must have at least two strictly increasing samples.
    A target bin must fit entirely within the source bin edges to receive a
    resampled value.
    """
    return spectres(
        np.asarray(new_wave),
        np.asarray(wave),
        np.asarray(flux),
        fill=np.nan,
        verbose=False,
    )
