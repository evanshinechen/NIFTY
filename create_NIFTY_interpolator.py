#!/usr/bin/env python
"""\
create_NIFTY_interpolator.py
Kevin Hainline, Jake Helton

Builds the model grid interpolator pickle files used by NIFTY.py.
Run this once per model before running NIFTY.py.

Supported models
----------------
SonoraElfOwl    -- Sonora Elf Owl v2 (.nc files, temperature-range subdirectories)
SonoraElfOwlPH3 -- Sonora Elf Owl + PH3 disequilibrium (.npz grid file)
ATMO2020        -- ATMO 2020++ (.dat text files, per-metallicity subdirectories)
LOWZ            -- LOWZ (.txt text files, flat directory + CSV index)

Usage
-----
python create_NIFTY_interpolator.py \\
    -model  SonoraElfOwl \\
    -path   /path/to/model/files/ \\
    -config BD_NIRCam_MIRI_filters.json \\
    [-output Sonora_v2_interp.pkl]

The -path argument should point to:
    SonoraElfOwl   : directory containing output_<Tmin>_<Tmax>/ subdirs
    SonoraElfOwlPH3: directory containing elf_owl_disequilibrium_PH3.npz
    ATMO2020       : directory containing grid_<mh>/ subdirectories
    LOWZ           : directory containing models/ subdirectory and
                     LOWZ_models_index.csv

Output
------
A single .pkl file containing:
    T_eff             -- 1-D array of log10(Teff) grid values
    logg              -- 1-D array of log10(g [cgs]) grid values
    kzz               -- 1-D array of log10(Kzz) values  [Sonora/LOWZ only]
    mh                -- 1-D array of [M/H] values
    co                -- 1-D array of C/O values           [Sonora/LOWZ only]
    filters           -- list of filter name strings
    wave              -- 1-D wavelength array in Angstroms
    phot_interpolator -- RegularGridInterpolator for broadband fluxes (nJy)
    spec_interpolator -- RegularGridInterpolator for spectra (erg/s/cm^2/Ang)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import glob
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tarfile
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sedpy import observate
from tqdm import tqdm


# ============================================================
# Shared utilities
# ============================================================

def load_filters(config_path):
    """
    Load filter names and central wavelengths from the NIFTY JSON config file.

    Returns
    -------
    filter_name : list of str
    filter_central_wavelength : np.ndarray  (microns)
    filter_sedpy : list of sedpy Filter objects
    """
    print("Opening config/filters JSON: " + config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    filter_name = list(config["filter_columns"].keys())
    filter_central_wavelength = np.array([
        config["filter_columns"][fn]["wavelength"] for fn in filter_name
    ])

    print("    " + str(len(filter_name)) + " filters found:")
    line = "       " + filter_name[0]
    for i in range(1, len(filter_name)):
        if not i % 6:
            line += "\n       " + filter_name[i]
        else:
            line += ", " + filter_name[i]
    print(line)
    print(" - - - - - - - - ")

    # Build sedpy filter objects once — reused for every spectrum
    filter_sedpy = [observate.Filter("jwst_" + fn) for fn in filter_name]

    return filter_name, filter_central_wavelength, filter_sedpy


# Standard lower-resolution wavelength grid shared by all models (Angstroms)
LOWER_RES_WAVE = np.arange(0.75, 15, 0.01) * 1e4


def ab_mag_to_fnu_cgs(ab_mag):
    """Convert AB magnitude to f_nu in cgs (erg/s/cm^2/Hz)."""
    return 10.0 ** ((ab_mag + 48.60) / (-2.5))


def compute_filter_fluxes(wave_ang_sorted, flux_ergscm2ang_sorted, filter_sedpy):
    """
    Pass a single spectrum through all filters.

    Parameters
    ----------
    wave_ang_sorted : array, wavelength in Angstroms (must be sorted ascending)
    flux_ergscm2ang_sorted : array, f_lambda in erg/s/cm^2/Ang (sorted same)
    filter_sedpy : list of sedpy Filter objects

    Returns
    -------
    fluxes : np.ndarray, shape (n_filters,), f_nu in cgs
    """
    fluxes = np.empty(len(filter_sedpy), dtype=np.float32)
    for i, f in enumerate(filter_sedpy):
        ab_mag = f.ab_mag(wave_ang_sorted, flux_ergscm2ang_sorted)
        fluxes[i] = ab_mag_to_fnu_cgs(ab_mag)
    return fluxes


def resample_spectrum(wave_ang_sorted, flux_ergscm2ang_sorted):
    """
    Resample a model spectrum onto LOWER_RES_WAVE using np.interp.

    np.interp is used rather than spectres for speed.  For a pre-built
    interpolator grid evaluated at lower resolution, the difference in
    flux conservation is negligible compared with the grid interpolation
    error itself.

    Returns
    -------
    resampled : np.ndarray, shape (len(LOWER_RES_WAVE),), dtype float32
    """
    return np.interp(
        LOWER_RES_WAVE,
        wave_ang_sorted,
        flux_ergscm2ang_sorted
    ).astype(np.float32)


def keyify(*args, nd=8):
    """
    Make a hashable dict key from a set of float parameter values.
    Rounding to nd decimal places avoids floating-point equality surprises.
    """
    return tuple(round(float(a), nd) for a in args)


def save_pickle(output_values_data, output_filename):
    """Pickle the interpolator dict and report."""
    with open(output_filename, 'wb') as f:
        pickle.dump(output_values_data, f)
    print("Saved interpolator to: " + output_filename)


@dataclass
class ModelGrid:
     # Points on each axis
    points : tuple[np.ndarray[tuple[int], np.dtype[np.float32]], ...]
     # Photometry data grid
    phot : npt.NDArray[np.float32]
     # Spectroscopy data grid
    spec : npt.NDArray[np.float32]
     # Map axis name to index
    axes_map : dict[str, int]
     # Name of the model
    model_name : str
     # How holes were filled or None if not filled
    fill_method : str | None
    # List of filter names
    filters : list[str]
    # Sampled wave
    wave : np.ndarray[tuple[int], np.dtype[np.float32]]

def load_model(path):
    """
    Load a ModelGrid from a file.

    Parameters
    ----------
    path : str
        Path to the file to load.

    Returns
    -------
    model_grid : ModelGrid
        The ModelGrid loaded from the file.
    """
    with tarfile.open(path, mode='r|gz') as tar:
        meta_member = tar.next()
        if meta_member is None:
            raise ValueError("Tarfile is empty.")
        meta_file = tar.extractfile(meta_member)
        assert meta_file is not None
        meta = json.loads(meta_file.read().decode('utf-8'))
        arrays = {}
        while True:
            member = tar.next()
            if member is None:
                break
            name = Path(member.name).stem
            array_file = tar.extractfile(member)
            assert array_file is not None
            arrays[name] = np.load(io.BytesIO(array_file.read()))
        points = tuple(arrays[name] for name in meta['axes'])
        axes_map = {name: i for i, name in enumerate(meta['axes'])}
    return ModelGrid(
        points=points,
        phot=arrays['phot'],
        spec=arrays['spec'],
        axes_map=axes_map,
        model_name=meta['model_name'],
        fill_method=meta['fill_method'],
        filters=meta['filters'],
        wave=arrays['wave'],
    )

def save_model(model_grid: ModelGrid, path):
    """
    Save a ModelGrid to a file.

    Parameters
    ----------
    model_grid : ModelGrid
        The ModelGrid to save.
    path : str
        The path to save the ModelGrid.
    """
    def _write_meta(meta):
        buffer = io.BytesIO(json.dumps(meta).encode('utf-8'))
        return buffer.getvalue()

    def _write_array(obj):
        buffer = io.BytesIO()
        np.save(buffer, obj)
        return buffer.getvalue()

    def _add_bytes_to_tar(bytes, name, tar):
        stream = io.BytesIO(bytes)
        tar_info = tarfile.TarInfo(name=name)
        tar_info.size = len(bytes)
        tar.addfile(tar_info, fileobj=stream)

    meta = {}
    meta['model_name'] = model_grid.model_name
    meta['fill_method'] = model_grid.fill_method
    meta['filters'] = model_grid.filters

    index_to_name = {index: name for name, index in model_grid.axes_map.items()}
    meta['axes'] = [index_to_name[i] for i in range(len(index_to_name))]

    arrays = {}
    arrays['phot'] = model_grid.phot
    arrays['spec'] = model_grid.spec
    arrays['wave'] = model_grid.wave
    for name, axis in zip(meta['axes'], model_grid.points):
        arrays[name] = axis
    with tarfile.open(path, mode='w|gz') as tar:
        _add_bytes_to_tar(_write_meta(meta), 'meta.json', tar)
        for name, array in arrays.items():
            _add_bytes_to_tar(_write_array(array), f'{name}.npy', tar)

def save_model_pickle(model_grid: ModelGrid, path):
    """
    Save a ModelGrid to a pickle file.

    Parameters
    ----------
    model_grid : ModelGrid
        The ModelGrid to save.
    path : str
        The path to save the ModelGrid.
    """
    axes = model_grid.axes_map
    phot_interp = RegularGridInterpolator(model_grid.points, model_grid.phot)
    spec_interp = RegularGridInterpolator(model_grid.points, model_grid.spec)
    out = {
        'T_eff':              model_grid.points[axes['teff']],
        'logg':               model_grid.points[axes['logg']],
        'kzz':                model_grid.points[axes['kzz']],
        'mh':                 model_grid.points[axes['mh']],
        'co':                 model_grid.points[axes['co']],
        'filters':            model_grid.filters,
        'wave':               model_grid.wave,
        'phot_interpolator':  phot_interp,
        'spec_interpolator':  spec_interp,
    }
    with open(path, 'wb') as f:
        pickle.dump(out, f)


# ============================================================
# Model: Sonora Elf Owl v2  (and PH3, which shares the format)
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       output_275.0_325.0/
#           spectra_logzz_2.0_teff_275.0_grav_17.0_mh_-0.5_co_0.5.nc
#           ...
#       output_350.0_400.0/
#           ...
#       ...
#
# File format: NetCDF (.nc), opened with xarray.
#   ds["wavelength"] : microns
#   ds["flux"]       : erg/s/cm^2/cm  (f_lambda per cm, not per Ang)
#
# Parameters encoded in filename (split on '_'):
#   index 2 -> log10(Kzz)
#   index 4 -> Teff (K)
#   index 6 -> surface gravity in cgs / 100  (so log10(grav)+2.0 = log10(g_cgs))
#   index 8 -> [M/H]
#   index 10 (strip '.nc') -> C/O
#
# Missing grid points: Sonora v2 has some holes at the lowest logg values.
# Strategy: build a sub-grid on the non-missing logg values (logg[1:]) and
# a restricted set of C/O values ([:3]) and use that as an extrapolator to
# fill in the holes in the full grid before building the final interpolator.
# ============================================================

def _parse_sonora_filename(path):
    """
    Parse Teff, kzz, logg, [M/H], C/O from a Sonora Elf Owl .nc filename.

    Returns log10(Teff), kzz, log10(g_cgs), mh, co
    """
    parts = os.path.basename(path).split('_')
    teff = float(parts[4])
    kzz  = float(parts[2])
    grav = float(parts[6])
    mh   = float(parts[8])
    co   = float(parts[10].split('.nc')[0])
    return np.log10(teff), kzz, np.log10(grav) + 2.0, mh, co


def build_sonora_elf_owl(model_path, filter_name, filter_sedpy, output_filename):
    """
    Build and save the Sonora Elf Owl v2 interpolator.

    Parameters
    ----------
    model_path : str
        Path to directory containing output_<Tmin>_<Tmax>/ subdirectories.
    filter_name : list of str
    filter_sedpy : list of sedpy Filter objects
    output_filename : str
        Path to write the output .pkl file.
    """
    try:
        import xarray
    except ImportError:
        sys.exit("Error: xarray is required for Sonora Elf Owl models. "
                 "Install with: pip install xarray")

    # Temperature-range subdirectory bounds
    temp_min = np.array([275.0, 350.0, 425.0, 500.0, 575.0,
                         700.0, 850.0, 1000.0, 1300.0, 1600.0, 1900.0, 2200.0])
    temp_max = np.array([325.0, 400.0, 475.0, 550.0, 650.0,
                         800.0, 950.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0])

    teff_list, logg_list, kzz_list, mh_list, co_list = [], [], [], [], []
    flux_list, spec_list = [], []

    for tmin, tmax in zip(temp_min, temp_max):
        directory = os.path.join(
            model_path,
            f"output_{round(tmin, 1)}_{round(tmax, 1)}"
        ) + "/"
        spectra_paths = glob.glob(directory + "spectra*")
        print(f"  Tmin={tmin} to Tmax={tmax}: {len(spectra_paths)} spectra")

        for path in tqdm(spectra_paths):
            lteff, kzz, llogg, mh, co = _parse_sonora_filename(path)

            with xarray.open_dataset(path) as ds:
                wave_um  = ds["wavelength"].values
                flux_cgs_per_cm = ds["flux"].values

            wave_ang = wave_um * 1e4
            # Convert erg/s/cm^2/cm -> erg/s/cm^2/Ang  (divide by 1e8)
            flux_ergscm2ang = flux_cgs_per_cm / 1e8

            sort_idx = np.argsort(wave_ang)
            wave_s = wave_ang[sort_idx]
            flux_s = flux_ergscm2ang[sort_idx]

            teff_list.append(lteff);  logg_list.append(llogg)
            kzz_list.append(kzz);    mh_list.append(mh)
            co_list.append(co)
            flux_list.append(compute_filter_fluxes(wave_s, flux_s, filter_sedpy))
            spec_list.append(resample_spectrum(wave_s, flux_s))

    output_teff = np.asarray(teff_list)
    output_logg = np.asarray(logg_list)
    output_kzz  = np.asarray(kzz_list)
    output_mh   = np.asarray(mh_list)
    output_co   = np.asarray(co_list)
    # Shape: (n_filters, n_spectra) and (n_wave, n_spectra)
    output_fluxes  = np.column_stack(flux_list)
    output_spectra = np.column_stack(spec_list)

    # Build the index map now that all spectra are loaded
    index_map = {}
    for q in range(len(output_teff)):
        index_map[keyify(output_teff[q], output_logg[q], output_kzz[q],
                         output_mh[q], output_co[q])] = q

    Teff_values = np.unique(output_teff)
    logg_values = np.unique(output_logg)
    kzz_values  = np.unique(output_kzz)
    mh_values   = np.unique(output_mh)
    co_values   = np.unique(output_co)

    # -----------------------------------------------------------
    # Pass 1: sub-grid extrapolator to fill holes in the full grid.
    # The v2 grid is missing some combinations at the lowest logg
    # and highest C/O, so we build a complete sub-grid on logg[1:]
    # and co[:3] first, then use it to fill in the missing cells.
    # -----------------------------------------------------------
    extrap_logg = logg_values[1:]
    extrap_co   = co_values[:3]

    n_extrap_shape = (len(Teff_values), len(extrap_logg), len(kzz_values),
                      len(mh_values), len(extrap_co))
    extrap_phot_grid = np.empty(n_extrap_shape + (output_fluxes.shape[0],))
    extrap_spec_grid = np.empty(n_extrap_shape + (output_spectra.shape[0],),
                                dtype=np.float32)

    ngood = nbad = 0
    for i0, tv in enumerate(Teff_values):
        for i1, lv in enumerate(extrap_logg):
            for i2, kv in enumerate(kzz_values):
                for i3, mv in enumerate(mh_values):
                    for i4, cv in enumerate(extrap_co):
                        q = index_map.get(keyify(tv, lv, kv, mv, cv))
                        if q is not None:
                            extrap_phot_grid[i0,i1,i2,i3,i4,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                            extrap_spec_grid[i0,i1,i2,i3,i4,:] = output_spectra[:,q]
                            ngood += 1
                        else:
                            print(f"    Missing in sub-grid: "
                                  f"Teff={10**tv:.1f} logg={lv} kzz={kv} "
                                  f"mh={mv} co={cv}")
                            nbad += 1

    print(f"  Sub-grid: {ngood} points found, {nbad} missing.")
    phot_extrapolator = RegularGridInterpolator(
        (Teff_values, extrap_logg, kzz_values, mh_values, extrap_co),
        extrap_phot_grid, method='linear', bounds_error=False, fill_value=None
    )
    spec_extrapolator = RegularGridInterpolator(
        (Teff_values, extrap_logg, kzz_values, mh_values, extrap_co),
        extrap_spec_grid, method='linear', bounds_error=False, fill_value=None
    )

    # -----------------------------------------------------------
    # Pass 2: fill the full grid, using the extrapolator for holes
    # -----------------------------------------------------------
    full_shape = (len(Teff_values), len(logg_values), len(kzz_values),
                  len(mh_values), len(co_values))
    phot_grid = np.empty(full_shape + (output_fluxes.shape[0],))
    spec_grid = np.empty(full_shape + (output_spectra.shape[0],), dtype=np.float32)

    n_total = n_extrapolations = 0
    for i0, tv in enumerate(Teff_values):
        for i1, lv in enumerate(logg_values):
            for i2, kv in enumerate(kzz_values):
                for i3, mv in enumerate(mh_values):
                    for i4, cv in enumerate(co_values):
                        n_total += 1
                        q = index_map.get(keyify(tv, lv, kv, mv, cv))
                        if q is not None:
                            phot_grid[i0,i1,i2,i3,i4,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                            spec_grid[i0,i1,i2,i3,i4,:] = output_spectra[:,q]
                        else:
                            n_extrapolations += 1
                            phot_grid[i0,i1,i2,i3,i4,:] = phot_extrapolator(
                                [tv, lv, kv, mv, cv])[0]
                            spec_grid[i0,i1,i2,i3,i4,:] = spec_extrapolator(
                                [tv, lv, kv, mv, cv])[0]

    pct = round(100 * n_extrapolations / n_total, 2)
    print(f"Extrapolated {n_extrapolations} / {n_total} grid points ({pct}%)")

    phot_interp = RegularGridInterpolator(
        (Teff_values, logg_values, kzz_values, mh_values, co_values),
        phot_grid, method='linear', fill_value=None
    )
    spec_interp = RegularGridInterpolator(
        (Teff_values, logg_values, kzz_values, mh_values, co_values),
        spec_grid, method='linear', fill_value=None
    )

    out = {
        'T_eff':              Teff_values,
        'logg':               logg_values,
        'kzz':                kzz_values,
        'mh':                 mh_values,
        'co':                 co_values,
        'filters':            filter_name,
        'wave':               LOWER_RES_WAVE,
        'phot_interpolator':  phot_interp,
        'spec_interpolator':  spec_interp,
    }
    save_pickle(out, output_filename)


# ============================================================
# Model: Sonora Elf Owl + PH3
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       elf_owl_disequilibrium_PH3.npz
#
# File format: NumPy .npz archive.
#   grid_function['Teff']    : 1-D array of Teff values (K)
#   grid_function['logg']    : 1-D array of surface gravity (cgs / 100,
#                              same convention as Sonora Elf Owl)
#   grid_function['logkzz']  : 1-D array of log10(Kzz)
#   grid_function['logmh']   : 1-D array of [M/H]
#   grid_function['cto']     : 1-D array of C/O
#   grid_function['wvno']    : 1-D array of wavenumber (cm^-1)
#                              -> wavelength in microns = 1e4 / wvno
#   grid_function['spectra'] : 5-D array (nTeff, nlogg, nkzz, nmh, nco, nwave)
#                              in erg/s/cm^2/cm
#
# The PH3 grid is complete (no missing combinations), so no extrapolation pass
# is needed. Parameters are stored in the array directly — no filename parsing.
# ============================================================

def build_sonora_ph3(model_path, filter_name, filter_sedpy, output_filename):
    """
    Build and save the Sonora Elf Owl + PH3 interpolator.

    Parameters
    ----------
    model_path : str
        Directory containing elf_owl_disequilibrium_PH3.npz.
    filter_name : list of str
    filter_sedpy : list of sedpy Filter objects
    output_filename : str
    """
    npz_path = os.path.join(model_path, "elf_owl_disequilibrium_PH3.npz")
    if not os.path.exists(npz_path):
        sys.exit(f"Error: expected file not found: {npz_path}")

    print("Loading PH3 grid from: " + npz_path)
    grid_function = np.load(npz_path)

    teff_pt   = grid_function['Teff']     # K
    grav_pt   = grid_function['logg']     # cgs / 100
    logkz_pt  = grid_function['logkzz']
    logmh_pt  = grid_function['logmh']
    cto_pt    = grid_function['cto']
    wvno_pt   = grid_function['wvno']     # cm^-1
    spec_pt   = grid_function['spectra']  # erg/s/cm^2/cm, shape (T,g,kzz,mh,co,wave)

    # Wavelength grid from wavenumber: 1e4/wvno gives microns
    wave_um  = 1e4 / wvno_pt
    wave_ang = wave_um * 1e4
    sort_idx = np.argsort(wave_ang)
    wave_ang_sorted = wave_ang[sort_idx]

    # Grid axes in the same convention used by Sonora Elf Owl and NIFTY.py
    Teff_values = np.log10(teff_pt)
    logg_values = np.log10(grav_pt) + 2.0
    kzz_values  = logkz_pt
    mh_values   = logmh_pt
    co_values   = cto_pt

    number_spectra = (len(teff_pt) * len(grav_pt) * len(logkz_pt)
                      * len(logmh_pt) * len(cto_pt))
    print(f"  Grid size: {number_spectra} spectra")

    output_fluxes  = np.zeros([len(filter_sedpy), number_spectra])
    output_spectra = np.zeros([len(LOWER_RES_WAVE), number_spectra])

    # Build the index map as we fill in the flat arrays, then use it to
    # populate the 5-D grids without float-equality np.where lookups.
    index_map = {}
    q = 0
    for i0 in tqdm(range(len(teff_pt))):
        for i1 in range(len(grav_pt)):
            for i2 in range(len(logkz_pt)):
                for i3 in range(len(logmh_pt)):
                    for i4 in range(len(cto_pt)):

                        flux_cgs_per_cm = spec_pt[i0, i1, i2, i3, i4]
                        flux_ergscm2ang = flux_cgs_per_cm[sort_idx] / 1e8

                        output_fluxes[:, q] = compute_filter_fluxes(
                            wave_ang_sorted, flux_ergscm2ang, filter_sedpy)
                        output_spectra[:, q] = resample_spectrum(
                            wave_ang_sorted, flux_ergscm2ang)

                        index_map[keyify(
                            Teff_values[i0], logg_values[i1],
                            kzz_values[i2],  mh_values[i3],
                            co_values[i4]
                        )] = q
                        q += 1

    # The PH3 grid is complete — populate directly from the index map
    phot_grid = np.empty((len(Teff_values), len(logg_values), len(kzz_values),
                          len(mh_values), len(co_values), output_fluxes.shape[0]))
    spec_grid = np.empty((len(Teff_values), len(logg_values), len(kzz_values),
                          len(mh_values), len(co_values), output_spectra.shape[0]))

    for i0, tv in enumerate(Teff_values):
        for i1, lv in enumerate(logg_values):
            for i2, kv in enumerate(kzz_values):
                for i3, mv in enumerate(mh_values):
                    for i4, cv in enumerate(co_values):
                        q = index_map[keyify(tv, lv, kv, mv, cv)]
                        phot_grid[i0,i1,i2,i3,i4,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                        spec_grid[i0,i1,i2,i3,i4,:] = output_spectra[:,q]

    phot_interp = RegularGridInterpolator(
        (Teff_values, logg_values, kzz_values, mh_values, co_values),
        phot_grid, method='linear', fill_value=None
    )
    spec_interp = RegularGridInterpolator(
        (Teff_values, logg_values, kzz_values, mh_values, co_values),
        spec_grid, method='linear', fill_value=None
    )

    out = {
        'T_eff':              Teff_values,
        'logg':               logg_values,
        'kzz':                kzz_values,
        'mh':                 mh_values,
        'co':                 co_values,
        'filters':            filter_name,
        'wave':               LOWER_RES_WAVE,
        'phot_interpolator':  phot_interp,
        'spec_interpolator':  spec_interp,
    }
    save_pickle(out, output_filename)


# ============================================================
# Model: ATMO2020++
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       grid_m1.0/
#           spec_jwst_t<Teff>_g<logg>_<mh>_kg_g<logg2>.dat
#           ...
#       grid_m0.5/
#       grid_p0/
#       grid_p0.3/
#
# File format: whitespace-delimited ASCII, two columns:
#   col 0: wavelength in microns
#   col 1: flux in W/m^2/um  ->  multiply by 0.1 to get erg/s/cm^2/Ang
#
# Filename format: spec_jwst_t<Teff>_g<logg>_<mh>_kg_g<logg2>.dat
#   split on '_': [0]=spec [1]=jwst [2]=t<Teff> [3]=g<logg> [4]=<mh> ...
#   teff from [2][1:]
#   logg from [3][1:]
#   mh   from [4], prefix 'm' -> negative, 'p' -> positive
#
# Parameters: Teff, logg, [M/H]  (no Kzz or C/O in this grid)
#
# The ATMO2020 grid is complete — no extrapolation pass needed.
# ============================================================

def _parse_atmo_filename(filename):
    """
    Parse Teff, logg, [M/H] from an ATMO2020 .dat filename.
    Uses the filename argument (not any outer-scope variable).

    Returns log10(Teff), logg, mh
    """
    parts = os.path.basename(filename).split('_')
    teff = float(parts[2][1:])
    logg = float(parts[3][1:])
    mh_raw = parts[4]
    if mh_raw.startswith('m'):
        mh = -1.0 * float(mh_raw[1:])
    elif mh_raw.startswith('p'):
        mh = float(mh_raw[1:])
    else:
        mh = float(mh_raw)
    return np.log10(teff), logg, mh


def build_atmo2020(model_path, filter_name, filter_sedpy, output_filename):
    """
    Build and save the ATMO2020++ interpolator.

    Parameters
    ----------
    model_path : str
        Directory containing grid_m1.0/, grid_m0.5/, grid_p0/, grid_p0.3/ etc.
    filter_name : list of str
    filter_sedpy : list of sedpy Filter objects
    output_filename : str
    """
    # Subdirectory names mirror the [M/H] values in the ATMO2020 release
    mh_subdirs = ['m1.0', 'm0.5', 'p0', 'p0.3']

    teff_list, logg_list, mh_list = [], [], []
    flux_list, spec_list = [], []

    for mh_stub in mh_subdirs:
        directory = os.path.join(model_path, 'grid_' + mh_stub) + '/'
        spectra_paths = glob.glob(directory + 'spec*')
        print(f"  [M/H]={mh_stub}: {len(spectra_paths)} spectra")

        for path in tqdm(spectra_paths):
            lteff, logg, mh = _parse_atmo_filename(path)

            data = np.loadtxt(path)
            wave_um   = data[:, 0]           # microns
            flux_wm2m = data[:, 1]           # W/m^2/um

            wave_ang = wave_um * 1e4
            # Convert W/m^2/um -> erg/s/cm^2/Ang  (multiply by 0.1)
            flux_ergscm2ang = flux_wm2m * 0.1

            sort_idx = np.argsort(wave_ang)
            wave_s = wave_ang[sort_idx]
            flux_s = flux_ergscm2ang[sort_idx]

            teff_list.append(lteff)
            logg_list.append(logg)
            mh_list.append(mh)
            flux_list.append(compute_filter_fluxes(wave_s, flux_s, filter_sedpy))
            spec_list.append(resample_spectrum(wave_s, flux_s))

    output_teff = np.asarray(teff_list)
    output_logg = np.asarray(logg_list)
    output_mh   = np.asarray(mh_list)
    output_fluxes  = np.column_stack(flux_list)
    output_spectra = np.column_stack(spec_list)

    # Build index map
    index_map = {}
    for q in range(len(output_teff)):
        index_map[keyify(output_teff[q], output_logg[q], output_mh[q])] = q

    Teff_values = np.unique(output_teff)
    logg_values = np.unique(output_logg)
    mh_values   = np.unique(output_mh)

    phot_grid = np.empty((len(Teff_values), len(logg_values), len(mh_values),
                          output_fluxes.shape[0]))
    spec_grid = np.empty((len(Teff_values), len(logg_values), len(mh_values),
                          output_spectra.shape[0]))

    n_missing = 0
    for i0, tv in enumerate(Teff_values):
        for i1, lv in enumerate(logg_values):
            for i2, mv in enumerate(mh_values):
                q = index_map.get(keyify(tv, lv, mv))
                if q is not None:
                    phot_grid[i0,i1,i2,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                    spec_grid[i0,i1,i2,:] = output_spectra[:,q]
                else:
                    # ATMO2020 grid should be complete; flag any surprises
                    print(f"  Warning: missing grid point "
                          f"Teff={10**tv:.1f} logg={lv} mh={mv}")
                    n_missing += 1

    if n_missing:
        print(f"  {n_missing} missing grid points — "
              f"RegularGridInterpolator will extrapolate at boundaries.")

    phot_interp = RegularGridInterpolator(
        (Teff_values, logg_values, mh_values),
        phot_grid, method='linear', fill_value=None
    )
    spec_interp = RegularGridInterpolator(
        (Teff_values, logg_values, mh_values),
        spec_grid, method='linear', fill_value=None
    )

    out = {
        'T_eff':              Teff_values,
        'logg':               logg_values,
        'mh':                 mh_values,
        'filters':            filter_name,
        'wave':               LOWER_RES_WAVE,
        'phot_interpolator':  phot_interp,
        'spec_interpolator':  spec_interp,
    }
    save_pickle(out, output_filename)


def load_lowz_file(path):
    data = pd.read_csv(path, sep=' ', comment='#', header=None, engine='c').values
    wave_um   = data[:, 0]    # microns
    flux_wm2m = data[:, 1]    # W/m^2/m

    wave_ang = wave_um * 1e4
    # Convert W/m^2/m -> erg/s/cm^2/Ang  (multiply by 1e-7)
    flux_ergscm2ang = flux_wm2m * 1e-7

    wave_s = wave_ang
    flux_s = flux_ergscm2ang
    flux = compute_filter_fluxes(wave_s, flux_s, filter_sedpy)
    spec = resample_spectrum(wave_s, flux_s)
    return flux, spec

# ============================================================
# Model: LOWZ
#
# Expected directory layout
# -------------------------
#   <model_path>/
#       LOWZ_models_index.csv    (columns: TEFF LOGG METALLICITY CTOO LOGKZZ FILENAME)
#       models/
#           LOW_Z_<...>.txt
#           ...
#
# File format: whitespace-delimited ASCII, two columns:
#   col 0: wavelength in microns
#   col 1: flux in W/m^2/m  ->  multiply by 1e-7 to get erg/s/cm^2/Ang
#
# The LOWZ grid has 13 missing parameter combinations (documented in comments
# below). Strategy is identical to Sonora Elf Owl: build a sub-grid on the
# well-sampled region (Teff > 600 K, C/O in [0.1, 0.85]) to use as an
# extrapolator, then fill the full grid.
# ============================================================

def build_lowz(model_path, filter_name, output_filename):
    """
    Build and save the LOWZ interpolator.

    Parameters
    ----------
    model_path : str
        Directory containing LOWZ_models_index.csv and models/ subdirectory.
    filter_name : list of str
    output_filename : str
    """

    path = Path(model_path)
    index_path = path / 'LOWZ_models_index.csv'
    models_path = path / 'models'

    if not index_path.exists():
        raise ValueError(f"Error: index file does not exist: {index_path}")

    if not models_path.exists():
        raise ValueError(f"Error: models directory not found: {models_path}")

    index = pd.read_csv(index_path).set_index("FILENAME")

    spectra_paths = list(models_path.iterdir())
    print(f"  LOWZ: {len(spectra_paths)} spectra")

    teff_list, logg_list, kzz_list, mh_list, co_list = [], [], [], [], []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(load_lowz_file, spectra_paths),
            total=len(spectra_paths)
        ))

    flux_list, spec_list = tuple(map(list, zip(*results)))

    for path in tqdm(spectra_paths):
        filename = path.name
        try:
            row = index.loc[filename]
        except KeyError:
            print(f"  Warning: {filename} not found in index CSV, skipping.")
            continue

        teff = row['TEFF']
        logg = row['LOGG']
        kzz  = row['LOGKZZ']
        mh   = row['METALLICITY']
        co   = row['CTOO']

        teff_list.append(np.log10(teff))
        logg_list.append(logg)
        kzz_list.append(kzz)
        mh_list.append(mh)
        co_list.append(co)

    output_teff = np.asarray(teff_list)
    output_logg = np.asarray(logg_list)
    output_kzz  = np.asarray(kzz_list)
    output_mh   = np.asarray(mh_list)
    output_co   = np.asarray(co_list)
    output_fluxes  = np.column_stack(flux_list)
    output_spectra = np.column_stack(spec_list)

    # Build index map
    index_map = {}
    for q in range(len(output_teff)):
        index_map[keyify(output_teff[q], output_logg[q], output_kzz[q],
                         output_mh[q], output_co[q])] = q

    Teff_values = np.unique(output_teff)
    logg_values = np.unique(output_logg)
    kzz_values  = np.unique(output_kzz)
    mh_values   = np.unique(output_mh)
    co_values   = np.unique(output_co)

    # -----------------------------------------------------------
    # Known missing grid points in LOWZ (13 total):
    #   Teff  log(g)  kzz  [M/H]  C/O
    #   500   5.0    -1.0  -0.5   0.55
    #   550   3.5    -1.0  -1.0   0.55
    #   550   3.5     2.0  -1.5   0.1
    #   550   5.25   -1.0  -1.5   0.1
    #   600   4.5    10.0  -1.5   0.55
    #   650   5.25    2.0  -0.5   0.55
    #   800   5.25   10.0   0.0   0.55
    #   850   3.5    10.0   0.0   0.55
    #   850   5.0    10.0  -2.0   0.55
    #   900   4.0     2.0  -2.0   0.55
    #   950   4.0    -1.0   0.0   0.55
    #   950   4.5    -1.0   0.0   0.55
    #   950   5.0    10.0  -2.0   0.55
    #
    # Strategy: build a sub-grid on Teff > 600 K and C/O in [0.1, 0.85]
    # (the two best-sampled C/O values) to use as an extrapolator.
    # -----------------------------------------------------------
    extrap_teff = Teff_values[2:]         # Teff > 600 K
    extrap_co   = co_values[[0, 2]]       # C/O = 0.1 and 0.85

    n_extrap_shape = (len(extrap_teff), len(logg_values), len(kzz_values),
                      len(mh_values), len(extrap_co))
    extrap_phot_grid = np.empty(n_extrap_shape + (output_fluxes.shape[0],))
    extrap_spec_grid = np.empty(n_extrap_shape + (output_spectra.shape[0],))

    ngood = nbad = 0
    for i0, tv in enumerate(extrap_teff):
        for i1, lv in enumerate(logg_values):
            for i2, kv in enumerate(kzz_values):
                for i3, mv in enumerate(mh_values):
                    for i4, cv in enumerate(extrap_co):
                        q = index_map.get(keyify(tv, lv, kv, mv, cv))
                        if q is not None:
                            extrap_phot_grid[i0,i1,i2,i3,i4,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                            extrap_spec_grid[i0,i1,i2,i3,i4,:] = output_spectra[:,q]
                            ngood += 1
                        else:
                            print(f"    Missing in sub-grid: "
                                  f"Teff={10**tv:.1f} logg={lv} kzz={kv} "
                                  f"mh={mv} co={cv}")
                            nbad += 1

    phot_extrapolator = RegularGridInterpolator(
        (extrap_teff, logg_values, kzz_values, mh_values, extrap_co),
        extrap_phot_grid, method='linear', bounds_error=False, fill_value=None
    )
    spec_extrapolator = RegularGridInterpolator(
        (extrap_teff, logg_values, kzz_values, mh_values, extrap_co),
        extrap_spec_grid, method='linear', bounds_error=False, fill_value=None
    )

    # Full grid pass
    full_shape = (len(Teff_values), len(logg_values), len(kzz_values),
                  len(mh_values), len(co_values))
    phot_grid = np.empty(full_shape + (output_fluxes.shape[0],))
    spec_grid = np.empty(full_shape + (output_spectra.shape[0],))

    raw_phot_grid = np.full(
        full_shape + (output_fluxes.shape[0],), np.nan, dtype=np.float32
    )
    raw_spec_grid = np.full(
        full_shape + (output_spectra.shape[0],), np.nan, dtype=np.float32
    )
    n_total = n_extrapolations = 0
    for i0, tv in enumerate(Teff_values):
        for i1, lv in enumerate(logg_values):
            for i2, kv in enumerate(kzz_values):
                for i3, mv in enumerate(mh_values):
                    for i4, cv in enumerate(co_values):
                        n_total += 1
                        q = index_map.get(keyify(tv, lv, kv, mv, cv))
                        if q is not None:
                            phot_grid[i0,i1,i2,i3,i4,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                            spec_grid[i0,i1,i2,i3,i4,:] = output_spectra[:,q]
                            raw_phot_grid[i0,i1,i2,i3,i4,:] = output_fluxes[:,q] / 1e-23 / 1e-9
                            raw_spec_grid[i0,i1,i2,i3,i4,:] = output_spectra[:,q]
                        else:
                            n_extrapolations += 1
                            phot_grid[i0,i1,i2,i3,i4,:] = phot_extrapolator(
                                [tv, lv, kv, mv, cv])[0]
                            spec_grid[i0,i1,i2,i3,i4,:] = spec_extrapolator(
                                [tv, lv, kv, mv, cv])[0]

    pct = round(100 * n_extrapolations / n_total, 2)
    print(f"Extrapolated {n_extrapolations} / {n_total} grid points ({pct}%)")

    phot_interp = RegularGridInterpolator(
        (Teff_values, logg_values, kzz_values, mh_values, co_values),
        phot_grid, method='linear', fill_value=None
    )
    spec_interp = RegularGridInterpolator(
        (Teff_values, logg_values, kzz_values, mh_values, co_values),
        spec_grid, method='linear', fill_value=None
    )

    out = {
        'T_eff':              Teff_values,
        'logg':               logg_values,
        'kzz':                kzz_values,
        'mh':                 mh_values,
        'co':                 co_values,
        'filters':            filter_name,
        'wave':               LOWER_RES_WAVE,
        'phot_interpolator':  phot_interp,
        'spec_interpolator':  spec_interp,
    }
    save_pickle(out, output_filename)

    # Return the raw model grid (no points filled in)
    model_grid = ModelGrid(
        points=(Teff_values, logg_values, kzz_values, mh_values, co_values),
        phot=raw_phot_grid,
        spec=raw_spec_grid,
        axes_map={n: i for i, n in enumerate(['teff', 'logg', 'kzz', 'mh', 'co'])},
        model_name='lowz',
        fill_method=None,
        filters=filter_name,
        wave=LOWER_RES_WAVE
    )
    return model_grid


def fill_point(points, values, target):
    """
    Estimate the value of a missing point in a grid by fitting a polynomial.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (d1, ), ..., (dN, )
        The points defining the regular grid in N dimensions.
    values : array_like, shape (d1, ..., dN, M)
        The data on the regular grid in N dimensions. Each value on the grid
        is a vector of length M.
    target: tuple of N ints
        The index of the target point in N dimensions.

    Returns
    --------
    value : ndarray of shape (M,)
        The estimated data vector at the target.
    """
    # The number of parameters.
    N = len(points)

    # The length of the output vector.
    M = values.shape[-1]

    # Slice the hypercube.
    cube_starts = []
    cube_points = []
    for coord, axis_points, axis_len in zip(target, points, values.shape):
        if axis_len < 3:
            raise ValueError(
                "Unable to construct hypercube: "
                "all axes must have a length of at least 3. "
                f"Recieved axis lengths {values.shape[:-1]}."
            )
        start = coord - 1
        # Shift the hypercube if it goes out of bounds.
        start = max(start, 0)
        start = min(start, axis_len - 3)
        cube_starts.append(start)

        # Extract the physical values for this axis window.
        cube_points.append(axis_points[start:start + 3])

    # Get the slices for extracting the hypercube.
    slices = tuple(slice(start, start + 3) for start in cube_starts)
    # Extract the entire output vector.
    slices = slices + (slice(None),)

    # Y is an array with shape (n_points, M) where
    # Y[i, :] represents the output vector at point i.
    Y = values[slices].reshape(-1, M)

    # axis_points is an array with shape (N, 3) where
    # axis_points[i, j] is coordinate j on axis i.
    # The cube edge has three points, so 0 <= j < 3.
    axis_points = np.stack(cube_points) # shape (N, 3)

    target_rel_cube = tuple(t - s for t, s in zip(target, cube_starts))
    target_point = axis_points[np.arange(N), target_rel_cube]

    # Get coordinates that are not the same as the target coordinate.
    rem_mask = np.asarray(target_rel_cube)[:, np.newaxis] != np.arange(3)
    remaining = axis_points[rem_mask].reshape(-1, 2)
    total_dist = (
        np.abs(target_point - remaining[:, 1]) +
        np.abs(target_point - remaining[:, 0])
    )

    # Arrays of shape (N,) denoting properties for each axis.
    axis_space = axis_points[:, 2] - axis_points[:, 0]
    axis_center = axis_points[:, 1]

    # X is an array with shape (n_points, N) where
    # X[i, :] represents the physical coordinates at point i.
    mesh_grids = np.meshgrid(*cube_points, indexing='ij')
    X = np.stack(mesh_grids, axis=-1).reshape(-1, N)

    # Identify points containing any NaN element in their output vector.
    mask = np.isnan(Y).any(axis=-1)

    # Mask the target point.
    # This is not necessary if the target point's value is known to be NaN.
    # We only set this for testing the fit at existing grid points.
    mask[np.ravel_multi_index(target_rel_cube, (3,) * N)] = True

    X = X[~mask]
    Y = Y[~mask]

    # Define the weight vector with shape (n_points,).
    # The weight along one axis is 1 - |x - t| / d, where x is the point
    # being weighted, t is the target point, and d is the total distance
    # between the target point and each of the other two points. The weight
    # of a point is the product of its weight along all axes.

    w = np.prod(1 - np.abs((X - target_point) / total_dist), axis=1)
    # w = np.prod(1 - np.square((X - target_point) / total_dist), axis=1)
    # w = np.full((X.shape[0],), 1.0, dtype=values.dtype)

    # Normalize the input vectors for numerical stability.
    X = (X - axis_center) / axis_space

    # We need at least 2^N points to solve a multilinear polynomial.
    num_terms = 2**N
    if len(X) < num_terms:
        raise ValueError(
            f"Insufficient valid points ({len(X)}) to fit a "
            f"multilinear polynomial requiring {num_terms} coefficients."
        )

    # Construct the design matrix for a multilinear polynomial.
    # For N variables, we require 2^N terms (combinations of subsets).

    # Assign every N-bit integer to a term.
    # term_i has shape (2^N,).
    term_i = np.arange(num_terms, dtype=np.int32)

    # Determine which variables should be included in each term.
    # include_term[i, j] is True when term i contains variable j.
    # include_term has shape (2^N, N).
    include_term = (term_i[:, np.newaxis] >> np.arange(N)) & 1

    # If the bit is 0, we want 1 (so it doesn't affect the product).
    # If the bit is 1, we want the variable value from X.
    # X[:, np.newaxis, :] has shape (n_points, 1, N).
    # include_term[np.newaxis, ...] has shape (1, 2^N, N).
    # factors has shape (n_points, 2^N, N).
    factors = np.where(include_term[np.newaxis, ...], X[:, np.newaxis, :], 1.0)

    # Multiply along the variable axis (axis=2) to get shape (n_points, 2^N).
    A = np.prod(factors, axis=2)

    # To perform, weighted least squares, we modify A and Y.
    # If W is the matrix with w as its diagonal, this is the equivalent of
    # A' = W^(1/2)A and Y' = W^(1/2)Y.
    # Solving with these new values using OLS is equivalent to WLS.
    w_sqrt = np.sqrt(w)[:, np.newaxis]
    A *= w_sqrt
    Y *= w_sqrt

    C, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    # Solve the polynomial at the target point.
    # See above for how the design matrix is created.
    # This is similar, except that only one row is needed for the one point.
    target_norm = (target_point - axis_center) / axis_space
    factors = np.where(include_term, target_norm[np.newaxis, :], 1.0)
    a_target = np.prod(factors, axis=1)

    # Compute the normalized predictions: shape (1, M).
    y_target = a_target @ C

    return y_target

def fill_model_fit(model: ModelGrid):
    """
    Fill the missing points in a model by fitting a polynomial around each one.

    Parameters
    ----------
    model : ModelGrid
        The model to fill in place.
    """
    for grid in [model.phot, model.spec]:
        # Get all indices that are missing points (NaN).
        # Grid has shape (d1, ..., dN, M)
        # nan_mask has shape (d1, ..., dN)
        nan_mask = np.isnan(grid).any(axis=-1)
        # missing has shape (number_missing, N)
        missing = np.argwhere(nan_mask)
        missing_list = [tuple(index) for index in missing]
        for target in tqdm(missing_list):
            grid[target] = fill_point(model.points, grid, target)
    model.fill_method = 'fit'

# ============================================================
# Argument parsing and entry point
# ============================================================

# Default output filenames, matching what NIFTY.py expects
DEFAULT_OUTPUT = {
    'SonoraElfOwl':    'Sonora_v2_interp.pkl',
    'SonoraElfOwlPH3': 'Sonora_PH3_interp.pkl',
    'ATMO2020':        'ATMO2020_interp.pkl',
    'LOWZ':            'LOWZ_interp.pkl',
}

parser = argparse.ArgumentParser(
    description="NIFTY interpolator builder — generates model grid .pkl files"
)
parser.add_argument(
    '-model', '--model',
    help="Model to build (SonoraElfOwl, SonoraElfOwlPH3, ATMO2020, LOWZ)",
    type=str, required=True, dest='model'
)
parser.add_argument(
    '-path', '--path',
    help="Path to the model files directory (see docstring for layout details)",
    type=str, required=True, dest='model_path'
)
parser.add_argument(
    '-config', '--config',
    help="Path to the NIFTY filter config JSON file",
    type=str, required=True, dest='config_file'
)
parser.add_argument(
    '-output', '--output',
    help=("Output .pkl filename. Defaults: " +
          ", ".join(f"{k} -> {v}" for k, v in DEFAULT_OUTPUT.items())),
    type=str, required=False, dest='output_file', default=None
)
args = parser.parse_args()


if __name__ == '__main__':

    model = args.model
    if model not in DEFAULT_OUTPUT:
        sys.exit(
            f"Error: '{model}' is not a recognised model.\n"
            f"Choose from: {', '.join(DEFAULT_OUTPUT.keys())}"
        )

    model_path = args.model_path
    if not os.path.isdir(model_path):
        sys.exit(f"Error: model path does not exist: {model_path}")

    config_file = args.config_file
    if not os.path.exists(config_file):
        sys.exit(f"Error: config file not found: {config_file}")

    output_file = args.output_file or DEFAULT_OUTPUT[model]

    print(" ")
    print("▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖   Near-Infrared")
    print("▐▛▚▖▐▌  █  ▐▌     █   ▝▚▞▘    Fitting for")
    print("▐▌ ▝▜▌  █  ▐▛▀▀▘  █    ▐▌     T and Y Dwarfs")
    print("▐▌  ▐▌▗▄█▄▖▐▌     █    ▐▌     Interpolator Builder")
    print(" ")
    print(f"Model    : {model}")
    print(f"Path     : {model_path}")
    print(f"Config   : {config_file}")
    print(f"Output   : {output_file}")
    print(" - - - - - - - - ")

    filter_name, filter_central_wavelength, filter_sedpy = load_filters(config_file)

    if model == 'SonoraElfOwl':
        build_sonora_elf_owl(model_path, filter_name, filter_sedpy, output_file)
    elif model == 'SonoraElfOwlPH3':
        build_sonora_ph3(model_path, filter_name, filter_sedpy, output_file)
    elif model == 'ATMO2020':
        build_atmo2020(model_path, filter_name, filter_sedpy, output_file)
    elif model == 'LOWZ':
        model_grid = build_lowz(model_path, filter_name, output_file)
        raw_model_path = 'lowz_model_raw.tar.gz'
        filled_model_path = 'lowz_model_filled.tar.gz'
        filled_model_pickle_path = 'lowz_model_filled_pickle.pkl'
        save_model(model_grid, raw_model_path)
        print(f"Saved raw model to: {raw_model_path}")
        print("Filling model...")
        fill_model_fit(model_grid)
        print("Model filled.")
        save_model(model_grid, filled_model_path)
        print(f"Saved filled model to: {filled_model_path}")
        save_model_pickle(model_grid, filled_model_pickle_path)
        print(f"Saved filled pickle model to: {filled_model_pickle_path}")
        print(f"Saved raw model to: {raw_model_path}")

    print(" - - - - - - - - ")
    print("Done!")
