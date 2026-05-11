#!/usr/bin/env python
"""\
NIFTY: Near-Infrared Fitting for T and Y Dwarfs
Kevin Hainline, Jake Helton

Usage: Run an MCMC fit using either the Sonora Elf Owl, ATMO2020, or LOWZ Models
       to NIRCam/MIRI photometry, or to NIRSpec prism spectroscopy.
       Produces an h5 file, as well as a corner plot and an SED showing the
       fit compared to the data.

Modes:
    phot  -- fit broadband photometry from a catalog (fits or whitespace-delimited txt)
    spec  -- fit a single NIRSpec prism spectrum (3-column whitespace-delimited txt)

Examples:
    python NIFTY.py -mode phot -model SonoraElfOwl -config_file filters.json \\
        -photometry catalog.fits -survey_stub JADES-GS -id 12345

    python NIFTY.py -mode spec -model SonoraElfOwl \\
        -spectroscopy spectrum.txt -survey_stub JADES-GS -id 12345
"""
# Imports necessary miscellaneous modules
import os
import ast
import sys
import time
import json
import warnings
import pickle
warnings.filterwarnings('ignore')
import argparse
import gc

# Imports necessary science modules
import emcee
import corner
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


# ============================================================
# Photometry loading (phot mode only)
# ============================================================

def load_flux_file(
	object_id,                  # int: The ID of the object to find
	photometry_path,            # str: Path to the file
	config,                     # config filter from the json file
	model_interp_object,        # Model interpolator
	filetype="fits",            # str: One of ["fits", "txt"]
	min_rel_err=0.05            # float: Minimum relative error floor
):
	"""
	Returns:
		fluxes: numpy array of length len(filters)
		flux_errors: numpy array of same shape
		used_indices: list of indices in `filters` that were successfully found
	"""
	number_interp_filters = len(model_interp_object['filters'])

	# For fits file input, with hdu extensions and filter names/errors defined
	# in the config json filters file.
	if (filetype == 'fits'):
		photometry_fits = fits.open(photometry_path, memmap=True)
		all_ID = photometry_fits['CIRC'].data['ID']

		object_index = np.where(all_ID == object_id)[0]
		if (len(object_index) == 0):
			print("Object ID "+str(object_id)+" not found! Skipping this object")
			return None, None, np.array([])

		object_index = object_index[0]

		object_flux = np.zeros(number_interp_filters)-9999
		object_flux_errors = np.zeros(number_interp_filters)-9999
		used_indices = []
		print("    Getting object fluxes for ID "+str(object_id))
		for q, filt in enumerate(model_interp_object["filters"]):

			if filt not in config["filter_columns"]:
				continue  # User didn't provide this filter, so we can skip things.

			col_info = config["filter_columns"][filt]
			ext = col_info["extension"]
			flux_col = col_info["flux"]
			err_col = col_info["error"]
			used_indices.append(q)

			try:
				flux_val = photometry_fits[ext].data[flux_col][object_index]
				if not np.isfinite(flux_val):
					continue

				err_val = photometry_fits[ext].data[err_col][object_index]
				if not np.isfinite(err_val):
					continue

				if (err_val / flux_val) < min_rel_err and (err_val / flux_val) > 0:
					err_val = flux_val * min_rel_err
					print(f"     Updating flux error in {filt} to minimum relative error.")

				object_flux[q] = flux_val
				object_flux_errors[q] = err_val

			except KeyError as e:
				print(f"Missing column for filter {filt}: {e}")
				continue

	# For text file input, with filter column names/errors defined
	# in the config json filters file.
	elif (filetype == 'txt'):

		try:
			photometry_data = pd.read_csv(photometry_path, sep=r'\s+', comment="#")
		except Exception as e:
			raise ValueError(f"Could not parse text file '{photometry_path}'. Make sure it's whitespace-delimited.\nSee README for expected format.\nOriginal error: {e}")

		all_ID = photometry_data["ID"].values

		object_index = np.where(all_ID == object_id)[0]
		if len(object_index) == 0:
			print(f"Object ID {object_id} not found! Skipping this object")
			return None, None, np.array([])

		object_index = object_index[0]

		object_flux = np.zeros(number_interp_filters)-9999
		object_flux_errors = np.zeros(number_interp_filters)-9999
		used_indices = []
		print("    Getting object fluxes for ID "+str(object_id))
		for q, filt in enumerate(model_interp_object["filters"]):

			if filt not in config["filter_columns"]:
				continue  # User didn't provide this filter, so we can skip things.

			col_info = config["filter_columns"][filt]
			flux_col = col_info["flux"]
			err_col = col_info["error"]
			used_indices.append(q)

			try:
				flux_val = photometry_data[flux_col][object_index]
				if not np.isfinite(flux_val):
					continue

				err_val = photometry_data[err_col][object_index]
				if not np.isfinite(err_val):
					continue

				if (err_val / flux_val) < min_rel_err and (err_val / flux_val) > 0:
					err_val = flux_val * min_rel_err
					print(f"     Updating flux error in {filt} to minimum relative error.")

				object_flux[q] = flux_val
				object_flux_errors[q] = err_val

			except KeyError as e:
				print(f"Missing column for filter {filt}: {e}")
				continue

	else:
		raise ValueError(f"Unsupported filetype: {filetype}")

	if len(used_indices) == 0:
		print(f"No matching filters found in config for object {object_id}.")

	return object_flux, object_flux_errors, np.array(used_indices, dtype=int)


# ============================================================
# Priors  (shared between phot and spec modes)
# ============================================================

# The priors are currently just set to flat, spanning the full range,
# with a weak 1/d prior on distance (log-flat).
def log_prior(theta, dmin=1e+0, dmax=2e+4):

	global model_to_use
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):

		Teff, logg, kzz, mh, co, d = theta

		temp_Cond_1 = (np.amin(Teff_values) <= Teff) & (Teff <= np.amax(Teff_values))
		temp_Cond_2 = (np.amin(logg_values) <= logg) & (logg <= np.amax(logg_values))
		temp_Cond_3 = (np.amin(kzz_values) <= kzz) & (kzz <= np.amax(kzz_values))
		temp_Cond_4 = (np.amin(mh_values) <= mh) & (mh <= np.amax(mh_values))
		temp_Cond_5 = (np.amin(co_values) <= co) & (co <= np.amax(co_values))
		temp_Cond_6 = (dmin <= d) & (d <= dmax)

		if temp_Cond_1 and temp_Cond_2 and temp_Cond_3 and temp_Cond_4 and temp_Cond_5 and temp_Cond_6:
			lp = -np.log10(d)
			return lp if np.isfinite(lp) else -np.inf

		return -np.inf

	if (model_to_use == 'ATMO2020'):

		Teff, logg, mh, d = theta

		temp_Cond_1 = (np.amin(Teff_values) <= Teff) & (Teff <= np.amax(Teff_values))
		temp_Cond_2 = (np.amin(logg_values) <= logg) & (logg <= np.amax(logg_values))
		temp_Cond_3 = (np.amin(mh_values) <= mh) & (mh <= np.amax(mh_values))
		temp_Cond_4 = (dmin <= d) & (d <= dmax)

		if temp_Cond_1 and temp_Cond_2 and temp_Cond_3 and temp_Cond_4:
			lp = -np.log10(d)
			return lp if np.isfinite(lp) else -np.inf

		return -np.inf


# ============================================================
# Likelihoods
# ============================================================

# Photometry likelihood. Includes a fractional model flux floor added
# in quadrature to the reported errors, to account for model systematics
# at the ~3% level.
def log_likelihood_phot(theta, flux_nJy, error_nJy):

	global model_to_use
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
		Teff, logg, kzz, mh, co, d = theta

		try: model_flux_nJy = model_phot_interp([np.log10(Teff), logg, kzz, mh, co])[0]
		except ValueError: return -np.inf

	if (model_to_use == 'ATMO2020'):
		Teff, logg, mh, d = theta

		try: model_flux_nJy = model_phot_interp([np.log10(Teff), logg, mh])[0]
		except ValueError: return -np.inf

	# Scale model fluxes to the object distance
	if (model_to_use == 'ATMO2020'):
		object_radius = 0.10276  # in units of 0.1 Rsun (model is normalised to 0.1 Rsun at 10 pc)
		model_flux_nJy = model_flux_nJy * np.square((object_radius / 0.1) * (10.0 / d))
	else:
		object_radius = 0.10276 * 2.2555823856078E-8  # in pc
		model_flux_nJy = model_flux_nJy * np.square(object_radius / d)

	condition = (~np.isnan(flux_nJy)) & (~np.isnan(error_nJy)) & (error_nJy > 0)

	# Add a fractional model floor in quadrature to account for model systematics
	if (args.frac_model_floor):
		frac_model_floor = float(args.frac_model_floor)
	else:
		frac_model_floor = 0.0

	sigma_eff = np.sqrt(error_nJy**2 + (frac_model_floor * model_flux_nJy)**2)

	chi2 = np.sum(np.square((flux_nJy[condition] - model_flux_nJy[condition]) / sigma_eff[condition]))

	if not np.isfinite(chi2):
		return -np.inf

	return -0.5 * chi2


# Interpolates the model spectrum onto the observed wavelength grid.
# Used only in spec mode.
def model_spec_wave_interp(Teff, logg, kzz, mh, co):
	global observed_wave
	global model_to_use
	global model_interp_wave

	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
		try: raw_model_flux_flam = model_spec_interp([np.log10(Teff), logg, kzz, mh, co])[0]
		except ValueError: return None

	if (model_to_use == 'ATMO2020'):
		try: raw_model_flux_flam = model_spec_interp([np.log10(Teff), logg, mh])[0]
		except ValueError: return None

	raw_model_flux_nJy = flambda_to_fnu(model_interp_wave, raw_model_flux_flam) / 1e-23 / 1e-9

	model_obs = np.interp(observed_wave, model_interp_wave, raw_model_flux_nJy)

	return model_obs


# Spectroscopy likelihood. No fractional model floor — with a full
# spectrum there are enough data points that the chi^2 is already
# well-constrained, and adding a floor tends to wash out real spectral
# features.
def log_likelihood_spec(theta, flux_nJy, error_nJy):

	global model_to_use
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
		Teff, logg, kzz, mh, co, d = theta

		model_flux_nJy = model_spec_wave_interp(Teff, logg, kzz, mh, co)
		if model_flux_nJy is None:
			return -np.inf

	if (model_to_use == 'ATMO2020'):
		Teff, logg, mh, d = theta

		# Pass dummy kzz/co values; model_spec_wave_interp ignores them for ATMO2020
		model_flux_nJy = model_spec_wave_interp(Teff, logg, -9999, mh, -9999)
		if model_flux_nJy is None:
			return -np.inf

	# Scale model fluxes to the object distance
	if (model_to_use == 'ATMO2020'):
		object_radius = 0.10276  # in units of 0.1 Rsun
		model_flux_nJy = model_flux_nJy * np.square((object_radius / 0.1) * (10.0 / d))
	else:
		object_radius = 0.10276 * 2.2555823856078E-8  # in pc
		model_flux_nJy = model_flux_nJy * np.square(object_radius / d)

	condition = (~np.isnan(flux_nJy)) & (~np.isnan(error_nJy)) & (error_nJy > 0)

	chi2 = np.sum(np.square((flux_nJy[condition] - model_flux_nJy[condition]) / error_nJy[condition]))

	if not np.isfinite(chi2):
		return -np.inf

	return -0.5 * chi2


# Posterior — routes to the correct likelihood depending on mode
def log_posterior(theta, flux_nJy, error_nJy):

	global fit_mode

	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf

	if fit_mode == 'phot':
		ll = log_likelihood_phot(theta, flux_nJy, error_nJy)
	else:
		ll = log_likelihood_spec(theta, flux_nJy, error_nJy)

	if not np.isfinite(ll):
		return -np.inf

	return lp + ll


# ============================================================
# Utility functions
# ============================================================

# Converting f_lambda to f_nu
def flambda_to_fnu(wave_ang, flux_flam):
	c = 2.998e+18
	flux_fnu = flux_flam * (wave_ang**2 / c)
	return flux_fnu

def calculate_chisq(flux, model, error):
	return np.sum(np.square((flux - model) / error))

# This is just showing off a little
def pretty_ID_at_top(ID, model_to_use):
	number_integers_in_ID = len(ID)
	extra_space = ''
	if (not number_integers_in_ID % 2):
		extra_space = ' '

	extra_star = '*'
	for star in range(0, number_integers_in_ID):
		if (star % 2):
			extra_star = extra_star + ' *'

	print(" * * * * * * * * * * * * * * * * * * * * * "+extra_star)
	print(" * * * * * * * *"+extra_space+" OBJECT ID "+ID+" * * * * * * * * ")

	if (model_to_use == 'SonoraElfOwl'):
		print(" * * * * * * * * *  Sonora Elf Owl * * * * "+extra_star)
	elif (model_to_use == 'SonoraElfOwlPH3'):
		print(" * * * * * * * Sonora Elf Owl + PH3 * * * "+extra_star)
	elif (model_to_use == 'ATMO2020'):
		print(" * * * * * * * * *  ATMO2020 * * * * * * * "+extra_star)
	elif (model_to_use == 'LOWZ'):
		print(" * * * * * * * * * * * LOWZ  * * * * * * * "+extra_star)

	print(" * * * * * * * * * * * * * * * * * * * * * "+extra_star)


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(
	description="NIFTY: Near-Infrared Fitting for T and Y Dwarfs"
)

######################
# Required Arguments #
######################

# Fit mode
parser.add_argument(
  '-mode', '--mode',
  help="Fitting mode: 'phot' (photometry) or 'spec' (spectroscopy)",
  action="store",
  type=str,
  dest="fit_mode",
  required=True
)

# Model to use
parser.add_argument(
  '-model', '--model_to_use',
  help="Model to use (SonoraElfOwl, SonoraElfOwlPH3, ATMO2020, or LOWZ)",
  action="store",
  type=str,
  dest="user_model",
  required=True
)

# Survey stub
parser.add_argument(
  '-survey_stub', '--survey_stub',
  help="Survey stub to prepend to output filenames (e.g. JADES-GS)",
  action="store",
  type=str,
  dest="name_stub",
  required=True
)

######################
# Mode-specific args #
######################

# phot mode: config file
parser.add_argument(
  '-config_file', '--config_file',
  help="[phot mode] NIRCam/MIRI filter config json file",
  action="store",
  type=str,
  dest="config_file",
  required=False
)

# phot mode: photometry catalog
parser.add_argument(
  '-photometry', '--photometry_file',
  help="[phot mode] Photometry catalog file (fits or whitespace-delimited txt)",
  action="store",
  type=str,
  dest="photometry_file",
  required=False
)

# spec mode: spectrum file
parser.add_argument(
  '-spectroscopy', '--spectroscopy_file',
  help="[spec mode] Spectrum file (3-column whitespace-delimited: wave flux error)",
  action="store",
  type=str,
  dest="spectroscopy_file",
  required=False
)

######################
# ID arguments       #
######################

# Single ID (required for spec mode; optional for phot mode if using list)
parser.add_argument(
  '-id', '--id_number',
  help="Single object ID number",
  action="store",
  type=int,
  dest="id_number",
  required=False
)

# phot mode: ID list file
parser.add_argument(
  '-idlist', '--id_number_list',
  help="[phot mode] File containing a list of ID numbers",
  action="store",
  type=str,
  dest="id_number_list",
  required=False
)

# phot mode: inline ID list
parser.add_argument(
  '-idarglist',
  help="[phot mode] Inline list of IDs, e.g. '[1,2,3]'",
  action="store",
  type=str,
  dest="idarglist",
  required=False
)

######################
# Optional Arguments #
######################

# Output folder
parser.add_argument(
  '-output',
  help="Optional output folder (default: <model>_output/)",
  action="store",
  type=str,
  dest="output_folder",
  required=False
)

# Fractional Error Floor
parser.add_argument(
  '-frac_model_floor',
  help="Optional fractional model error floor",
  action="store",
  type=str,
  dest="frac_model_floor",
  required=False
)

args = parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

	fit_mode = args.fit_mode
	if fit_mode not in ('phot', 'spec'):
		sys.exit("Error: -mode must be 'phot' or 'spec'")

	# Validate mode-specific required arguments
	if fit_mode == 'phot':
		if not args.config_file:
			sys.exit("Error: -config_file is required in phot mode")
		if not args.photometry_file:
			sys.exit("Error: -photometry is required in phot mode")
		if not (args.id_number or args.id_number_list or args.idarglist):
			sys.exit("Error: at least one of -id, -idlist, or -idarglist is required in phot mode")
	if fit_mode == 'spec':
		if not args.spectroscopy_file:
			sys.exit("Error: -spectroscopy is required in spec mode")
		if not args.id_number:
			sys.exit("Error: -id is required in spec mode")

	model_to_use = args.user_model

	if (model_to_use == 'SonoraElfOwl'):
		model_name = 'Sonora Elf Owl'
		output_name_stub = model_name.replace(' ', '_')
		model_grid_interpolator = 'Sonora_v2_interp.pkl'
	elif (model_to_use == 'SonoraElfOwlPH3'):
		model_name = 'Sonora Elf Owl + PH3'
		output_name_stub = model_name.replace(' ', '_')
		model_grid_interpolator = 'Sonora_PH3_interp.pkl'
	elif (model_to_use == 'ATMO2020'):
		model_name = 'ATMO2020'
		output_name_stub = model_name.replace(' ', '_')
		model_grid_interpolator = 'ATMO2020_interp.pkl'
	elif (model_to_use == 'LOWZ'):
		model_name = 'LOWZ'
		output_name_stub = model_name.replace(' ', '_')
		model_grid_interpolator = 'LOWZ_interp.pkl'
	else:
		sys.exit('Not a valid model. Choose from: SonoraElfOwl, SonoraElfOwlPH3, ATMO2020, LOWZ')


	# # # # # # # # # # # # # # # # # # # # # # # #
	# Print the NIFTY banner                       #
	# # # # # # # # # # # # # # # # # # # # # # # #

	print(" ")
	print("▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖   Near-Infrared")
	print("▐▛▚▖▐▌  █  ▐▌     █   ▝▚▞▘    Fitting for")
	print("▐▌ ▝▜▌  █  ▐▛▀▀▘  █    ▐▌     T and Y Dwarfs")
	print("▐▌  ▐▌▗▄█▄▖▐▌     █    ▐▌     Kevin Hainline and Jake Helton")
	print(" ")
	print("https://github.com/kevinhainline/NIFTY")
	print(" ")
	print("Fitting mode: "+fit_mode)
	print(" ")


	# # # # # # # # # # # # # # # # # # # # #
	# Read in the model interpolation grid   #
	# # # # # # # # # # # # # # # # # # # # #

	print("For these sources, we'll be using the "+model_name+" Models")
	print("Reading in the "+model_name+" Grid Interpolator: "+model_grid_interpolator)
	with open(model_grid_interpolator, 'rb') as inp:
		model_interp_object = pickle.load(inp)

	Teff_values = 10**(model_interp_object['T_eff'])
	logg_values = model_interp_object['logg']
	mh_values = model_interp_object['mh']
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
		kzz_values = model_interp_object['kzz']
		co_values = model_interp_object['co']

	number_interp_filters = len(model_interp_object['filters'])
	model_phot_interp = model_interp_object['phot_interpolator']
	model_spec_interp = model_interp_object['spec_interpolator']

	# Store the model wavelength grid as a global so spec functions can access it
	model_interp_wave = model_interp_object['wave']

	print("  The model parameter range explored:")
	print("   Teff: "+str(round(np.min(Teff_values), 1))+' to '+str(round(np.max(Teff_values), 1)))
	print("   log(g): "+str(np.min(np.round(logg_values, 2)))+' to '+str(np.max(np.round(logg_values, 2))))
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
		print("   kzz: "+str(np.min(kzz_values))+' to '+str(np.max(kzz_values)))
	print("   [M/H]: "+str(np.min(mh_values))+' to '+str(np.max(mh_values)))
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
		print("   C/O: "+str(np.min(co_values))+' to '+str(np.max(co_values)))

	print(" - - - - - - - - ")

	if (args.frac_model_floor):
		print("Setting fractional model floor: "+str(args.frac_model_floor))
		
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# Phot mode: open config file and build the object list  #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

	if fit_mode == 'phot':

		print(" - - - - - - - - ")
		filters_file = args.config_file
		print("Opening up config/filters json file: "+filters_file)

		with open(filters_file, 'r') as f:
			config = json.load(f)

		filter_name = list(config["filter_columns"].keys())
		filter_central_wavelength = np.array([
			config["filter_columns"][f]["wavelength"] for f in filter_name
		])
		wavelength_lookup = {f: config["filter_columns"][f]["wavelength"] for f in filter_name}

		number_filters = len(filter_name)
		print("    There are "+str(number_filters)+" filters to fit:")
		output_filter_string = "       "+filter_name[0]
		for filt in range(1, number_filters):
			if (not filt % 6):
				output_filter_string = output_filter_string + '\n       '+filter_name[filt]
			else:
				output_filter_string = output_filter_string + ', ' + filter_name[filt]
		print(output_filter_string)
		print(" - - - - - - - - ")

		# Build the ID list
		if (args.id_number_list):
			ID_input_file = np.loadtxt(args.id_number_list)
			if (len(ID_input_file.shape) > 1):
				ID_numbers = ID_input_file[:, 0].astype(int)
			else:
				ID_numbers = ID_input_file.astype(int)
			number_objects = len(ID_numbers)

		if (args.id_number):
			print("Working on ID "+str(args.id_number))
			ID_numbers = np.zeros(1, dtype=int)
			ID_numbers[0] = int(args.id_number)
			if (args.id_number_list):
				print("You can't specify an individual ID and a list; ignoring the list.")
			number_objects = 1

		if (args.idarglist):
			ID_numbers = np.array(ast.literal_eval(args.idarglist), dtype=int)
			number_objects = len(ID_numbers)

		# Open the photometry catalog
		photometry_file = args.photometry_file
		base, ext = os.path.splitext(photometry_file)
		if ext == ".gz":
			_, ext = os.path.splitext(base)
		ext = ext.lower()

		if ext in [".fits", ".fit", ".fts"]:
			filetype = "fits"
			all_ID = fits.open(photometry_file, memmap=True)['CIRC'].data['ID'].astype(int)
		else:
			filetype = "txt"
			try:
				photometry_data = pd.read_csv(photometry_file, sep=r'\s+', comment="#")
			except Exception as e:
				raise ValueError(f"Could not parse text file '{photometry_file}'. Make sure it's whitespace-delimited.\nSee README for expected format.\nOriginal error: {e}")
			all_ID = photometry_data["ID"].values

		if (number_objects > 1):
			time_for_MCMC_convergence = np.zeros(number_objects)-9999


	# # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# Spec mode: load the spectrum and set up single object #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # #

	if fit_mode == 'spec':

		spectroscopy_file = args.spectroscopy_file

		observed_wave_unsorted      = np.loadtxt(spectroscopy_file)[:, 0]  # microns
		observed_flux_unsorted      = np.loadtxt(spectroscopy_file)[:, 1]  # nJy
		observed_flux_errors_unsorted = np.loadtxt(spectroscopy_file)[:, 2]  # nJy

		sort_idx = np.argsort(observed_wave_unsorted)
		# Convert wavelength to Angstroms to match the model wavelength grid
		observed_wave       = observed_wave_unsorted[sort_idx] * 1e4
		object_flux         = observed_flux_unsorted[sort_idx]
		object_flux_errors  = observed_flux_errors_unsorted[sort_idx]

		ID_numbers     = np.array([int(args.id_number)], dtype=int)
		number_objects = 1


	# # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# Set up output folder                                  #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # #

	if (args.output_folder):
		if (not args.output_folder.endswith('/')):
			args.output_folder = args.output_folder + '/'
		optional_output_folder = args.output_folder
	else:
		optional_output_folder = output_name_stub+'_output/'

	if (not os.path.isdir(optional_output_folder)):
		os.mkdir(optional_output_folder)


	# # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# Main loop over objects                                #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # #

	for objid in range(0, number_objects):

		print(" ")
		print(" ")
		print("Fitting Object "+str(objid+1)+"/"+str(number_objects))
		print(" ")
		pretty_ID_at_top(str(ID_numbers[objid]), model_to_use)

		object_ID   = int(ID_numbers[objid])
		survey_stub = args.name_stub

		# In phot mode we load fluxes from the catalog here;
		# in spec mode they were already loaded above.
		if fit_mode == 'phot':

			object_index = np.where(all_ID == object_ID)[0]
			if (len(object_index) == 0):
				print("Object ID "+str(object_ID)+" not found! Skipping this object")
				continue

			object_flux, object_flux_errors, used_indices = load_flux_file(
				object_ID,
				photometry_file,
				config,
				model_interp_object,
				filetype=filetype,
				min_rel_err=0.05
			)

			if object_flux is None:
				continue

		print("     - - - - - - - - ")
		print("    Assuming source is at 1 Jupiter radius.")
		print("     - - - - - - - - ")


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# Initialise and run the MCMC sampler using emcee    #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		print("    Initializing and running the MCMC sampler")

		if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
			ndim = 6
		if (model_to_use == 'ATMO2020'):
			ndim = 4

		nsteps   = int(1e+5)
		nwalkers = int(1e+2)

		# Initial walker positions. If extra positional arguments were
		# passed (legacy behaviour) they override the defaults.
		if (len(sys.argv) == 3):
			Teff_initial = float(sys.argv[2])
		elif (len(sys.argv) == 4):
			Teff_initial = float(sys.argv[2])
			logg_initial = float(sys.argv[3])
		elif (len(sys.argv) == 5):
			Teff_initial = float(sys.argv[2])
			logg_initial = float(sys.argv[3])
			mh_initial   = float(sys.argv[4])
		else:
			Teff_initial = np.median(Teff_values)
			logg_initial = np.median(logg_values)
			mh_initial   = np.median(mh_values)

		if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
			kzz_initial = np.median(kzz_values)
			co_initial  = np.median(co_values)
		d_initial = 1e+3

		if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
			pos = np.c_[
				Teff_initial + 1e+2*np.random.randn(nwalkers),
				logg_initial + 1e-1*np.random.randn(nwalkers),
				kzz_initial  + 1e-1*np.random.randn(nwalkers),
				mh_initial   + 1e-1*np.random.randn(nwalkers),
				co_initial   + 1e-1*np.random.randn(nwalkers),
				d_initial    + 1e+2*np.random.randn(nwalkers),
			]

			labels = [
				r'$T_{\mathrm{eff}}/\mathrm{K}$',
				r'$\mathrm{log}_{10} \left( g \right)$',
				r'$\mathrm{log}_{10} \left( K_{\mathrm{zz}} \right)$',
				r'[M/H]',
				r'[C/O]',
				r'$d/\mathrm{pc}$',
			]
		if (model_to_use == 'ATMO2020'):
			pos = np.c_[
				Teff_initial + 1e+2*np.random.randn(nwalkers),
				logg_initial + 1e-1*np.random.randn(nwalkers),
				mh_initial   + 1e-1*np.random.randn(nwalkers),
				d_initial    + 1e+2*np.random.randn(nwalkers),
			]

			labels = [
				r'$T_{\mathrm{eff}}/\mathrm{K}$',
				r'$\mathrm{log}_{10} \left( g \right)$',
				r'[M/H]',
				r'$d/\mathrm{pc}$',
			]


		hfile   = survey_stub+f'_{int(object_ID):06d}.h5'
		backend = emcee.backends.HDFBackend(optional_output_folder+hfile)
		backend.reset(nwalkers, ndim)

		t1 = time.time()

		sampler = emcee.EnsembleSampler(
			nwalkers,
			ndim,
			log_posterior,
			moves=[
				(emcee.moves.DEMove(), 0.8),
				(emcee.moves.DESnookerMove(), 0.2),
			],
			args=(object_flux, object_flux_errors),
			backend=backend
		)

		# Run for up to nsteps, checking autocorrelation every 100 steps.
		# Stop early if the chain is longer than 100x the autocorrelation
		# time and that estimate has changed by less than 1%.
		temp_index = 0; temp_autocorr = np.empty(nsteps); temp_tau = np.inf

		for sample in sampler.sample(pos, iterations=nsteps, progress=True, progress_kwargs={'ncols': 100}):

			if sampler.iteration % 100: continue

			tau = sampler.get_autocorr_time(tol=0)
			temp_autocorr[temp_index] = np.mean(tau)
			temp_index += 1

			converged  = np.all(1e+2*temp_tau < sampler.iteration)
			converged &= np.all(np.abs(temp_tau-tau)/tau < 1e-2)

			if converged: break
			else: temp_tau = tau

		t2 = time.time()
		print('    '+survey_stub+f'-{int(object_ID):06d}, emcee fitting took {(t2-t1)/60.0:.1f} minutes to converge.')
		print("     - - - - - - - - ")

		if fit_mode == 'phot' and number_objects > 1:
			time_for_MCMC_convergence[objid] = (t2-t1)/60.0


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# Post-processing: burn-in, thin, build sample arrays #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		print("    Creating MCMC output for plotting")
		tau    = sampler.get_autocorr_time(quiet=True)
		burnin = int(2.0*np.amax(tau))
		thin   = int(np.amin(tau)/2.0)
		samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

		# extended_samples stores the values exactly as fitted (distance in pc)
		# for the corner plot.  model_photometry is only populated in phot mode.
		extended_samples, model_photometry = [], []

		for index, sample in enumerate(samples):

			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
				if fit_mode == 'phot':
					model_photometry.append([model_phot_interp([np.log10(sample[0]), sample[1], sample[2], sample[3], sample[4]])[0]])
				extended_samples.append([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]])

			if (model_to_use == 'ATMO2020'):
				if fit_mode == 'phot':
					model_photometry.append([model_phot_interp([np.log10(sample[0]), sample[1], sample[2]])[0]])
				extended_samples.append([sample[0], sample[1], sample[2], sample[3]])

		print("     - - - - - - - - ")


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# Corner plot                                         #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		print("    Plotting corner plot")
		fig = plt.figure(figsize=(9, 9))

		corner.corner(
			data=np.array(extended_samples),
			labels=labels,
			show_titles=True,
			bins=40,
			levels=[0.68, 0.95, 0.99],
			quantiles=[0.16, 0.50, 0.84],
			title_kwargs={'fontsize': 10},
			label_kwargs={'fontsize': 15},
			plot_datapoints=False,
			scale_hist=False,
			smooth1d=False,
			smooth=True,
			labelpad=0,
			lw=3,
			hist_kwargs={'lw': 2},
			contourf_kwargs={'lw': 2},
			fig=fig,
		)

		ax_list = fig.axes

		for ax in ax_list:
			ax.tick_params(axis='both', which='major', direction='out',
				bottom=True, top=False, left=True, right=False, length=4.5, width=2, labelsize=12)
			ax.tick_params(axis='both', which='minor', direction='out',
				bottom=True, top=False, left=True, right=False, length=3.0, width=2, labelsize=12)
			for axis in ['top', 'bottom', 'left', 'right']:
				ax.spines[axis].set_linewidth(2)

		corner_filename = survey_stub+f'_{int(object_ID):06d}_Corner_'+output_name_stub+'.png'
		plt.savefig(optional_output_folder+corner_filename,
			dpi=300, bbox_inches='tight')
		plt.close()
		print("     - - - - - - - - ")


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# Build model spectrum percentile envelopes           #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		print("    Getting median/upper/lower limits from the spectrum")
		model_spectroscopy = []
		sample_object_radius = np.zeros(len(samples[:, 0]))

		for index, sample in enumerate(samples):
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
				spectrum_ergscm2Agn = model_spec_interp([np.log10(sample[0]), sample[1], sample[2], sample[3], sample[4]])[0]
			if (model_to_use == 'ATMO2020'):
				spectrum_ergscm2Agn = model_spec_interp([np.log10(sample[0]), sample[1], sample[2]])[0]

			spectrum_nJy = flambda_to_fnu(model_interp_object['wave'], spectrum_ergscm2Agn) / 1e-23 / 1e-9

			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
				sample_object_radius[index] = 0.10276 * 2.2555823856078E-8  # in pc
				model_spectroscopy.append([spectrum_nJy * np.square(sample_object_radius[index] / sample[5])])
			if (model_to_use == 'ATMO2020'):
				sample_object_radius[index] = 0.10276
				model_spectroscopy.append([spectrum_nJy * np.square((sample_object_radius[index] / 0.1) * (10.0 / sample[3]))])

		median_values = np.percentile(model_spectroscopy, 50, axis=0)[0]
		lower_values  = np.percentile(model_spectroscopy, 16, axis=0)[0]
		upper_values  = np.percentile(model_spectroscopy, 84, axis=0)[0]

		if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
			median_distance = np.percentile(samples[:, 5], 50, axis=0)
		if (model_to_use == 'ATMO2020'):
			median_distance = np.percentile(samples[:, 3], 50, axis=0)

		median_object_radius = np.percentile(sample_object_radius, 50, axis=0)

		if fit_mode == 'phot':
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
				median_phot = np.percentile(model_photometry, 50, axis=0)[0] * np.square(median_object_radius / median_distance)
			if (model_to_use == 'ATMO2020'):
				median_phot = np.percentile(model_photometry, 50, axis=0)[0] * np.square((median_object_radius / 0.1) * (10.0 / median_distance))

		print("     - - - - - - - - ")


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# Chi-square calculation                              #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		if fit_mode == 'phot':
			only_pos_errors = np.where(object_flux_errors[used_indices] > 0)[0]
			chisq_model = calculate_chisq(
				object_flux[used_indices][only_pos_errors],
				median_phot[used_indices][only_pos_errors],
				object_flux_errors[used_indices][only_pos_errors]
			)
			number_data_points = len(object_flux[used_indices][only_pos_errors])

		if fit_mode == 'spec':
			median_value_interp = np.interp(
				observed_wave / 1e4,
				model_interp_object['wave'] / 1e4,
				median_values
			)
			only_pos_errors = np.where(object_flux_errors > 0)[0]
			chisq_model = calculate_chisq(
				object_flux[only_pos_errors],
				median_value_interp[only_pos_errors],
				object_flux_errors[only_pos_errors]
			)
			number_data_points = len(object_flux[only_pos_errors])

		if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
			number_free_parameters = 6
		if (model_to_use == 'ATMO2020'):
			number_free_parameters = 4

		degrees_of_freedom = number_data_points - number_free_parameters
		if (degrees_of_freedom > 0):
			chisq_model_reduced = chisq_model / degrees_of_freedom
		else:
			chisq_model_reduced = -9999
			print("      Warning: number of free parameters > number of data points, reduced chi-square will be -9999")


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# SED / spectrum plot                                 #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		print("    Plotting SED / spectrum")
		fig = plt.figure(figsize=(8, 4))
		ax  = plt.subplot(1, 1, 1)

		if fit_mode == 'phot':

			ax.scatter(filter_central_wavelength, object_flux[used_indices],
				s=40, color='black', alpha=1.0, zorder=15, label='Observed Photometry')
			ax.errorbar(filter_central_wavelength[only_pos_errors],
				object_flux[used_indices][only_pos_errors],
				yerr=object_flux_errors[used_indices][only_pos_errors],
				color='black', ls='None', alpha=0.8, zorder=14)
			ax.scatter(filter_central_wavelength,
				median_phot[used_indices],
				color='None', edgecolor='red', linewidth=3.0, alpha=0.9,
				marker='s', s=80, zorder=13,
				label='Model Photometry, $\\chi_{\\mathrm{red}}^2$ = '+str(round(chisq_model_reduced, 2)))
			ax.fill_between(model_interp_object['wave']/1e4, lower_values, upper_values,
				step='mid', color='red', alpha=0.1, label='68\\% Confidence', zorder=11)
			ax.step(model_interp_object['wave']/1e4, median_values,
				color='red', alpha=0.4, label='Model Flux', zorder=12)

			ax.set_title('JADES ID '+str(object_ID), fontsize=15)
			xlim_min = np.min(filter_central_wavelength) - 0.1*np.min(filter_central_wavelength)
			xlim_max = np.max(filter_central_wavelength) + 0.1*np.max(filter_central_wavelength)
			ax.set_xlim(xlim_min, xlim_max)
			ymin = np.min(object_flux[used_indices]) / 10.0
			if (ymin < 1e-2):
				ymin = 1e-2
			ymax = np.max(object_flux[used_indices]) + 10.*np.max(object_flux[used_indices])
			ax.set_ylim(ymin, ymax)
			plt.loglog()

		if fit_mode == 'spec':

			ax.step(observed_wave/1e4, object_flux,
				color='black', alpha=0.5, label='Observed Spectrum', zorder=15)
			ax.errorbar(observed_wave/1e4, object_flux, yerr=object_flux_errors,
				color='black', alpha=0.2, zorder=14)
			ax.fill_between(model_interp_object['wave']/1e4, lower_values, upper_values,
				step='mid', color='red', alpha=0.5, label='68% Confidence', zorder=17)
			ax.step(model_interp_object['wave']/1e4, median_values,
				color='red', alpha=1.0,
				label='Model Flux, $\\chi_{\\mathrm{red}}^2$ = '+str(round(chisq_model_reduced, 2)),
				zorder=18)

			ax.set_title('Source ID '+str(object_ID), fontsize=15)
			xlim_min, xlim_max = 0.8, 5.2
			ax.set_xlim(xlim_min, xlim_max)
			ymax_values = np.where((observed_wave/1e4 > xlim_min) & (observed_wave/1e4 < xlim_max))[0]
			ylim_max = 1.2 * np.max(object_flux[ymax_values])
			ylim_min = -0.1 * ylim_max
			ax.plot([xlim_min, xlim_max], [0, 0], ls='--', color='black', alpha=0.3, zorder=1)
			ax.set_ylim(ylim_min, ylim_max)

		ax.set_xlabel('Wavelength (microns)')
		ax.set_ylabel('F$_{\\nu}$ / (nJy) ')
		plt.legend()

		SED_filename = survey_stub+f'_{int(object_ID):06d}_SED_'+output_name_stub+'.png'
		plt.savefig(optional_output_folder+SED_filename,
			dpi=300, bbox_inches='tight')
		plt.close()
		print("     - - - - - - - - ")


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# Output files                                        #
		# # # # # # # # # # # # # # # # # # # # # # # # # # #

		print("    Creating Output Files ")

		final_Teff_500   = np.percentile(samples[:, 0], 50, axis=0)
		final_Teff_lower = np.percentile(samples[:, 0], 16, axis=0)
		final_Teff_upper = np.percentile(samples[:, 0], 84, axis=0)

		final_logg_500   = np.percentile(samples[:, 1], 50, axis=0)
		final_logg_lower = np.percentile(samples[:, 1], 16, axis=0)
		final_logg_upper = np.percentile(samples[:, 1], 84, axis=0)

		if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'SonoraElfOwlPH3') or (model_to_use == 'LOWZ')):
			final_kzz_500   = np.percentile(samples[:, 2], 50, axis=0)
			final_kzz_lower = np.percentile(samples[:, 2], 16, axis=0)
			final_kzz_upper = np.percentile(samples[:, 2], 84, axis=0)

			final_mh_500   = np.percentile(samples[:, 3], 50, axis=0)
			final_mh_lower = np.percentile(samples[:, 3], 16, axis=0)
			final_mh_upper = np.percentile(samples[:, 3], 84, axis=0)

			final_co_500   = np.percentile(samples[:, 4], 50, axis=0)
			final_co_lower = np.percentile(samples[:, 4], 16, axis=0)
			final_co_upper = np.percentile(samples[:, 4], 84, axis=0)

			# Distance is stored and reported in parsecs
			final_distance_500   = np.percentile(samples[:, 5], 50, axis=0)
			final_distance_lower = np.percentile(samples[:, 5], 16, axis=0)
			final_distance_upper = np.percentile(samples[:, 5], 84, axis=0)

			output_data = np.array([[object_ID,
				final_Teff_lower,     final_Teff_500,     final_Teff_upper,
				final_logg_lower,     final_logg_500,     final_logg_upper,
				final_kzz_lower,      final_kzz_500,      final_kzz_upper,
				final_mh_lower,       final_mh_500,       final_mh_upper,
				final_co_lower,       final_co_500,       final_co_upper,
				final_distance_lower, final_distance_500, final_distance_upper,
				chisq_model, chisq_model_reduced]])

			header = "# ID Teff_lower Teff Teff_upper logg_lower logg logg_upper kzz_lower kzz kzz_upper M/H_lower M/H M/H_upper co_lower co co_upper distance_pc_lower distance_pc distance_pc_upper chisq chisq_reduced"
			fmt    = ['%d'] + ['%.2f']*20

		if (model_to_use == 'ATMO2020'):
			final_mh_500   = np.percentile(samples[:, 2], 50, axis=0)
			final_mh_lower = np.percentile(samples[:, 2], 16, axis=0)
			final_mh_upper = np.percentile(samples[:, 2], 84, axis=0)

			# Distance is stored and reported in parsecs
			final_distance_500   = np.percentile(samples[:, 3], 50, axis=0)
			final_distance_lower = np.percentile(samples[:, 3], 16, axis=0)
			final_distance_upper = np.percentile(samples[:, 3], 84, axis=0)

			output_data = np.array([[object_ID,
				final_Teff_lower,     final_Teff_500,     final_Teff_upper,
				final_logg_lower,     final_logg_500,     final_logg_upper,
				final_mh_lower,       final_mh_500,       final_mh_upper,
				final_distance_lower, final_distance_500, final_distance_upper,
				chisq_model, chisq_model_reduced]])

			header = "# ID Teff_lower Teff Teff_upper logg_lower logg logg_upper M/H_lower M/H M/H_upper distance_pc_lower distance_pc distance_pc_upper chisq chisq_reduced"
			fmt    = ['%d'] + ['%.2f']*14

		# Parameter summary file
		output_fit_file_name = optional_output_folder+survey_stub+f'_{int(object_ID):06d}_parameters_'+output_name_stub+'.txt'
		np.savetxt(output_fit_file_name, output_data, fmt=fmt, delimiter=' ', header=header, comments='')

		# Model SED file (full spectrum envelope)
		output_model_sed_file_name = optional_output_folder+survey_stub+f'_{int(object_ID):06d}_SED_'+output_name_stub+'.txt'
		np.savetxt(output_model_sed_file_name,
			np.c_[model_interp_object['wave']/1e4, lower_values, median_values, upper_values],
			fmt='%f %f %f %f', header='Wavelength_um Flux_l68 Flux_50 Flux_u68')

		# Phot mode: also save model photometry at filter wavelengths
		if fit_mode == 'phot':
			output_model_photometry_file_name = optional_output_folder+survey_stub+f'_{int(object_ID):06d}_photometry_'+output_name_stub+'.txt'
			np.savetxt(output_model_photometry_file_name,
				np.c_[filter_central_wavelength, median_phot[used_indices]],
				fmt='%f %f', header='Wavelength_um Flux_Model')

		print("     - - - - - - - - ")
		print("    Output files:")
		print("        H5 file with chains:  "+optional_output_folder+hfile)
		print("        Corner plot:          "+optional_output_folder+corner_filename)
		print("        SED plot:             "+optional_output_folder+SED_filename)
		print("        SED file:             "+output_model_sed_file_name)
		if fit_mode == 'phot':
			print("        Model photometry:     "+output_model_photometry_file_name)
		print("        Output parameters:    "+output_fit_file_name)

		# Cleanup
		del backend, sampler, samples, tau
		del model_photometry, model_spectroscopy
		gc.collect()


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# Summary (phot mode, multiple objects only)              #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

	if fit_mode == 'phot' and number_objects > 1:
		valid = np.where(time_for_MCMC_convergence > 0)[0]
		if len(valid) > 0:
			min_time_index = np.argmin(time_for_MCMC_convergence[valid])
			max_time_index = np.argmax(time_for_MCMC_convergence[valid])
			median_time    = np.median(time_for_MCMC_convergence[valid])

			print(" - - - - - - - - ")
			print("  Median convergence time:  "+str(np.round(median_time, 1))+" minutes")
			print("  Minimum convergence time: ID "+str(ID_numbers[valid[min_time_index]])+", "+str(np.round(time_for_MCMC_convergence[valid[min_time_index]], 1))+" minutes")
			print("  Maximum convergence time: ID "+str(ID_numbers[valid[max_time_index]])+", "+str(np.round(time_for_MCMC_convergence[valid[max_time_index]], 1))+" minutes")

	print(" - - - - - - - - ")
	print("Done, thanks for using NIFTY!")
