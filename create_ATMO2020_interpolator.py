import os
import sys
import glob
import math
import astropy
import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

import pandas as pd

import xarray
from spectres import spectres
from sedpy import observate

from multiprocessing import Pool

from tqdm import tqdm

import pickle

from scipy.interpolate import RegularGridInterpolator

# ATMO 2020++ 
# spec_jwst_t1000_g2.5_m0.5_kg_g1.25.dat
def get_teff_logg_from_file_name(best_fitting_template_name):
	
	teff = float(output_spec_name.split('_')[2][1:])
	grav = float(output_spec_name.split('_')[3][1:])
	mh_raw = output_spec_name.split('_')[4]
	if mh_raw.startswith('m'):
		mh = -1.0 * float(mh_raw[1:])
	else:
		mh = float(mh_raw[1:])
	
	return teff, grav, mh

# Opening up the filter file (this can change in the future)
filters_file = 'BD_NIRCam_MIRI_filters.txt'
filter_name = np.loadtxt(filters_file, dtype = 'U10')[:,0]
filter_central_wavelength = np.loadtxt(filters_file, dtype = 'U10')[:,1].astype(float)
number_nircam_filters = len(np.where(filter_central_wavelength < 5)[0])
number_miri_filters = len(np.where(filter_central_wavelength > 5)[0])
number_filters = len(filter_name)


atmo_path = sys.argv[1]

model_file_mh_values = np.array(['m1.0', 'm0.5', 'p0', 'p0.3'])

number_directories = len(model_file_mh_values)

# Lower resolution wavelength grid
lower_res_wave = np.arange(0.75, 15, 0.01)*1e4
number_wave_elements = len(lower_res_wave)

for x in range(0, number_directories):

	print("[M/H] = "+model_file_mh_values[x])
	directory = atmo_path + 'grid_'+model_file_mh_values[x]+'/'

	spectra_path = glob.glob(directory+'spec*')
	number_spectra = len(spectra_path)
	print("    number spectra = "+str(number_spectra))

	# Set up the final output files
	if (x == 0):
		output_teff = np.zeros(number_spectra)
		output_grav = np.zeros(number_spectra)
		output_mh = np.zeros(number_spectra)
	
		# these should be in terms of erg/s/cm^2/Hz
		output_fluxes = np.zeros([number_filters,number_spectra])
		output_spectra = np.zeros([number_wave_elements,number_spectra])

	# If you're in the second folder, set up new subsample directories
	else:
		subsample_teff = np.zeros(number_spectra)
		subsample_grav = np.zeros(number_spectra)
		subsample_mh = np.zeros(number_spectra)
	
		# these should be in terms of erg/s/cm^2/Hz
		subsample_fluxes = np.zeros([number_filters,number_spectra])
		subsample_spectra = np.zeros([number_wave_elements,number_spectra])
	
	for q in tqdm(range(number_spectra)):
		
		output_spec_name = spectra_path[q].split('/')[-1]
		
		if (x == 0):
			output_teff[q], output_grav[q], output_mh[q] = get_teff_logg_from_file_name(output_spec_name)

		else:
			subsample_teff[q], subsample_grav[q], subsample_mh[q] = get_teff_logg_from_file_name(output_spec_name)

		#ds = xarray.load_dataset(spectra_path[q])
			
		spectra_wave = np.loadtxt(spectra_path[q])[:,0] # micron
		spectra_flux_Wm2m = np.loadtxt(spectra_path[q])[:,1]*1e6*3.086e+19	# flambda, [W/m2/um]
		
		spectra_wave_Angstrom = spectra_wave * 1e4
	
		# Here's the raw erg/s/cm^2/Angstrom ATMO model
		spectra_flux_ergscm2Ang = spectra_flux_Wm2m * 1e-7
				
		# And now, here's the sorted angstrom grid. 
		angstrom_sorted = np.argsort(spectra_wave_Angstrom)
	
		spectrum_wave_Angstrom_sorted = spectra_wave_Angstrom[angstrom_sorted]
		spectra_flux_ergscm2Ang_sorted = spectra_flux_ergscm2Ang[angstrom_sorted]
		
		# Now we have to go and pass the spectrum through the filters to get the fluxes
	
		for filt in range(0, number_filters):
		
			filter_sedpy = observate.Filter("jwst_"+filter_name[filt])
			
			filter_sedpy_abmag = filter_sedpy.ab_mag(spectrum_wave_Angstrom_sorted, spectra_flux_ergscm2Ang_sorted)
			if (x == 0):
				output_fluxes[filt, q] = 10**((filter_sedpy_abmag + 48.60)/(-2.5))
			else:
				subsample_fluxes[filt, q] = 10**((filter_sedpy_abmag + 48.60)/(-2.5))

		if (x == 0):
			output_spectra[:, q] = spectres(lower_res_wave, spectrum_wave_Angstrom_sorted, spectra_flux_ergscm2Ang_sorted)
		else:
			subsample_spectra[:, q] = spectres(lower_res_wave, spectrum_wave_Angstrom_sorted, spectra_flux_ergscm2Ang_sorted)

				
	if (x > 0):
		output_teff = np.append(output_teff, subsample_teff)
		output_grav = np.append(output_grav, subsample_grav)
		output_mh = np.append(output_mh, subsample_mh)
	
		# these should be in terms of erg/s/cm^2/Hz
		output_fluxes = np.hstack((output_fluxes, subsample_fluxes))
		output_spectra = np.hstack((output_spectra, subsample_spectra))

Teff_values = np.unique(output_teff)
logg_values = np.unique(output_grav)
mh_values = np.unique(output_mh)

ATMO2020_phot_grid = np.empty((len(Teff_values), len(logg_values), len(mh_values), output_fluxes.shape[0]))
ATMO2020_spec_grid = np.empty((len(Teff_values), len(logg_values), len(mh_values), output_spectra.shape[0]))
for teff in range(0, len(Teff_values)):
	for logg in range(0, len(logg_values)):
		for mh in range(0, len(mh_values)):
			line_index = np.where(
				(output_teff == Teff_values[teff]) & 
				(output_grav == logg_values[logg]) & 
				(output_mh == mh_values[mh]))[0] 
			ATMO2020_phot_grid[teff, logg, mh, :] = output_fluxes[:,line_index[0]]/1e-23/1e-9
			ATMO2020_spec_grid[teff, logg, mh, :] = output_spectra[:,line_index[0]]

ATMO2020_phot_interp = RegularGridInterpolator((Teff_values, logg_values, mh_values), ATMO2020_phot_grid, method='linear', fill_value=None)
ATMO2020_spec_interp = RegularGridInterpolator((Teff_values, logg_values, mh_values), ATMO2020_spec_grid, method='linear', fill_value=None)

output_values_data = {}
output_values_data['T_eff'] = np.unique(output_teff)
output_values_data['grav'] = np.unique(output_grav)
output_values_data['mh'] = np.unique(output_mh)
output_values_data['filters'] = filter_name
output_values_data['wave'] = lower_res_wave
output_values_data['phot_interpolator'] = ATMO2020_phot_interp
output_values_data['spec_interpolator'] = ATMO2020_spec_interp

# and save this to a pickle file which can be read in for fitting. 
output_values_filename = 'ATMO2020_interp.pkl'
with open(output_values_filename, 'wb') as f:
	pickle.dump(output_values_data, f)
