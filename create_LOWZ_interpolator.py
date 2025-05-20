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

from spectres import spectres
from sedpy import observate

from multiprocessing import Pool

from tqdm import tqdm

import pickle

from scipy.interpolate import RegularGridInterpolator

#  LOWZ
#        TEFF  LOGG  METALLICITY  CTOO  LOGKZZ                                           FILENAME	
def get_teff_logg_Kzz_from_file_name(atmo_readme, best_fitting_template_name):

	source_index = np.where(atmo_readme['FILENAME'] == best_fitting_template_name)[0][0]
	teff = atmo_readme['TEFF'][source_index]
	kzz = atmo_readme['LOGKZZ'][source_index]
	grav = atmo_readme['LOGG'][source_index]
	mh = atmo_readme['METALLICITY'][source_index]
	co = atmo_readme['CTOO'][source_index]
	
	# We need to make sure that we return the logarithm of reported temperature
	return np.log10(teff), kzz, grav, mh, co

# Opening up the filter file (this can change in the future)
filters_file = 'BD_NIRCam_MIRI_filters.txt'
filter_name = np.loadtxt(filters_file, dtype = 'U10')[:,0]
filter_central_wavelength = np.loadtxt(filters_file, dtype = 'U10')[:,1].astype(float)
number_nircam_filters = len(np.where(filter_central_wavelength < 5)[0])
number_miri_filters = len(np.where(filter_central_wavelength > 5)[0])
number_filters = len(filter_name)


atmo_path = sys.argv[1]

atmo_readme = pd.read_csv(atmo_path+'LOWZ_models_index.csv')

spectra_path = glob.glob(atmo_path+'models/LOW_Z*')
number_spectra = len(spectra_path)
print("    number spectra = "+str(number_spectra))

output_teff = np.zeros(number_spectra)
output_kzz = np.zeros(number_spectra)
output_logg = np.zeros(number_spectra)
output_mh = np.zeros(number_spectra)
output_co = np.zeros(number_spectra)

# Lower resolution wavelength grid
lower_res_wave = np.arange(0.75, 15, 0.01)*1e4
number_wave_elements = len(lower_res_wave)

# these should be in terms of erg/s/cm^2/Hz
output_fluxes = np.zeros([number_filters,number_spectra])
output_spectra = np.zeros([number_wave_elements,number_spectra])

for q in tqdm(range(number_spectra)):
	
	output_spec_name = spectra_path[q].split('/')[-1]
	
	output_teff[q], output_kzz[q], output_logg[q], output_mh[q], output_co[q] = get_teff_logg_Kzz_from_file_name(atmo_readme, output_spec_name)
		
	spectra_wave = np.loadtxt(spectra_path[q])[:,0] # micron
	spectra_flux_Wm2m = np.loadtxt(spectra_path[q])[:,1] # flambda, [W/m2/m]
	
	spectra_wave_Angstrom = spectra_wave * 1e4

	# Here's the raw erg/s/cm^2/Angstrom LOWZ model
	spectra_flux_ergscm2Ang = spectra_flux_Wm2m * 1e-7
			
	# And now, here's the sorted angstrom grid. 
	angstrom_sorted = np.argsort(spectra_wave_Angstrom)

	spectrum_wave_Angstrom_sorted = spectra_wave_Angstrom[angstrom_sorted]
	spectra_flux_ergscm2Ang_sorted = spectra_flux_ergscm2Ang[angstrom_sorted]
	
	# Now we have to go and pass the spectrum through the filters to get the fluxes

	for filt in range(0, number_filters):
	
		filter_sedpy = observate.Filter("jwst_"+filter_name[filt])
		
		filter_sedpy_abmag = filter_sedpy.ab_mag(spectrum_wave_Angstrom_sorted, spectra_flux_ergscm2Ang_sorted)
		
		output_fluxes[filt, q] = 10**((filter_sedpy_abmag + 48.60)/(-2.5))
			
	output_spectra[:, q] = spectres(lower_res_wave, spectrum_wave_Angstrom_sorted, spectra_flux_ergscm2Ang_sorted)


# The LOWZ models are missing some combinations of input parameters:
# Teff  log(g) kzz [M/H] C/O
# 500.0 5.0   -1.0 -0.5 0.55
# 550.0 3.5   -1.0 -1.0 0.55
# 550.0 3.5    2.0 -1.5 0.1
# 550.0 5.25  -1.0 -1.5 0.1
# 600.0 4.5   10.0 -1.5 0.55
# 650.0 5.25   2.0 -0.5 0.55
# 800.0 5.25  10.0  0.0 0.55
# 850.0 3.5   10.0  0.0 0.55
# 850.0 5.0   10.0 -2.0 0.55
# 900.0 4.0    2.0 -2.0 0.55
# 950.0 4.0   -1.0  0.0 0.55
# 950.0 4.5   -1.0  0.0 0.55
# 950.0 5.0   10.0 -2.0 0.55

# So, here I limit to only > 600 K, and only use the C/O values of 0.1 and 0.85...
extrapolation_Teff_values = np.unique(output_teff)[2:]
extrapolation_logg_values = np.unique(output_logg)
extrapolation_kzz_values = np.unique(output_kzz)
extrapolation_mh_values = np.unique(output_mh)
extrapolation_co_values = np.unique(output_co)[[0,2]]

nbad = 0
ngood = 0
ntotal = 0
extrapolation_LOWZ_phot_grid = np.empty((len(extrapolation_Teff_values), len(extrapolation_logg_values), len(extrapolation_kzz_values), len(extrapolation_mh_values), len(extrapolation_co_values), output_fluxes.shape[0]))
extrapolation_LOWZ_spec_grid = np.empty((len(extrapolation_Teff_values), len(extrapolation_logg_values), len(extrapolation_kzz_values), len(extrapolation_mh_values), len(extrapolation_co_values), output_spectra.shape[0]))
for teff in range(0, len(extrapolation_Teff_values)):
    for logg in range(0, len(extrapolation_logg_values)):
    	for kzz in range(0, len(extrapolation_kzz_values)):
	    	for mh in range(0, len(extrapolation_mh_values)):
	    		for co in range(0, len(extrapolation_co_values)):
	    			ntotal = ntotal + 1
	    			line_index = np.where(
	    				(output_teff == extrapolation_Teff_values[teff]) & 
	    				(output_logg == extrapolation_logg_values[logg]) & 
	    				(output_kzz == extrapolation_kzz_values[kzz]) & 
	    				(output_mh == extrapolation_mh_values[mh]) & 
	    				(output_co == extrapolation_co_values[co]))[0] 
	    			if (len(line_index) > 0):
	    				extrapolation_LOWZ_phot_grid[teff, logg, kzz, mh, co, :] = output_fluxes[:,line_index[0]]/1e-23/1e-9
	    				extrapolation_LOWZ_spec_grid[teff, logg, kzz, mh, co, :] = output_spectra[:,line_index[0]]
	    				ngood = ngood + 1
	    			else:
	    				print(extrapolation_Teff_values[teff], extrapolation_logg_values[logg], extrapolation_kzz_values[kzz], extrapolation_mh_values[mh], extrapolation_co_values[co])
	    				nbad = nbad + 1

#...and I create an extrapolator, which only gets used when there's not a  
# matching model, so only in those 13 cases in the comments above.
LOWZ_phot_exterp = RegularGridInterpolator((extrapolation_Teff_values, extrapolation_logg_values, extrapolation_kzz_values, extrapolation_mh_values, extrapolation_co_values), extrapolation_LOWZ_phot_grid, method='linear', bounds_error = False, fill_value=None)
LOWZ_spec_exterp = RegularGridInterpolator((extrapolation_Teff_values, extrapolation_logg_values, extrapolation_kzz_values, extrapolation_mh_values, extrapolation_co_values), extrapolation_LOWZ_spec_grid, method='linear', bounds_error = False, fill_value=None)

Teff_values = np.unique(output_teff)
logg_values = np.unique(output_logg)
kzz_values = np.unique(output_kzz)
mh_values = np.unique(output_mh)
co_values = np.unique(output_co)

n_total = 0
n_extrapolations = 0

LOWZ_phot_grid = np.empty((len(Teff_values), len(logg_values), len(kzz_values), len(mh_values), len(co_values), output_fluxes.shape[0]))
LOWZ_spec_grid = np.empty((len(Teff_values), len(logg_values), len(kzz_values), len(mh_values), len(co_values), output_spectra.shape[0]))
for teff in range(0, len(Teff_values)):
    for logg in range(0, len(logg_values)):
    	for kzz in range(0, len(kzz_values)):
	    	for mh in range(0, len(mh_values)):
	    		for co in range(0, len(co_values)):
	    			n_total = n_total + 1
	    			line_index = np.where(
	    				(output_teff == Teff_values[teff]) & 
	    				(output_logg == logg_values[logg]) & 
	    				(output_kzz == kzz_values[kzz]) & 
	    				(output_mh == mh_values[mh]) & 
	    				(output_co == co_values[co]))[0] 
	    			if (len(line_index) > 0):
	    				LOWZ_phot_grid[teff, logg, kzz, mh, co, :] = output_fluxes[:,line_index[0]]/1e-23/1e-9
	    				LOWZ_spec_grid[teff, logg, kzz, mh, co, :] = output_spectra[:,line_index[0]]
	    			else:
	    				n_extrapolations = n_extrapolations + 1
	    				LOWZ_phot_grid[teff, logg, kzz, mh, co, :] = LOWZ_phot_exterp([Teff_values[teff], logg_values[logg], kzz_values[kzz], mh_values[mh], co_values[co]])[0]
	    				LOWZ_spec_grid[teff, logg, kzz, mh, co, :] = LOWZ_spec_exterp([Teff_values[teff], logg_values[logg], kzz_values[kzz], mh_values[mh], co_values[co]])[0]

print("I had to extrapolate "+str(n_extrapolations)+" out of "+str(n_total)+" parameter combinations. ("+str(round(100*(n_extrapolations/n_total),2))+"%)")

#...and then I package up everything, the real fluxes, and the extrapolated fluxes in the 13 cases, into one big new interpolator
LOWZ_phot_interp = RegularGridInterpolator((Teff_values, logg_values, kzz_values, mh_values, co_values), LOWZ_phot_grid, method='linear', fill_value=None)
LOWZ_spec_interp = RegularGridInterpolator((Teff_values, logg_values, kzz_values, mh_values, co_values), LOWZ_spec_grid, method='linear', fill_value=None)

output_values_data = {}
output_values_data['T_eff'] = np.unique(output_teff)
output_values_data['kzz'] = np.unique(output_kzz)
output_values_data['logg'] = np.unique(output_logg)
output_values_data['mh'] = np.unique(output_mh)
output_values_data['co'] = np.unique(output_co)
output_values_data['filters'] = filter_name
output_values_data['wave'] = lower_res_wave
output_values_data['phot_interpolator'] = LOWZ_phot_interp
output_values_data['spec_interpolator'] = LOWZ_spec_interp

# and save this to a pickle file which can be read in for fitting. 
output_values_filename = 'LOWZ_interp.pkl'
with open(output_values_filename, 'wb') as f:
	pickle.dump(output_values_data, f)
