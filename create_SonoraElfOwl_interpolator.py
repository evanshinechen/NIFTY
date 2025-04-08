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

import xarray
from spectres import spectres
from sedpy import observate

from multiprocessing import Pool

from tqdm import tqdm

import pickle

from scipy.interpolate import RegularGridInterpolator

# Sonora Elf Owl
# spectra_logzz_7.0_teff_300.0_grav_31.0_mh_1.0_co_1.5.nc
def get_teff_logg_Kzz_from_file_name(best_fitting_template_name):
	
	teff = float(best_fitting_template_name.split('/')[-1].split('_')[4])
	kzz = float(best_fitting_template_name.split('/')[-1].split('_')[2])
	grav = float(best_fitting_template_name.split('/')[-1].split('_')[6])
	mh = float(best_fitting_template_name.split('/')[-1].split('_')[8])
	co = float(best_fitting_template_name.split('/')[-1].split('_')[10].split('.nc')[0])
	
	return teff, kzz, grav, mh, co

# Opening up the filter file (this can change in the future)
filters_file = 'BD_NIRCam_MIRI_filters.txt'
filter_name = np.loadtxt(filters_file, dtype = 'U10')[:,0]
filter_central_wavelength = np.loadtxt(filters_file, dtype = 'U10')[:,1].astype(float)
number_nircam_filters = len(np.where(filter_central_wavelength < 5)[0])
number_miri_filters = len(np.where(filter_central_wavelength > 5)[0])
number_filters = len(filter_name)


sonora_path = sys.argv[1]

model_file_temp_min = np.array([275.0, 350.0, 425.0, 500.0, 575.0, 700.0, 850.0, 1000.0, 1300.0, 1600.0, 1900.0, 2200.0]) 
model_file_temp_max = np.array([325.0, 400.0, 475.0, 550.0, 650.0, 800.0, 950.0, 1200.0, 1400.0, 1800.0, 2100.0, 2400.0])

number_directories = len(model_file_temp_min)

lower_res_wave = np.arange(0.75, 15, 0.01)*1e4
number_wave_elements = len(lower_res_wave)
for x in range(0, number_directories):

	print("Tmin = "+str(model_file_temp_min[x])+" to Tmax = "+str(model_file_temp_max[x]))
	directory = sonora_path + 'output_'+str(round(model_file_temp_min[x],1))+'_'+str(round(model_file_temp_max[x],1))+'/'

	spectra_path = glob.glob(directory+'spectra*')
	number_spectra = len(spectra_path)
	print("    number spectra = "+str(number_spectra))

	# Set up the final output files
	if (x == 0):
		output_teff = np.zeros(number_spectra)
		output_kzz = np.zeros(number_spectra)
		output_grav = np.zeros(number_spectra)
		output_logg = np.zeros(number_spectra)
		output_mh = np.zeros(number_spectra)
		output_co = np.zeros(number_spectra)
	
		# these should be in terms of erg/s/cm^2/Hz
		output_fluxes = np.zeros([number_filters,number_spectra])
		output_spectra = np.zeros([number_wave_elements,number_spectra])

	# If you're in the second folder, set up new subsample directories
	else:
		subsample_teff = np.zeros(number_spectra)
		subsample_kzz = np.zeros(number_spectra)
		subsample_grav = np.zeros(number_spectra)
		subsample_logg = np.zeros(number_spectra)
		subsample_mh = np.zeros(number_spectra)
		subsample_co = np.zeros(number_spectra)
	
		# these should be in terms of erg/s/cm^2/Hz
		subsample_fluxes = np.zeros([number_filters,number_spectra])
		subsample_spectra = np.zeros([number_wave_elements,number_spectra])
	
	for q in tqdm(range(number_spectra)):
		
		output_spec_name = spectra_path[q].split('/')[-1]
		
		if (x == 0):
			output_teff[q], output_kzz[q], output_grav[q], output_mh[q], output_co[q] = get_teff_logg_Kzz_from_file_name(output_spec_name)
			output_logg[q] = np.log10(output_grav[q])
		else:
			subsample_teff[q], subsample_kzz[q], subsample_grav[q], subsample_mh[q], subsample_co[q] = get_teff_logg_Kzz_from_file_name(output_spec_name)
			subsample_logg[q] = np.log10(subsample_grav[q])
		
		ds = xarray.load_dataset(spectra_path[q])
	
		# Wavelength [um]       Fp [W/m2/m]  
		spectra_wave = ds["wavelength"].values # micron
		spectra_flux_ergscm2cm = ds['flux'].values # flambda, erg/s/cm^2/cm
		
		spectra_wave_Angstrom = spectra_wave * 1e4
	
		# Here's the raw erg/s/cm^2/Angstrom Elf Owl model
		spectra_flux_ergscm2Ang = spectra_flux_ergscm2cm / 1e+8
		
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
		output_kzz = np.append(output_kzz, subsample_kzz)
		output_grav = np.append(output_grav, subsample_grav)
		output_logg = np.append(output_logg, subsample_logg)
		output_mh = np.append(output_mh, subsample_mh)
		output_co = np.append(output_co, subsample_co)
	
		# these should be in terms of erg/s/cm^2/Hz
		output_fluxes = np.hstack((output_fluxes, subsample_fluxes))
		output_spectra = np.hstack((output_spectra, subsample_spectra))

extrapolation_Teff_values = np.unique(output_teff)
extrapolation_grav_values = np.unique(output_grav)[1:]
extrapolation_logg_values = np.unique(output_logg)[1:]
extrapolation_kzz_values = np.unique(output_kzz)
extrapolation_mh_values = np.unique(output_mh)
extrapolation_co_values = np.unique(output_co)[[0,1,2]]

ngood = 0
nbad = 0
extrapolation_Sonora_phot_grid = np.empty((len(extrapolation_Teff_values), len(extrapolation_grav_values), len(extrapolation_kzz_values), len(extrapolation_mh_values), len(extrapolation_co_values), output_fluxes.shape[0]))
extrapolation_Sonora_spec_grid = np.empty((len(extrapolation_Teff_values), len(extrapolation_grav_values), len(extrapolation_kzz_values), len(extrapolation_mh_values), len(extrapolation_co_values), output_spectra.shape[0]))
for teff in range(0, len(extrapolation_Teff_values)):
    for logg in range(0, len(extrapolation_grav_values)):
    	for kzz in range(0, len(extrapolation_kzz_values)):
	    	for mh in range(0, len(extrapolation_mh_values)):
	    		for co in range(0, len(extrapolation_co_values)):
	    			line_index = np.where(
	    				(output_teff == extrapolation_Teff_values[teff]) & 
	    				(output_grav == extrapolation_grav_values[logg]) & 
	    				(output_kzz == extrapolation_kzz_values[kzz]) & 
	    				(output_mh == extrapolation_mh_values[mh]) & 
	    				(output_co == extrapolation_co_values[co]))[0] 
	    			if (len(line_index) > 0):
	    				extrapolation_Sonora_phot_grid[teff, logg, kzz, mh, co, :] = output_fluxes[:,line_index[0]]/1e-23/1e-9
	    				extrapolation_Sonora_spec_grid[teff, logg, kzz, mh, co, :] = output_spectra[:,line_index[0]]
	    				ngood = ngood + 1
	    			else:
	    				print(Teff_values[teff], logg_values[logg], kzz_values[kzz], mh_values[mh], co_values[co])
	    				nbad = nbad + 1

# And create the phot interpolator 
print("Creating the grid interpolator")
Sonora_phot_extrapolator = RegularGridInterpolator((extrapolation_Teff_values, extrapolation_grav_values, extrapolation_kzz_values, extrapolation_mh_values, extrapolation_co_values), extrapolation_Sonora_phot_grid, method='linear', bounds_error = False, fill_value=None)
Sonora_spec_extrapolator = RegularGridInterpolator((extrapolation_Teff_values, extrapolation_grav_values, extrapolation_kzz_values, extrapolation_mh_values, extrapolation_co_values), extrapolation_Sonora_spec_grid, method='linear', bounds_error = False, fill_value=None)

Teff_values = np.unique(output_teff)
grav_values = np.unique(output_grav)
logg_values = np.unique(output_logg)
kzz_values = np.unique(output_kzz)
mh_values = np.unique(output_mh)
co_values = np.unique(output_co)

Sonora_phot_grid = np.empty((len(Teff_values), len(logg_values), len(kzz_values), len(mh_values), len(co_values), output_fluxes.shape[0]))
Sonora_spec_grid = np.empty((len(Teff_values), len(logg_values), len(kzz_values), len(mh_values), len(co_values), output_spectra.shape[0]))
for teff in range(0, len(Teff_values)):
    for logg in range(0, len(grav_values)):
    	for kzz in range(0, len(kzz_values)):
	    	for mh in range(0, len(mh_values)):
	    		for co in range(0, len(co_values)):
	    			line_index = np.where(
	    				(output_teff == Teff_values[teff]) & 
	    				(output_grav == grav_values[logg]) & 
	    				(output_kzz == kzz_values[kzz]) & 
	    				(output_mh == mh_values[mh]) & 
	    				(output_co == co_values[co]))[0] 

	    			if (len(line_index) > 0):
	    				Sonora_phot_grid[teff, logg, kzz, mh, co, :] = output_fluxes[:,line_index[0]]/1e-23/1e-9
	    				Sonora_spec_grid[teff, logg, kzz, mh, co, :] = output_spectra[:,line_index[0]]
	    			else:
	    				print("Extrapolating!")
	    				Sonora_phot_grid[teff, logg, kzz, mh, co, :] = Sonora_phot_extrapolator([Teff_values[teff], grav_values[logg], kzz_values[kzz], mh_values[mh], co_values[co]])[0]
	    				Sonora_spec_grid[teff, logg, kzz, mh, co, :] = Sonora_spec_extrapolator([Teff_values[teff], grav_values[logg], kzz_values[kzz], mh_values[mh], co_values[co]])[0]

#...and then I package up everything, the real fluxes, and the extrapolated fluxes in the 13 cases, into one big new interpolator
Sonora_phot_interp = RegularGridInterpolator((Teff_values, logg_values, kzz_values, mh_values, co_values), Sonora_phot_grid, method='linear', fill_value=None)
Sonora_spec_interp = RegularGridInterpolator((Teff_values, logg_values, kzz_values, mh_values, co_values), Sonora_spec_grid, method='linear', fill_value=None)

output_values_data = {}
output_values_data['T_eff'] = np.unique(output_teff)
output_values_data['kzz'] = np.unique(output_kzz)
output_values_data['grav'] = np.unique(output_logg)
output_values_data['mh'] = np.unique(output_mh)
output_values_data['co'] = np.unique(output_co)
output_values_data['filters'] = filter_name
output_values_data['wave'] = lower_res_wave
output_values_data['phot_interpolator'] = Sonora_phot_interp
output_values_data['spec_interpolator'] = Sonora_spec_interp

# and save this to a pickle file which can be read in for fitting. 
output_values_filename = 'Sonora_interp.pkl'
with open(output_values_filename, 'wb') as f:
	pickle.dump(output_values_data, f)
