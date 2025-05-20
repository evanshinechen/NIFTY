#!/usr/bin/env python
"""\
NIFTY: Near-Infrared Fitting for T and Y Dwarfs
Kevin Hainline, Jake Helton

Usage: Run an MCMC fit using either the Sonora Elf Owl, ATMO2020, or LOWZ Models 
       to NIRCam and MIRI photometry. Produces an h5 file, as well as a corner 
       plot and an SED showing the fit compared to the photometry.
"""
# Imports necessary miscellaneous modules
import os
import ast
import sys
import time
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

# The priors are currently just set to flat, spanning the full range
def log_prior(theta, dmin=1e+0, dmax=2e+4):
    
    # Get which model we're using
	global model_to_use
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
    
		Teff, logg, kzz, mh, co, d = theta
		
		temp_Cond_1 = (np.amin(Teff_values) <= Teff) & (Teff <= np.amax(Teff_values))
		temp_Cond_2 = (np.amin(logg_values) <= logg) & (logg <= np.amax(logg_values))
		temp_Cond_3 = (np.amin(kzz_values) <= kzz) & (kzz <= np.amax(kzz_values))
		temp_Cond_4 = (np.amin(mh_values) <= mh) & (mh <= np.amax(mh_values))
		temp_Cond_5 = (np.amin(co_values) <= co) & (co <= np.amax(co_values))
		temp_Cond_6 = (dmin <= d) & (d <= dmax)
		
		if temp_Cond_1 and temp_Cond_2 and temp_Cond_3 and temp_Cond_4 and temp_Cond_5 and temp_Cond_6: return -np.log10(d)
		return -np.inf

	if (model_to_use == 'ATMO2020'):

		Teff, logg, mh, d = theta
		
		temp_Cond_1 = (np.amin(Teff_values) <= Teff) & (Teff <= np.amax(Teff_values))
		temp_Cond_2 = (np.amin(logg_values) <= logg) & (logg <= np.amax(logg_values))
		temp_Cond_3 = (np.amin(mh_values) <= mh) & (mh <= np.amax(mh_values))
		temp_Cond_4 = (dmin <= d) & (d <= dmax)
		
		if temp_Cond_1 and temp_Cond_2 and temp_Cond_3 and temp_Cond_4: return -np.log10(d)
		return -np.inf
	
# The log likelihood function, that calculates the goodness of fit
def log_likelihood(theta, flux_nJy, error_nJy):

    # Get which model we're using
	global model_to_use
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
		Teff, logg, kzz, mh, co, d = theta
		
		try: model_flux_nJy = model_phot_interp([np.log10(Teff), logg, kzz, mh, co])[0]#[0:14]
		except ValueError: return -np.inf

	if (model_to_use == 'ATMO2020'):
		Teff, logg, mh, d = theta
		
		try: model_flux_nJy = model_phot_interp([np.log10(Teff), logg, mh])[0]#[0:14]
		except ValueError: return -np.inf

	object_radius = 0.10276 * 2.2555823856078E-8 # in pc
	
	# Have to multiply the model fluxes by a value that depends on the distance
	model_flux_nJy = model_flux_nJy*np.square(object_radius/d)
	condition = (~np.isnan(flux_nJy)) & (~np.isnan(error_nJy)) & (error_nJy > 0)
	chi2 = np.sum(np.square((flux_nJy[condition]-model_flux_nJy[condition])/error_nJy[condition]))
    
	return -0.5*chi2

# The posterior function
def log_posterior(theta, flux_nJy, error_nJy):
    
	lp = log_prior(theta)
	if not np.isfinite(lp): return -np.inf
	return lp + log_likelihood(theta, flux_nJy, error_nJy)

# Converting f_lambda to f_nu
def flambda_to_fnu(wave_ang, flux_flam):
	c = 2.998e+18
	flux_fnu = flux_flam * (wave_ang**2 / c)
	return flux_fnu

# This is just showing off a little
def pretty_ID_at_top(ID, model_to_use):
	number_integers_in_ID = len(ID)
	extra_space = ''
	if (not number_integers_in_ID%2):
		extra_space = ' '
	
	extra_star = '*'
	for star in range(0, number_integers_in_ID):
		if (star%2): 
			extra_star = extra_star +' *'
	
	print(" * * * * * * * * * * * * * * * * * * * * * "+extra_star)
	print(" * * * * * * * *"+extra_space+" OBJECT ID "+ID+" * * * * * * * * ")

	if (model_to_use == 'SonoraElfOwl'):
		print(" * * * * * * * * *  Sonora Elf Owl * * * * "+extra_star)
	elif (model_to_use == 'ATMO2020'):
		print(" * * * * * * * * *  ATMO2020 * * * * * * * "+extra_star)
	elif (model_to_use == 'LOWZ'):
		print(" * * * * * * * * * * * LOWZ  * * * * * * * "+extra_star)

	print(" * * * * * * * * * * * * * * * * * * * * * "+extra_star)

######################
# Required Arguments #
######################

parser = argparse.ArgumentParser()

# Model To Use
parser.add_argument(
  '-model','--model_to_use',
  help="Model to use (SonoraElfOwl, ATMO2020, or LOWZ)",
  action="store",
  type=str,
  dest="user_model",
  required=True
)

# Filter File
parser.add_argument(
  '-filters','--filter_file',
  help="NIRCam or MIRI Filters to Fit",
  action="store",
  type=str,
  dest="filters",
  required=True
)

# JADES Photometry File
parser.add_argument(
  '-photometry','--photometry_file',
  help="JADES Photometry File",
  action="store",
  type=str,
  dest="photometry_file",
  required=True
)

# JADES Survey Stub
parser.add_argument(
  '-survey_stub','--survey_stub',
  help="JADES-GS or JADES-GN",
  action="store",
  type=str,
  dest="name_stub",
  required=True
)

# JADES Aperture Size
parser.add_argument(
  '-aperture','--aperture_size',
  help="JADES Aperture Size",
  action="store",
  type=str,
  dest="aperture_size",
  required=True
)


######################
# Optional Arguments #
######################

# ID number
parser.add_argument(
  '-id','--id_number',
  help="ID Number?",
  action="store",
  type=int,
  dest="id_number",
  required=False
)

# ID list
parser.add_argument(
  '-idlist','--id_number_list',
  help="List of ID Numbers?",
  action="store",
  type=str,
  dest="id_number_list",
  required=False
)

# command line argument list of objects
parser.add_argument(
  '-idarglist',
  help="Command line argument list of objects",
  action="store",
  type=str,
  dest="idarglist",
  required=False
)

# command line argument list of objects
parser.add_argument(
  '-output',
  help="Optional output folder",
  action="store",
  type=str,
  dest="output_folder",
  required=False
)

args=parser.parse_args()



if __name__ == '__main__':

	model_to_use = args.user_model#'ATMO2020' # 'ATMO2020', 'LOWZ'
	
	if (model_to_use == 'SonoraElfOwl'):
		model_name = 'Sonora Elf Owl'
		output_name_stub = model_name.replace(' ','_')
		model_grid_interpolator = 'Sonora_v2_interp.pkl'
	elif (model_to_use == 'ATMO2020'):
		model_name = 'ATMO2020'
		output_name_stub = model_name.replace(' ','_')
		model_grid_interpolator = 'ATMO2020_interp.pkl'
	elif (model_to_use == 'LOWZ'):
		model_name = 'LOWZ'
		output_name_stub = model_name.replace(' ','_')
		model_grid_interpolator = 'LOWZ_interp.pkl'
	else:
		sys.exit('Not a valid model')
	
	
	# # # # # # # # # # # # # # # # # # # # # # # #
	# Make the code print a pretty ID at the top  #
	# # # # # # # # # # # # # # # # # # # # # # # #
	
	print(" ")
	print("▗▖  ▗▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖   Near-Infrared")
	print("▐▛▚▖▐▌  █  ▐▌     █   ▝▚▞▘    Fitting for")
	print("▐▌ ▝▜▌  █  ▐▛▀▀▘  █    ▐▌     T and Y Dwarfs")
	print("▐▌  ▐▌▗▄█▄▖▐▌     █    ▐▌     Kevin Hainline and Jake Helton")
	print(" ")
	print("https://github.com/kevinhainline/NIFTY")
	print(" ")
	
	
	# # # # # # # # # # # # # # #
	# Open up the filters file  #
	# # # # # # # # # # # # # # #
	
	print(" - - - - - - - - ")
	filters_file = args.filters
	print("Opening up filters file: "+filters_file)
	filter_name = np.loadtxt(filters_file, dtype = 'U10')[:,0]
	filter_central_wavelength = np.loadtxt(filters_file, dtype = 'U10')[:,1].astype(float)
	number_nircam_filters = len(np.where(filter_central_wavelength < 5)[0])
	number_miri_filters = len(np.where(filter_central_wavelength > 5)[0])
	number_filters = len(filter_name)
	print("    There are "+str(number_filters)+" filters to fit:")
	output_filter_string = "       "+filter_name[0]
	for filt in range(1, number_filters):
		if (not filt%6):
			output_filter_string = output_filter_string + '\n       '+filter_name[filt]
		else:
			output_filter_string = output_filter_string + ', ' + filter_name[filt]
	print(output_filter_string)
	print(" - - - - - - - - ")
	
	# # # # # # # # # # # # # # # # # # # # # 
	# Read in the model interpolation grid  #
	# # # # # # # # # # # # # # # # # # # # #
		
	print("For these sources, we'll be using the "+model_name+" Models")
	print("Reading in the "+model_name+" Grid Interpolator: "+model_grid_interpolator)
	with open(model_grid_interpolator, 'rb') as inp:
		model_interp_object = pickle.load(inp)
	
	Teff_values = 10**(model_interp_object['T_eff'])
	logg_values = model_interp_object['logg']
	mh_values = model_interp_object['mh']
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
		kzz_values = model_interp_object['kzz']
		co_values = model_interp_object['co']
	
	number_interp_filters = len(model_interp_object['filters'])
	model_phot_interp = model_interp_object['phot_interpolator']
	model_spec_interp = model_interp_object['spec_interpolator']
	
	print("  The model parameter range explored:")
	print("   Teff: "+str(round(np.min(Teff_values),1))+' to '+str(round(np.max(Teff_values),1)))
	print("   log(g): "+str(np.min(np.round(logg_values,2)))+' to '+str(np.max(np.round(logg_values,2))))
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
		print("   kzz: "+str(np.min(kzz_values))+' to '+str(np.max(kzz_values)))
	print("   [M/H]: "+str(np.min(mh_values))+' to '+str(np.max(mh_values)))
	if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
		print("   C/O: "+str(np.min(co_values))+' to '+str(np.max(co_values)))
	
	print(" - - - - - - - - ")
	
	# # # # # # # # # # #
	# Get Object Fluxes #
	# # # # # # # # # # #

	# Create a subsample given an ID number list
	if (args.id_number_list):
		ID_input_file = np.loadtxt(args.id_number_list)
		if (len(ID_input_file.shape) > 1):
			ID_numbers = ID_input_file[:,0].astype(int)
		else:
			ID_numbers = ID_input_file.astype(int)

		number_objects = len(ID_numbers)
	
	# ...or, just use a single ID
	if (args.id_number):
		ID_numbers = np.zeros(1, dtype = int)
		ID_numbers[0] = int(args.id_number)
		#number_input_objects = len(ID_numbers)
		if (args.id_number_list):
			"You can't specify an individual ID and a list, ignoring the list."
	
		number_objects = 1
		
	# ...or, specify the IDs as an argument list
	if (args.idarglist):
		ID_numbers = np.array(ast.literal_eval(args.idarglist), dtype = int)
		
		number_objects = len(ID_numbers)

	# Break if the user doesn't provide IDs
	if ((not args.id_number_list) and (not args.id_number) and (not args.idarglist)):
		sys.exit('No ID numbers provided')

	# Creating the optional output folder
	if (args.output_folder):
		if (not args.output_folder.endswith('/')):
			args.output_folder = args.output_folder + '/'
		optional_output_folder = args.output_folder
	else:
		optional_output_folder = output_name_stub+'_output/'
	
	if (not os.path.isdir(optional_output_folder)):
		os.mkdir(optional_output_folder)

	# We only need to open the input photometry file once, at the 
	# start. 
	photometry_file = args.photometry_file
	photometry_fits = fits.open(photometry_file, memmap = True)
	all_ID = photometry_fits['CIRC'].data['ID'].astype(int)
	
	if (number_objects > 1):
		time_for_MCMC_convergence = np.zeros(number_objects)-9999

	# Go through the individual sources and fit them one by one! 
	for objid in range(0, number_objects):
		
		print(" ")
		print(" ")
		print("Fitting Object "+str(objid+1)+"/"+str(number_objects))
		print(" ")
		pretty_ID_at_top(str(ID_numbers[objid]), model_to_use)
		
		object_ID = int(ID_numbers[objid])
		fluxes_to_use = args.aperture_size
		survey_stub = args.name_stub 

		object_index = np.where(all_ID == object_ID)[0]
		if (len(object_index) == 0):
			print("Object ID"+str(object_ID)+" not found! Skipping this object")
		else:
			object_index = object_index[0]

			object_RA = photometry_fits['CIRC'].data['RA'][object_index]
			object_DEC = photometry_fits['CIRC'].data['DEC'][object_index]
					
			min_rel_err = 0.05
			object_flux = np.zeros(number_interp_filters)-9999
			object_flux_errors = np.zeros(number_interp_filters)-9999
			used_indices = np.empty(0, dtype = int)
			print("    Getting object fluxes for ID "+str(object_ID))
			for q in range(0, number_interp_filters):
				interp_filter_index = np.where(filter_name == model_interp_object['filters'][q])[0]
			
				if (len(interp_filter_index) > 0):
					used_indices = np.append(used_indices, np.array([q], dtype = int))
					object_flux[q] = photometry_fits['CIRC'].data[model_interp_object['filters'][q]+'_'+fluxes_to_use][object_index]
					try:
						object_flux_errors[q] = photometry_fits['CIRC'].data[model_interp_object['filters'][q]+'_'+fluxes_to_use+'_en'][object_index]
					except KeyError:
						object_flux_errors[q] = photometry_fits['CIRC'].data[model_interp_object['filters'][q]+'_'+fluxes_to_use+'_e'][object_index]
						
					# make sure that the SNR can't get too high, based on min_rel_err.
					if (((object_flux_errors[q]/object_flux[q]) < min_rel_err) & ((object_flux_errors[q]/object_flux[q]) > 0)):
						object_flux_errors[q] = object_flux[q] * min_rel_err
						print("     Updating Flux in filter "+str(model_interp_object['filters'][q])+" to be at the minimum relative error.")
			
			print("     - - - - - - - - ")
			print("    Assuming source is at 1 Jupiter radius.") 
			print("     - - - - - - - - ")
			
			# # # # # # # # # # # # # # # # # # # # # # # # # # # 
			# Initializes and runs the MCMC sampler using emcee #
			# # # # # # # # # # # # # # # # # # # # # # # # # # # 
			
			print("    Initializing and running the MCMC sampler")
			
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
				ndim = 6 # Number of dimensions
			if (model_to_use == 'ATMO2020'):
				ndim = 4 # Number of dimensions
				
			nsteps = int(1e+5) # Number of steps
			nwalkers = int(1e+2) # Number of walkers
			Maximum_SNR = 20.0 # Maximum signal-to-noise ratio for photometry
			
			# Allow a user specified initial guess, Teff, logg, mh
			if (len(sys.argv) == 3):
				Teff_initial = float(sys.argv[2])
			elif (len(sys.argv) == 4):
				Teff_initial = float(sys.argv[2])
				logg_initial = float(sys.argv[3])
			elif (len(sys.argv) == 5):
				Teff_initial = float(sys.argv[2])
				logg_initial = float(sys.argv[3])
				mh_initial = float(sys.argv[4])
			else:
				Teff_initial = np.median(Teff_values)
				logg_initial = np.median(logg_values)
				mh_initial = np.median(mh_values)
			
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
				kzz_initial = np.median(kzz_values)
				co_initial = np.median(co_values)
			d_initial = 1e+3
			
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
				pos = np.c_[
					Teff_initial + 1e+2*np.random.randn(nwalkers),
					logg_initial + 1e-1*np.random.randn(nwalkers),
					kzz_initial + 1e-1*np.random.randn(nwalkers),
					mh_initial + 1e-1*np.random.randn(nwalkers),
					co_initial + 1e-1*np.random.randn(nwalkers),
					d_initial + 1e+2*np.random.randn(nwalkers),
				]
				
				labels = [
					r'$T_{\mathrm{eff}}/\mathrm{K}$',
					r'$\mathrm{log}_{10} \left( g \right)$',
					r'$\mathrm{log}_{10} \left( K_{\mathrm{zz}} \right)$',
					r'[M/H]',
					r'[C/O]',
					r'$d/\mathrm{kpc}$',
				]
			if (model_to_use == 'ATMO2020'):
				pos = np.c_[
					Teff_initial + 1e+2*np.random.randn(nwalkers),
					logg_initial + 1e-1*np.random.randn(nwalkers),
					mh_initial + 1e-1*np.random.randn(nwalkers),
					d_initial + 1e+2*np.random.randn(nwalkers),
				]
				
				labels = [
					r'$T_{\mathrm{eff}}/\mathrm{K}$',
					r'$\mathrm{log}_{10} \left( g \right)$',
					r'[M/H]',
					r'$d/\mathrm{kpc}$',
				]
			
			
			hfile = survey_stub+f'_{int(object_ID):06d}.h5'
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
				
			# We run the chain for up to 100,000 steps, checking
			# the autocorrelation time every 100 steps. If the chain 
			# is longer than 100 times the estimated autocorrelation
			# time, and if this estimate changed by less than 1%,
			# we will consider things converged.
			
			temp_index = 0; temp_autocorr = np.empty(nsteps); temp_tau = np.inf
			
			for sample in sampler.sample(pos, iterations=nsteps, progress=True, progress_kwargs={'ncols':100}):
				
				if sampler.iteration % 100: continue
				
				tau = sampler.get_autocorr_time(tol=0)
				temp_autocorr[temp_index] = np.mean(tau)
				temp_index += 1
				
				converged = np.all(1e+2*temp_tau < sampler.iteration)
				converged &= np.all(np.abs(temp_tau-tau)/tau < 1e-2)
				
				if converged: break
				else: temp_tau = tau
			
			#clear_output()
			t2 = time.time()
			print('    '+survey_stub+f'-{int(object_ID):06d}, emcee fitting took {(t2-t1)/60.0:.1f} minutes to converge.')
			print("     - - - - - - - - ")
	
			if (number_objects > 1):
				time_for_MCMC_convergence[objid] = (t2-t1)/60.0
	
			 
			# Investigates the autocorrelation time
			# Which we use to burn-in and thin the chain
			
			print("    Creating MCMC output for plotting")
			tau = sampler.get_autocorr_time(quiet = True) # quiet = True is so that if the chain is too short, the program doesn't quit with a warning.
			burnin = int(2.0*np.amax(tau))
			thin = int(np.amin(tau)/2.0)
			samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
			extended_samples, model_photometry = [], []
			
			for index, sample in enumerate(samples):
				
				if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
					#model_photometry.append([model_phot_interp([sample[0], sample[1], sample[2], sample[3], sample[4]])[0][0:14]])
					model_photometry.append([model_phot_interp([np.log10(sample[0]), sample[1], sample[2], sample[3], sample[4]])[0]])
					extended_samples.append([sample[0], sample[1],  sample[2], sample[3], sample[4], sample[5]/1000])
				if (model_to_use == 'ATMO2020'):
					#model_photometry.append([model_phot_interp([sample[0], sample[1], sample[2]])[0][0:14]])
					model_photometry.append([model_phot_interp([np.log10(sample[0]), sample[1], sample[2]])[0]])
					extended_samples.append([sample[0], sample[1],  sample[2], sample[3]/1000])
			
			print("     - - - - - - - - ")
			
			print("    Plotting corner plot")
			fig = plt.figure(figsize=(9, 9))
			
			corner.corner(
				data=np.array(extended_samples),
				labels=labels,
				show_titles=True,
				#title_fmt='.0f', # I think that the number of significant digits is fine! 
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
			
				for axis in ['top','bottom','left','right']: 
			
					ax.spines[axis].set_linewidth(2)
			
			corner_filename = survey_stub+f'_{int(object_ID):06d}_Corner_'+output_name_stub+'.png'
			plt.savefig(optional_output_folder+corner_filename, 
				dpi=300, bbox_inches='tight')
			
			plt.close()
			print("     - - - - - - - - ")
			
			# # # # # # # # # # # # # # # # # # #
			# Create Spectrum for Plotting SED  #
			# # # # # # # # # # # # # # # # # # #
			
			# We assume that these objects are at a Jupiter radius
			object_radius = 0.10276 * 2.2555823856078E-8 # in pc
			
			print("    Getting median/upper/lower limits from the spectrum")
			model_spectroscopy = []
			sample_object_radius = np.zeros(len(samples[:,0]))
			for index, sample in enumerate(samples):
				if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
					spectrum_ergscm2Agn = model_spec_interp([np.log10(sample[0]), sample[1], sample[2], sample[3], sample[4]])[0] # in erg/s/cm^2/Angstrom
				if (model_to_use == 'ATMO2020'):
					spectrum_ergscm2Agn = model_spec_interp([np.log10(sample[0]), sample[1], sample[2]])[0] # in erg/s/cm^2/Angstrom
				
				spectrum_nJy = flambda_to_fnu(model_interp_object['wave'], spectrum_ergscm2Agn)/1e-23/1e-9
			
				sample_object_radius[index] = 0.10276 * 2.2555823856078E-8 # in pc
			
				if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
					model_spectroscopy.append([spectrum_nJy * np.square(sample_object_radius[index]/sample[5])])
				if (model_to_use == 'ATMO2020'):
					model_spectroscopy.append([spectrum_nJy * np.square(sample_object_radius[index]/sample[3])])
			
			median_values = np.percentile(model_spectroscopy, 50, axis=0)[0]   # 50th percentile (median)
			lower_values = np.percentile(model_spectroscopy, 16, axis=0)[0]   # 16th percentile (lower bound)
			upper_values = np.percentile(model_spectroscopy, 84, axis=0)[0]   # 84th percentile (upper bound)
			
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
				median_distance = np.percentile(samples[:,5], 50, axis=0)
			if (model_to_use == 'ATMO2020'):
				median_distance = np.percentile(samples[:,3], 50, axis=0)
			
			median_object_radius = np.percentile(sample_object_radius, 50, axis=0)
			median_phot = np.percentile(model_photometry, 50, axis=0)[0]* np.square(median_object_radius/median_distance)   # 50th percentile (median)
				
			print("     - - - - - - - - ")
			
			print("    Plotting spectrum with SED")
			fig = plt.figure(figsize=(8, 4))
			ax = plt.subplot(1,1,1)
			
			
			ax.scatter(filter_central_wavelength, object_flux[used_indices], s = 40, color = 'black', alpha = 1.0, zorder = 15, label = 'Observed Photometry')
			only_pos_errors = np.where(object_flux_errors[used_indices] > 0)[0]
			ax.errorbar(filter_central_wavelength[only_pos_errors], object_flux[used_indices][only_pos_errors], yerr = object_flux_errors[used_indices][only_pos_errors], color = 'black', ls = 'None', alpha = 0.8, zorder = 14)
			
			ax.scatter(filter_central_wavelength, median_phot[used_indices], color = 'None', edgecolor = 'red', linewidth = 3.0, alpha = 0.9, marker = 's', s = 80, zorder = 13, label = 'Model Photometry')
			ax.fill_between(model_interp_object['wave']/1e4, lower_values, upper_values, step='mid', color='red', alpha=0.1, label='68\% Confidence', zorder = 11)
			ax.step(model_interp_object['wave']/1e4, median_values, color = 'red', alpha = 0.4, label = 'Model Flux', zorder = 12)
			
			ax.set_title('JADES ID '+str(object_ID), fontsize = 15)
			ax.set_xlim(0.75, 5.2)
			ymin = np.min(object_flux[used_indices])/10.0
			if (ymin < 1e-2):
				ymin = 1e-2
			ymax = np.max(object_flux[used_indices]) + 10.*np.max(object_flux[used_indices])
			ax.set_ylim(ymin, ymax)#10000.0)
			ax.set_xlabel('Wavelength (microns)')
			ax.set_ylabel('F$_{\\nu}$ / (nJy) ')
			plt.legend()
			plt.loglog()
			
			SED_filename = survey_stub+f'_{int(object_ID):06d}_SED_'+output_name_stub+'.png'
			plt.savefig(optional_output_folder+SED_filename, 
				dpi=300, bbox_inches='tight')
			
			plt.close()
			
			print("     - - - - - - - - ")
			
			print("    Creating Output File ")
			
			final_Teff_500 = np.percentile(samples[:,0], 50, axis=0)
			final_Teff_lower = np.percentile(samples[:,0], 16, axis=0)
			final_Teff_upper = np.percentile(samples[:,0], 84, axis=0)
			
			final_logg_500 = np.percentile(samples[:,1],  50, axis=0)
			final_logg_lower = np.percentile(samples[:,1],  16, axis=0)
			final_logg_upper = np.percentile(samples[:,1],  84, axis=0)
			
			if ((model_to_use == 'SonoraElfOwl') or (model_to_use == 'LOWZ')):
				final_kzz_500 = np.percentile(samples[:,2],  50, axis=0)
				final_kzz_lower = np.percentile(samples[:,2],  16, axis=0)
				final_kzz_upper = np.percentile(samples[:,2],  84, axis=0)
				
				final_mh_500 = np.percentile(samples[:,3],  50, axis=0)
				final_mh_lower = np.percentile(samples[:,3],  16, axis=0)
				final_mh_upper = np.percentile(samples[:,3],  84, axis=0)
				
				final_co_500 = np.percentile(samples[:,4],  50, axis=0)
				final_co_lower = np.percentile(samples[:,4],  16, axis=0)
				final_co_upper = np.percentile(samples[:,4],  84, axis=0)
				
				final_distance_500 = np.percentile(samples[:,5],  50, axis=0)/1e3
				final_distance_lower = np.percentile(samples[:,5],  16, axis=0)/1e3
				final_distance_upper = np.percentile(samples[:,5],  84, axis=0)/1e3
			
				# Combine values into a single row
				output_data = np.array([[object_ID, final_Teff_lower, final_Teff_500, final_Teff_upper,
					final_logg_lower, final_logg_500, final_logg_upper,
					final_kzz_lower, final_kzz_500, final_kzz_upper,
					final_mh_lower, final_mh_500, final_mh_upper,
					final_co_lower, final_co_500, final_co_upper,
					final_distance_lower, final_distance_500, final_distance_upper]])
				
				# Define the header
				header = "# ID Teff_lower Teff Teff_upper logg_lower logg logg_upper kzz_lower kzz kzz_upper M/H_lower M/H M/H_upper co_lower co co_upper distance_lower distance distance_upper"
				
				# Define the format string
				fmt = ['%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f']  
			
			if (model_to_use == 'ATMO2020'):
				final_mh_500 = np.percentile(samples[:,2],  50, axis=0)
				final_mh_lower = np.percentile(samples[:,2],  16, axis=0)
				final_mh_upper = np.percentile(samples[:,2],  84, axis=0)
				
				final_distance_500 = np.percentile(samples[:,3],  50, axis=0)/1e3
				final_distance_lower = np.percentile(samples[:,3],  16, axis=0)/1e3
				final_distance_upper = np.percentile(samples[:,3],  84, axis=0)/1e3
			
				# Combine values into a single row
				output_data = np.array([[object_ID, final_Teff_lower, final_Teff_500, final_Teff_upper,
					final_logg_lower, final_logg_500, final_logg_upper,
					final_mh_lower, final_mh_500, final_mh_upper,
					final_distance_lower, final_distance_500, final_distance_upper]])
				
				# Define the header
				header = "# ID Teff_lower Teff Teff_upper logg_lower logg logg_upper M/H_lower M/H M/H_upper distance_lower distance distance_upper"
				
				# Define the format string
				fmt = ['%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f']  
				
			# Save to file
			output_fit_file_name = optional_output_folder+survey_stub+f'_{int(object_ID):06d}_parameters_'+output_name_stub+'.txt'
			np.savetxt(output_fit_file_name, output_data, fmt=fmt, delimiter=' ', header=header, comments='')
			
			
			print("     - - - - - - - - ")
			print("    Output files:")
			print("        H5file with chains: "+optional_output_folder+hfile)
			print("        corner plot:        "+optional_output_folder+corner_filename)
			print("        SED plot:           "+optional_output_folder+SED_filename)
			print("        output parameters:  "+output_fit_file_name)
		
			# Doing a little cleanup here at the end. 
			del backend
			del sampler
			del samples
			del tau
			del model_photometry
			del model_spectroscopy
			gc.collect()

	if (number_objects > 1):
		min_time_index = np.argmin(time_for_MCMC_convergence[np.where(time_for_MCMC_convergence > 0)[0]])
		max_time_index = np.argmax(time_for_MCMC_convergence[np.where(time_for_MCMC_convergence > 0)[0]])
		median_time = np.median(time_for_MCMC_convergence[np.where(time_for_MCMC_convergence > 0)[0]])

		print(" - - - - - - - - ")
		print("  Median convergence time: "+str(np.round(median_time,1))+" minutes")
		print("  Minimum convergence time: ID "+str(ID_numbers[min_time_index])+", "+str(np.round(time_for_MCMC_convergence[min_time_index],1))+" minutes")
		print("  Maximum convergence time: ID "+str(ID_numbers[max_time_index])+", "+str(np.round(time_for_MCMC_convergence[max_time_index],1))+" minutes")
	
	print(" - - - - - - - - ")
	print("Done, thanks for using NIFTY!")
