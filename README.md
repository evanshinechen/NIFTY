# NIFTY
*Near-Infrared Fitting for T and Y Dwarfs*

*Kevin Hainline and Jake Helton*

NIFTY is a code designed to fit JWST NIRCam and MIRI photometry for cold brown 
dwarfs with the LOWZ (Meisner et al. 2021), ATMO2020 (Philips et al. 2020), or Sonora 
Elf Owl (Mukherjee et al. 2024) models, in a Bayesian framework, using the emcee
sampler (https://emcee.readthedocs.io/en/stable/user/sampler/). 

The code creates interpolation grids from the models themselves, and then runs a fit 
directly on the input photometry, producing a corner plot, a fit SED, and a file
with the 16th, 50th, and 84th percentiles on the fit parameters. In our tests,
fits take < 10 minutes per object for ~14 photometric bands, but some take longer
when convergence isn't quick. 

## Installation 

Installation can be done through conda, or micromamba: 

```
(base) % conda env create -f environment.yml
```

This will create a new environment, `NIFTY`, from which you can run `NIFTY`.

You'll also need to install some packages through pip:
```
(base) % conda activate NIFTY
(NIFTY) % python -m pip install astro-sedpy corner spectres xarray
```

## Creating the Interpolation Files

Before you can run NIFTY, you need to download the models you are interested in fitting
to the photometry, and then you have to run a provided interpolation script which
will create a pickle file that is used by NIFTY in the fitting. 

```
(NIFTY) % python create_LOWZ_interpolator.py /Path/to/LOWZ/
(NIFTY) % python create_SonoraElfOwl_interpolator.py /Path/to/Sonora_Elf_Owl/
(NIFTY) % python create_ATMO2020_interpolator.py /Path/to/meisner_2023/
```
This will produce `LOWZ_interp.pkl`, `Sonora_interp.pkl`, and `ATMO2020_interp.pkl`, which
should stay in the same directory as where NIFTY is run. You'll only need to create these
once. Some of these models are fairly small, but the Sonora Elf Owl models are many, many GB,
and it will take a few hours to create the interpolator. 

## Running NIFTY

If you want to run NIFTY on one source:

```
(NIFTY) % python -W ignore NIFTY.py -filters BD_NIRCam_filters.txt -photometry /Path/to/JADES/photometry_file.fits -survey_stub JADES-GS -aperture CIRC1 -model LOWZ -id 20541
```
Here, you can see that `-filters` points to a file that has the filters you want to fit,
`-photometry` currently needs to point to the JADES photometry file with the fluxes
you'll want to fit, in columns labeled with the filter names from the filter file,
`-survey_stub` is mostly for output files, and just helps with identifying the survey
being explored. The argument `-aperture` should specify the size of the photometric 
aperture that the user wants from the photometry file. The argument `-model` can 
be `SonoraElfOwl`, `LOWZ`, or `ATMO2020`. 

You can also run on multiple sources in an ascii file, with the first column being the IDs:

```
(NIFTY) % python -W ignore NIFTY.py -filters BD_NIRCam_filters.txt -photometry /Path/to/JADES/photometry_file.fits -survey_stub JADES-GS -aperture CIRC1 -model LOWZ -idlist all_source_IDs.dat
```

or you can provide a list of IDs as an argument: 
```
(NIFTY) % python -W ignore NIFTY.py -filters BD_NIRCam_filters.txt -photometry /Path/to/JADES/photometry_file.fits -survey_stub JADES-GS -aperture CIRC1 -model LOWZ -idarglist '20541, 452029, 430165'
```

