<img src="nifty_logo_zoom.png" alt="NIFTY logo" width="300"/>

# NIFTY
*Near-Infrared Fitting for T and Y Dwarfs*

*Kevin Hainline and Jake Helton*

NIFTY is a code designed to fit JWST NIRCam and MIRI photometry for cold brown 
dwarfs with the LOWZ (Meisner et al. 2021), ATMO2020 (Philips et al. 2020), or Sonora 
Elf Owl v2 (Mukherjee et al. 2024, Wogan et al. 2025) v2 models, in a Bayesian framework, 
using the emcee sampler (https://emcee.readthedocs.io/en/stable/user/sampler/). 

The code creates interpolation grids in log space from the models themselves, and then 
runs a fit directly on the input photometry, producing a corner plot, a fit SED, and 
a file with the 16th, 50th, and 84th percentiles on the fit parameters. In our tests,
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
This will produce `LOWZ_interp.pkl`, `Sonora_v2_interp.pkl`, and `ATMO2020_interp.pkl`, which
should stay in the same directory as where NIFTY is run. You'll only need to create these
once. Some of these models are fairly small, but the Sonora Elf Owl models are many, many GB,
and it will take a few hours to create the interpolator. 

## Running NIFTY

If you want to run NIFTY on one source:

```
(NIFTY) % python -W ignore NIFTY.py -config_file BD_NIRCam_filters.json -photometry /Path/to/JADES/photometry_file.fits -survey_stub JADES-GS -model LOWZ -id 20541
```
Here, you can see that `-config_file` points to a json file that has the filters you want
to fit (see below),`-photometry` currently needs to point to the photometry file (.fits or
.txt, also, see below) with the fluxes you'll want to fit,
`-survey_stub` is mostly for output files, and just helps with identifying the survey
being explored. The argument `-model` can be `SonoraElfOwl`, `LOWZ`, or `ATMO2020`. 

You can also run on multiple sources in an ascii file, with the first column being the IDs:

```
(NIFTY) % python -W ignore NIFTY.py -config_file BD_NIRCam_filters.json -photometry /Path/to/JADES/photometry_file.fits -survey_stub JADES-GS -model LOWZ -idlist all_source_IDs.dat
```

or you can provide a list of IDs as an argument: 
```
(NIFTY) % python -W ignore NIFTY.py -config_file BD_NIRCam_filters.json -photometry /Path/to/JADES/photometry_file.fits -survey_stub JADES-GS -model LOWZ -idarglist '20541, 452029, 430165'
```

## Configuration File (--config_file)

Instead of requiring a separate filter file and hardcoded assumptions about aperture 
suffixes or FITS extensions, NIFTY uses a single JSON file to describe all 
relevant filter info.

Here’s a sample entry from a valid config file:

```
{
  "filter_columns": {
    "F115W": {
      "extension": "CIRC",
      "flux": "F115W_CIRC3",
      "error": "F115W_CIRC3_e",
      "wavelength": 1.154
    },
    "F444W": {
      "extension": "MIRI",
      "flux": "F444W_CIRC3",
      "error": "F444W_CIRC3_e",
      "wavelength": 4.408
    }
  }
}
```

Here:
`extension` is the FITS HDU where that filter’s data lives (can be the same for all 
filters or vary). If you are supplying a text file, then this line won't matter. 
`flux` and `error` are the exact column names in the file.
`wavelength` is the central wavelength of the filter, in microns, used for plotting.

We include an example .json file (`BD_NIRCam_MIRI_filters.json` for use with 
JADES observations. 

## Photometry File Format (--photometry)

NIFTY can read:

- .fits files (with filter fluxes and errors in specified extensions, and column names)
- Plain text catalogs, whitespace-delimited, with one line per object. 

You will need to include ID (capitalized), but you should specify the filter and error
column names  in the .json file, here's a simple example:

```
ID     F115W_CIRC3     F115W_CIRC3_e     F444W_CIRC3     F444W_CIRC3_e
101    0.123           0.010             0.456           0.025
```

The header line for this file must match the JSON config exactly. Do not prefix this
first line with a `#`.

If the file has an unsupported format, or if the expected columns are missing, 
NIFTY will give a clear error with suggestions.