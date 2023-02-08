
# mSSFP (Multi-SSFP Reconstruction Library)

mSSFP is library for image reconstuction for multi-acqusition SSFP. This library supports ssfp simulations, various phantom generators, and various ssfp recontructions using muliple phase-cycled ssfp images. 

Steady-Stead Free Precession (SSFP) MRI is class of fast pulse sequence capable of generating high SNR images. However, SSFP is highly-sensitive to off-resonance effects, which cause banding artifacts. Multiple SSFP images with different phase cycle amounts can be combined to suppress banding artifacts and for the estimation of quantitative biomarker like T1/T2 relaxation parameter mappings. Multiple methods for band suppression have been developed over the years, and this library gives working code and notebook examples for a variety of these reconstrcution techniques.

## Features

### Simultations
  - ssfp

### Phantoms
  - Shepp-Logan phantom
  - Simple block phantoms
  - Brain phantom
### Banding Artifact Removal Recons
  - Sum of squares 
  - Eliptical singal model 
  - Super field of view (superFOV)
### Quantitative MR Recons
  - PLANET for T2/T1 mapping

## Notebooks

Jupyter notebooks for examples of how to use the mSSFP library.

- Basic SSFP Simulations ([notebook](notebooks/1_sspf_simulations.ipynb))
- Phantom Examples ([notebook](notebooks/2_phantoms.ipynb))
- SSFP Banding Artifact Removal ([notebook](notebooks/3_ssfp_band_removal.ipynb))
- PLANET for T2/T1 Mapping of SSFP ([notebook](notebooks/4_ssfp_brain_planet.ipynb))
- SuperFOV for accelerated SSFP ([simple notebook](notebooks/5_superFOV.ipynb), [detailed notebook](notebooks/5a_superFOV_detailed.ipynb))

## Development

This project requires python 3.8+ and has the dependancies in requirement.txt

To setup a python enviroment with conda:

> ```
> conda create -n mssfp python=3.8 
> conda activate mssfp
> ```
> Then install packages with pip using requirements file 
> ```
> pip install -r requirements.txt
> ```
