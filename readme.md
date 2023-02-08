
# mSSFP (Multi-SSFP Reconstruction Library)

mSSFP is library for image reconstuction for SSFP. This library supports ssfp simulations, various phantom generators, ssfp recontructions.

Multiple SSFP images with different phase cycle amounts can be combined to suppress banding artifacts and for the estimation of quantitative biomarker like T1/T2 relaxation parameter mappings. Multiple methods for band suppression have been developed over the years, but each method has limitations.

## Features

### Simultations
  - ssfp

### phantoms
  - Shepp-Logan Phantom
  - Brain Phantom
  - Simple Block Phantoms
### Banding Arfact Removl Recons
  - Sum of squares 
  - Eliptical singal model 
  - Super field of view (superFOV)
### Quantitative MR Recons
  - PLANET for T2/T1 mapping

## Notebooks

Jupyter notebooks for examples of how to use the mSSFP library.

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
