# Changelog

All notable changes to this project will be documented in this file.

## [0.0.12] - 2025-05-29

- Added support for df_window parameter and verbose option in generate_ssfp_dataset
- Added support perlin noise option in generate_ssfp_dataset
- Removed warnings from ssfp simulation for division by zero

## [0.0.7] - 2025-05-21

- Updated plot_dataset with args for num_rows, figsize and dpi

## [0.0.5] - 2025-03-19

- Added support for array of alphas for ssfp simulation
- Added support to generate phantoms from a list of tissue parameters

## [0.0.4] - 2024-11-15

- Updates to phantom generators
  - Updated slice selection in load_dataslice
  - Added print_dataset_info, get_dataset_info
  - Added noise option off-resonance field map simulation 
  - Cleaned up and refactored phantom code
  - phantom.generate_ssfp_dataset() can be used to generate a 3d ssfp dataset from a various phantom types

## [0.0.1] - 2024-11-05 - Initial development

- Initial release with core SSFP simulation and reconstruction capabilities
- Phantom generators including Shepp-Logan, block phantoms, and brain phantoms
- Multiple banding artifact removal reconstructions:
  - Sum of squares
  - Elliptical signal model
  - Super field of view (superFOV)
- PLANET implementation for T2/T1 mapping
- Jupyter notebook examples demonstrating various features
- Basic SSFP simulations
- Support for Python 3.8+
