# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased - 0.1.0]

### Added
- Updates to phantom generators
  - pdated slice selection in load_dataslice
  - Added print_dataset_info, get_dataset_info
  - Added noise option off-resonance field map simulation 

## [0.0.1] - 2024-11-05 - Initial development

### Added
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
