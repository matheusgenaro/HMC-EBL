# HMC-EBL
A simple implementation of Hamiltonian Monte Carlo (HMC) to infer Extragalactic Background Light (EBL) parameters using flux observations of gamma ray sources.
The EBL model adopted is the one by Finke et al. (2010), considering the dust fractions of the PAH and Small Grains components as the free parameters.
The intrinsic spectra of the sources can be described by three parametrizations: power law (PL), log-parabola (LP) and power law with exponential cut-off (PLC).

A sample of 12 synthetic BL Lac spectra is supplied. 
In 'chains' folder, there are also the effective samples from the analysis of IACT data by Genaro et al. (2023, in prep).

## Required Python packages
numpy
scipy

## Usage
- The files provided are tunned for running HMC considering 5 synthetic BL Lacs.
- 'start_hmc.py' defines the intrinsic models for each source, the number of HMC steps, integration steps and size, as well as the inverse mass matrix ('invmass_matrix.dat') and initial positions files ('initialization.py').
- 'sources_list.txt' lists the data files (flux observations).
- If you want to renormalize the flux data (recommended for HMC computations), insert in 'normalization.dat' the factors that each spectrum will be multiplied to.
- For different data file formatting, modifications in the read_data() function in run_hmc.py may be required.
