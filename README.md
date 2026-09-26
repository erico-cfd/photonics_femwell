## Double-Ring Photon-Pair Generator Simulation

This repository contains the numerical models used to design and analyze a silicon photonics double-ring source based on spontaneous four-wave mixing (SFWM).

## Main steps

### Waveguide optimization

FEMWELL is used to simulate the silicon waveguide and evaluate:

- effective index
- group index
- group-velocity dispersion

The geometry is swept to choose a suitable waveguide for the pump wavelength.

### Double-ring model

The analytical model describes the two coupled microrings using the intracavity fields.

It calculates:

- ring 1 buildup
- ring 2 buildup
- drop-port transmission
- pump, signal and idler behavior

### Coupling optimization

A grid sweep is performed over the three coupling coefficients.

For each combination, the code evaluates:

- pump rejection
- signal/idler transmission
- buildup on each ring

The final design is selected according to the defined pump-rejection constraint.

### Physical coupler gaps

FEMWELL is then used to convert the coupling coefficients into physical gaps.

The even and odd supermodes of the directional coupler are calculated, and the relation between coupling and gap is obtained.

The final gaps are also rounded to the assumed fabrication grid (10nm resolution).

## Main outputs

The scripts generate:

- sweep optimization
- ring buildup spectra
- drop-port transmission spectrum
- optimized coupling coefficients
- physical coupler gaps

## Python files

- transfer_function.py — Calculates the waveguide modes and dispersion, sizes the two rings for the Vernier configuration, and simulates the initial through/drop transmission.

- gvd.py — Sweeps the silicon waveguide width and wavelength with FEMWELL to find a geometry with suitable group-velocity dispersion (GVD slightly above 0).

- physical_analytic_opt.py — Calculates the coupling coefficients from the analytical design conditions and evaluates buildup, pump rejection, extraction efficiency and coupling-gap sensitivity.

- sweep_opt.py — Sweeps the three coupling coefficients, selects the design satisfying the pump-rejection constraint, calculates the ring buildup and drop transmission, and converts the selected couplings into physical gaps.
