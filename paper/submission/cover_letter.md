# Cover letter — SoftwareX

Dear Editors,

Please find enclosed our submission to SoftwareX, "fwhFoam: A Ffowcs
Williams–Hawkings solver with on-surface acoustic source localization for
OpenFOAM."

fwhFoam is an open-source aeroacoustic post-processing library for the
OpenFOAM CFD toolbox. It predicts far-field aerodynamic sound with
Farassat's formulation 1A of the Ffowcs Williams–Hawkings equation, for
permeable and impermeable integration surfaces, and — from the same surface
data — computes the surface sound-power density σ of the Delfs–Ruck
diffraction-filter theory, which localizes the acoustic sources on the
integration surface. The power identity ∮σ dS = P_far-field relates the two
and serves as an internal consistency check.

We make no claim of novelty for the FW-H formulation itself, which is
standard and available in existing OpenFOAM libraries such as libAcoustics;
we reference that work explicitly. The contribution of the software is (i) a
compact, documented and independently verified FW-H implementation released
openly with a reproducible verification and convergence suite, and (ii) the
on-surface source-localization diagnostic, which to our knowledge is not
available in existing OpenFOAM FW-H tools.

The software, documentation, tests and tutorials are publicly available
under GPL-3.0 at:

    https://github.com/Sparsh-Sharma/fwhFoam

The repository contains a documented README and a LICENSE file, and the
manuscript follows the SoftwareX Original Software Publication template. The
far-field solver is verified against analytic monopole, dipole and convected-
monopole solutions to ~0.1% with second-order spatial and temporal
convergence; the σ localization is verified through its power identity to a
few tenths of a percent on analytic sources.

This work has not been published previously and is not under consideration
elsewhere.

Sincerely,

Sparsh Sharma
German Aerospace Center (DLR)
sparsh.sharma@dlr.de
