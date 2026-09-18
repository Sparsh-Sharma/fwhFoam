"""pyfwh: Python companion tools for fwhFoam.

Modules
-------
io        : read/write the FWH-DATA surface format, read observer signals
analytic  : analytic acoustic fields (monopole, dipole, convected monopole)
geometry  : integration-surface generators (icosphere)
spectra   : PSD / SPL / OASPL helpers
"""

from . import io, analytic, geometry, spectra, localization

__version__ = "1.0.0"
