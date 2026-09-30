"""pyfwh: Python companion tools for fwhFoam.

Modules
-------
io        : read/write the FWH-DATA surface format, read observer signals
analytic  : analytic acoustic fields (monopole, dipole, convected monopole)
geometry  : integration-surface generators (icosphere)
spectra   : PSD / SPL / OASPL helpers
localization : sigma surface sound-power density (source localization)
filter    : diffraction filter (acoustic surface pressure)
gpu       : CuPy-accelerated localization kernels (NumPy fallback)
"""

from . import io, analytic, geometry, spectra, localization, filter, gpu

__version__ = "1.1.0"
