# pyfwh

Python companion tools for [fwhFoam](../README.md).

## Install

```bash
pip install -e tools           # from the repository root
# or, with plotting and test extras:
pip install -e "tools[plot,test]"
```

## Modules

| Module | Contents |
|--------|----------|
| `pyfwh.io` | read/write the FWH-DATA surface format (`read`, `write`, `FWHData`); read `observer_*.dat` signals (`read_observer`) |
| `pyfwh.analytic` | exact acoustic fields: `MonopoleField`, `DipoleField`, `ConvectedMonopoleField` |
| `pyfwh.geometry` | integration-surface generators: `icosphere`, `write_stl` |
| `pyfwh.spectra` | `psd`, `spl`, `oaspl`, `amplitude_at` |

## Example

```python
import numpy as np
from pyfwh import analytic, geometry, io

# build a permeable surface and sample an analytic monopole on it
cen, nrm, area, verts, faces = geometry.icosphere(radius=1.0, subdivisions=3)
field = analytic.MonopoleField(amplitude=1e-3, omega=2*np.pi*340.29)

times = np.arange(0, 6/340.29, (1/340.29)/64)
p = np.array([np.real(field.pressure(cen, t)) for t in times])
u = np.array([np.real(field.velocity(cen, t)) for t in times])
rho = field.rho0 + np.array([np.real(field.density(cen, t)) for t in times])

io.FWHData(cen, nrm, area, times, p, u, rho).write("monopole.fwh")
# -> run: fwhSolve -dict fwhSolveDict
```

See [`../tests/analytic`](../tests/analytic) for the full verification
workflow.
