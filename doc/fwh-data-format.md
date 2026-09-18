# FWH-DATA surface-data format (v1)

`fwhFoam` stores raw integration-surface data in a simple, self-describing
binary format so that observer signals can be recomputed offline
(`fwhSolve`) or by third-party tools (`pyfwh`). A parallel OpenFOAM run
writes one file per processor, `surfaceData_proc<N>.fwh`, each holding
that processor's local faces.

## Layout

An ASCII header terminated by a line `END_HEADER`, followed by a binary
payload of little-endian IEEE-754 `float64` values:

```
FWH-DATA 1\n
nFaces <N>\n
hasRho <0|1>\n
binary double\n
END_HEADER\n
<geometry block>
<record 0><record 1>...
```

- Line 1 `FWH-DATA 1` — magic string and format version.
- `nFaces <N>` — number of surface faces `N` in this file.
- `hasRho <0|1>` — whether each record carries a density field.
- `binary double` — payload encoding (only `double`/float64 in v1).

### Geometry block

`N` records of **7** doubles each, in face order:

```
cx cy cz  nx ny nz  area
```

- `(cx,cy,cz)` — face-centre coordinates [m].
- `(nx,ny,nz)` — **unit** normal, pointing away from the enclosed sources
  (towards the observers).
- `area` — face area [m²].

### Time records

Each record is:

```
t                      1 double     — source time [s]
p'[0..N-1]             N doubles    — gauge pressure p−p_ref [Pa]
u[0..N-1]              3N doubles   — fluid velocity (x,y,z) [m/s], CFD frame
rho[0..N-1]            N doubles    — density [kg/m^3]   (only if hasRho=1)
```

Records are written at a **constant** time interval. The number of records
is implicit (read until EOF); a truncated trailing record is ignored.

## Conventions

- `p'` is the gauge pressure in **Pa**. For incompressible OpenFOAM
  (kinematic pressure in m²/s²) the `fwh` function object multiplies by
  `rho0` before writing, so files are always in Pa.
- `u` is the **total** fluid velocity in the CFD (wind-tunnel) frame,
  including any mean flow.
- Normals point into the fluid / towards the observer.

## Reading in Python

```python
from pyfwh import io
data = io.read("surfaceData_proc0.fwh")   # -> FWHData
data.centres      # (N,3)
data.normals      # (N,3)
data.areas        # (N,)
data.times        # (T,)
data.p            # (T,N)
data.u            # (T,N,3)
data.rho          # (T,N) or None
```

## Endianness / portability

The payload is little-endian `float64`. All current OpenFOAM target
platforms (x86-64, ARM64) are little-endian, so no byte-swapping is
performed. A future format version would add an explicit endianness tag if
big-endian support is required.
