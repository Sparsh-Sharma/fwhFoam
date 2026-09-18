"""Read and write the FWH-DATA surface-data format used by fwhFoam.

The format is a short ASCII header terminated by a line ``END_HEADER``
followed by little-endian IEEE-754 float64 binary payload:

    FWH-DATA 1\\n
    nFaces <N>\\n
    hasRho <0|1>\\n
    binary double\\n
    END_HEADER\\n
    <geometry: N x (cx cy cz  nx ny nz  area)>
    <records: t, p'[N], u[3N], (rho[N] if hasRho)>

- Face normals point away from the enclosed sources (towards observers).
- p' is the gauge pressure in Pa (p - p_ref).
- u is the fluid velocity in the CFD (wind-tunnel) frame [m/s].
"""

from __future__ import annotations

import numpy as np

_MAGIC = b"FWH-DATA"


class FWHData:
    """In-memory representation of an FWH-DATA surface dataset."""

    def __init__(self, centres, normals, areas, times, p, u, rho=None):
        self.centres = np.asarray(centres, dtype=np.float64)   # (N,3)
        self.normals = np.asarray(normals, dtype=np.float64)   # (N,3)
        self.areas = np.asarray(areas, dtype=np.float64)       # (N,)
        self.times = np.asarray(times, dtype=np.float64)       # (T,)
        self.p = np.asarray(p, dtype=np.float64)               # (T,N)
        self.u = np.asarray(u, dtype=np.float64)               # (T,N,3)
        self.rho = None if rho is None else np.asarray(rho, np.float64)

    @property
    def n_faces(self):
        return self.centres.shape[0]

    @property
    def n_times(self):
        return self.times.shape[0]

    def write(self, path):
        write(path, self)


def write(path, data: "FWHData"):
    """Write an :class:`FWHData` object to *path* in FWH-DATA format."""
    n = data.n_faces
    has_rho = data.rho is not None
    with open(path, "wb") as f:
        header = (
            f"FWH-DATA 1\n"
            f"nFaces {n}\n"
            f"hasRho {1 if has_rho else 0}\n"
            f"binary double\n"
            f"END_HEADER\n"
        )
        f.write(header.encode("ascii"))

        geom = np.empty((n, 7), dtype="<f8")
        geom[:, 0:3] = data.centres
        geom[:, 3:6] = data.normals
        geom[:, 6] = data.areas
        f.write(geom.tobytes())

        for k, t in enumerate(data.times):
            f.write(np.float64(t).astype("<f8").tobytes())
            f.write(data.p[k].astype("<f8").tobytes())
            f.write(data.u[k].astype("<f8").reshape(-1).tobytes())
            if has_rho:
                f.write(data.rho[k].astype("<f8").tobytes())


def read(path) -> "FWHData":
    """Read an FWH-DATA file into an :class:`FWHData` object."""
    with open(path, "rb") as f:
        n_faces = None
        has_rho = True
        # Header (line-based ASCII)
        magic_ok = False
        while True:
            line = _readline(f)
            if line is None:
                raise ValueError("Unexpected EOF in FWH-DATA header")
            line = line.strip()
            if line == b"END_HEADER":
                break
            if line.startswith(_MAGIC):
                magic_ok = True
            elif line.startswith(b"nFaces"):
                n_faces = int(line.split()[1])
            elif line.startswith(b"hasRho"):
                has_rho = int(line.split()[1]) != 0
        if not magic_ok or n_faces is None:
            raise ValueError(f"{path} is not a valid FWH-DATA file")

        geom = np.frombuffer(f.read(n_faces * 7 * 8), dtype="<f8")
        geom = geom.reshape(n_faces, 7)
        centres = geom[:, 0:3].copy()
        normals = geom[:, 3:6].copy()
        areas = geom[:, 6].copy()

        rec_doubles = 1 + n_faces + 3 * n_faces + (n_faces if has_rho else 0)
        rec_bytes = rec_doubles * 8

        times, ps, us, rhos = [], [], [], []
        while True:
            chunk = f.read(rec_bytes)
            if len(chunk) < rec_bytes:
                break
            vals = np.frombuffer(chunk, dtype="<f8")
            off = 0
            times.append(vals[off]); off += 1
            ps.append(vals[off:off + n_faces].copy()); off += n_faces
            us.append(vals[off:off + 3 * n_faces].reshape(n_faces, 3).copy())
            off += 3 * n_faces
            if has_rho:
                rhos.append(vals[off:off + n_faces].copy()); off += n_faces

    return FWHData(
        centres, normals, areas,
        np.array(times),
        np.array(ps),
        np.array(us),
        np.array(rhos) if has_rho else None,
    )


def _readline(f):
    """Read one \\n-terminated line from a binary stream, or None at EOF."""
    buf = bytearray()
    while True:
        b = f.read(1)
        if not b:
            return None if not buf else bytes(buf)
        if b == b"\n":
            return bytes(buf)
        buf += b


def read_observer(path):
    """Read a fwhFoam observer_*.dat signal.

    Returns a dict with keys t, p, pThickness, pLoading (numpy arrays) and
    metadata parsed from the comment header (position, c0, rho0, U0,
    validWindow).
    """
    meta = {}
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s[1:].strip()
                if body.startswith("position:"):
                    meta["position"] = _parse_vec(body.split(":", 1)[1])
                elif body.startswith("validWindow:"):
                    parts = body.split(":", 1)[1].split()
                    meta["validWindow"] = (float(parts[0]), float(parts[1]))
                elif body.startswith("c0:"):
                    meta["header"] = body
                continue
            rows.append([float(x) for x in s.split()])
    a = np.array(rows) if rows else np.zeros((0, 4))
    return {
        "t": a[:, 0],
        "p": a[:, 1],
        "pThickness": a[:, 2] if a.shape[1] > 2 else None,
        "pLoading": a[:, 3] if a.shape[1] > 3 else None,
        "meta": meta,
    }


def _parse_vec(s):
    s = s.replace("(", " ").replace(")", " ")
    return np.array([float(x) for x in s.split()])
