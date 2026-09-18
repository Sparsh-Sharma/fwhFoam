"""Integration-surface generators for fwhFoam verification.

Provides a geodesic icosphere (subdivided icosahedron) as a closed
triangulated surface, returned as face centres, outward unit normals and
areas -- exactly the geometry fields fwhFoam consumes.
"""

from __future__ import annotations

import numpy as np


def icosphere(radius=1.0, subdivisions=3, centre=(0, 0, 0)):
    """Return (centres, normals, areas, vertices, faces) for an icosphere.

    Parameters
    ----------
    radius : float
    subdivisions : int
        Number of edge-bisection refinement levels (0 = base icosahedron,
        20 faces; each level multiplies the face count by 4).
    centre : (3,) array

    Returns
    -------
    centres : (F,3) face centroids
    normals : (F,3) outward unit normals
    areas   : (F,) triangle areas
    vertices: (V,3)
    faces   : (F,3) int vertex indices
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1)[:, None]

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)

    verts = verts / np.linalg.norm(verts, axis=1)[:, None]

    centre = np.asarray(centre, dtype=np.float64)
    verts_scaled = verts * radius + centre

    v0 = verts_scaled[faces[:, 0]]
    v1 = verts_scaled[faces[:, 1]]
    v2 = verts_scaled[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    normals = (centroids - centre)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    # Ensure triangle winding gives outward normal (flip face if needed)
    tri_n = cross / np.linalg.norm(cross, axis=1)[:, None]
    flip = np.sum(tri_n * normals, axis=1) < 0
    faces[flip] = faces[flip][:, ::-1]

    return centroids, normals, areas, verts_scaled, faces


def _subdivide(verts, faces):
    verts = list(map(tuple, verts))
    index = {v: i for i, v in enumerate(verts)}
    midcache = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in midcache:
            return midcache[key]
        va = np.asarray(verts[a]); vb = np.asarray(verts[b])
        vm = (va + vb) / 2.0
        vm = vm / np.linalg.norm(vm)
        tm = tuple(vm)
        idx = index.get(tm)
        if idx is None:
            idx = len(verts)
            verts.append(tm)
            index[tm] = idx
        midcache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

    return np.asarray(verts, dtype=np.float64), np.asarray(new_faces, np.int64)


def cylinder_shell(radius=8.0, zmin=-0.5, zmax=0.5, n_theta=120, n_z=1,
                   centre=(0, 0, 0)):
    """Triangulated open cylindrical shell (no end caps).

    Suitable as a permeable FW-H surface for a 2D (span-periodic /
    empty-direction) cylinder case. Returns (vertices, faces) for STL
    output; outward normals are guaranteed by triangle winding.
    """
    cx, cy, cz = centre
    zs = np.linspace(zmin, zmax, n_z + 1) + cz
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    verts = []
    for z in zs:
        for a in th:
            verts.append((cx + radius * np.cos(a), cy + radius * np.sin(a), z))
    verts = np.asarray(verts)

    faces = []
    for k in range(n_z):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a = k * n_theta + j
            b = k * n_theta + j2
            c = (k + 1) * n_theta + j
            d = (k + 1) * n_theta + j2
            # outward-wound triangles (CCW seen from outside)
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int64)


def write_stl(path, vertices, faces, name="fwhSurface"):
    """Write an ASCII STL of the triangulated surface."""
    v = np.asarray(vertices); f = np.asarray(faces)
    with open(path, "w") as fh:
        fh.write(f"solid {name}\n")
        for tri in f:
            p0, p1, p2 = v[tri[0]], v[tri[1]], v[tri[2]]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else n
            fh.write(f"  facet normal {n[0]:.7e} {n[1]:.7e} {n[2]:.7e}\n")
            fh.write("    outer loop\n")
            for p in (p0, p1, p2):
                fh.write(f"      vertex {p[0]:.7e} {p[1]:.7e} {p[2]:.7e}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write(f"endsolid {name}\n")
