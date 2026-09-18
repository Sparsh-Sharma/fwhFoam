#!/usr/bin/env python3
"""Generate a permeable cylindrical control-surface STL for this tutorial.

Writes constant/triSurface/fwhCylinder.stl -- an open cylindrical shell
(no end caps) of radius 8 D spanning the 2D domain thickness, suitable as
a permeable FW-H surface enclosing the cylinder and its near wake.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import geometry

os.makedirs("constant/triSurface", exist_ok=True)
v, f = geometry.cylinder_shell(radius=8.0, zmin=-0.5, zmax=0.5,
                               n_theta=200, n_z=1)
out = "constant/triSurface/fwhCylinder.stl"
geometry.write_stl(out, v, f, name="fwhCylinder")
print(f"wrote {out} ({len(f)} triangles, radius 8)")
