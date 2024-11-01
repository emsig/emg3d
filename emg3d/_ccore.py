import os
import ctypes as ct

import numpy as np


C_DOUBLEP = ct.POINTER(ct.c_double)
path = os.path.dirname(os.path.realpath(__file__))
csolve = np.ctypeslib.load_library("csolve", path)
cgaussseidel = np.ctypeslib.load_library("cgaussseidel", path)


def solve(amat, bvec):
    """C-Wrapper for wavenumber."""
    n = bvec.size
    csolve.solve(
        int(n),
        amat.ctypes.data_as(C_DOUBLEP),
        bvec.ctypes.data_as(C_DOUBLEP),
    )


def gaussseidel(
        ex, ey, ez, sx, sy, sz, eta_x, eta_y, eta_z, zeta, hx, hy, hz, nu
    ):
    """C-Wrapper for gauss_seidel."""
    cgaussseidel.gaussseidel(
        ex.ctypes.data_as(c_doublep),
        ey.ctypes.data_as(c_doublep),
        ez.ctypes.data_as(c_doublep),
        sx.ctypes.data_as(c_doublep),
        sy.ctypes.data_as(c_doublep),
        sz.ctypes.data_as(c_doublep),
        eta_x.ctypes.data_as(c_doublep),
        eta_y.ctypes.data_as(c_doublep),
        eta_z.ctypes.data_as(c_doublep),
        zeta.ctypes.data_as(c_doublep),
        hx.ctypes.data_as(c_doublep),
        hy.ctypes.data_as(c_doublep),
        hx.ctypes.data_as(c_doublep),
        int(nu),
    )
