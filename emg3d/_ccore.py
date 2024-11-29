import os
import ctypes as ct

import numpy as np


C_DOUBLEP = ct.POINTER(ct.c_double)
path = os.path.dirname(os.path.realpath(__file__))
csolve = np.ctypeslib.load_library("csolve", path)
cgaussseidel = np.ctypeslib.load_library("cgauss_seidel", path)


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
    # Get dimensions
    nx = len(hx)
    ny = len(hy)
    nz = len(hz)
    print("type ex=",ex.dtype, ex.shape)
    print("type ey=",ey.dtype, ey.shape)
    print("type ez=",ez.dtype, ez.shape)
    print("type sx=",sx.dtype, sx.shape)
    print("type sy=",sy.dtype, sy.shape)
    print("type sz=",sz.dtype, sz.shape)
    print("type eta_x=",eta_x.dtype, eta_x.shape)
    print("type eta_y=",eta_y.dtype, eta_y.shape)
    print("type eta_z=",eta_z.dtype, eta_z.shape)
    print("type zeta=",zeta.dtype, zeta.shape)
    print("type hx=",hx.dtype, hx.shape)
    print("type hx=",hy.dtype, hy.shape)
    print("type hz=",hz.dtype, hz.shape)
    cgaussseidel.gauss_seidel(
        ex.ctypes.data_as(C_DOUBLEP),
        ey.ctypes.data_as(C_DOUBLEP),
        ez.ctypes.data_as(C_DOUBLEP),
        sx.ctypes.data_as(C_DOUBLEP),
        sy.ctypes.data_as(C_DOUBLEP),
        sz.ctypes.data_as(C_DOUBLEP),
        eta_x.ctypes.data_as(C_DOUBLEP),
        eta_y.ctypes.data_as(C_DOUBLEP),
        eta_z.ctypes.data_as(C_DOUBLEP),
        zeta.ctypes.data_as(C_DOUBLEP),
        hx.ctypes.data_as(C_DOUBLEP),
        hy.ctypes.data_as(C_DOUBLEP),
        hz.ctypes.data_as(C_DOUBLEP),
        int(nx),
        int(ny),
        int(nz),
        int(nu),
    )
