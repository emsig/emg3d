"""

Helpers
=======

These are some helper functions for the test suite.
"""
import numpy as np


def get_h(ncore, npad, width, factor):
    """Get cell widths for TensorMesh."""
    pad = ((np.ones(npad)*np.abs(factor))**(np.arange(npad)+1))*width
    return np.r_[pad[::-1], np.ones(ncore)*width, pad]


def dummy_field(nx, ny, nz, imag=True):
    """Return complex dummy arrays of shape nx*ny*nz.

    Numbers are from 1..nx*ny*nz for the real part, and 1/100 of it for the
    imaginary part.

    """
    if imag:
        out = np.arange(1., nx*ny*nz+1) + 1j*np.arange(1., nx*ny*nz+1)/100.
    else:
        out = np.arange(1., nx*ny*nz+1)

    return out.reshape(nx, ny, nz)
