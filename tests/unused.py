"""

Unused
======

These are unused code snippets that might come in handy at some point, maybe
even just for testing purposes.

"""
# Copyright 2018-2021 The emg3d Developers.
#
# This file is part of emg3d.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.


import numba as nb
# import numpy as np

# Numba-settings
_numba_setting = {'nogil': True, 'fastmath': True, 'cache': True}


@nb.njit(**_numba_setting)
def cellaverages2edges(sx, sy, sz, eta_x, eta_y, eta_z):
    r"""Interpolate cell averaged properties to edges (fields)."""

    # Get dimensions
    nx, ny, nz = eta_x.shape

    for iz in range(nz):
        izm = max(0, iz-1)
        for iy in range(ny):
            iym = max(0, iy-1)
            for ix in range(nx):
                ixm = max(0, ix-1)

                stx = (eta_x[ix, iym, izm] + eta_x[ix, iym, iz] +
                       eta_x[ix, iy, izm] + eta_x[ix, iy, iz])
                sty = (eta_y[ixm, iy, izm] + eta_y[ix, iy, izm] +
                       eta_y[ixm, iy, iz] + eta_y[ix, iy, iz])
                stz = (eta_z[ixm, iym, iz] + eta_z[ix, iym, iz] +
                       eta_z[ixm, iy, iz] + eta_z[ix, iy, iz])

                sx[ix, iy, iz] *= stx/4
                sy[ix, iy, iz] *= sty/4
                sz[ix, iy, iz] *= stz/4
