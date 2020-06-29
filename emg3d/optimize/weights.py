"""

:mod:`weights` -- Weights
=========================

Weights for optimization routines.

"""
# Copyright 2018-2020 The emg3d Developers.
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


import numpy as np
from scipy import signal, interpolate

__all__ = ['DataWeighting', 'DepthWeighting']


class DataWeighting:
    r"""Data Weighting; Plessix and Mulder, equation 18.

    .. math::

        W(x_s, x_r, \omega) =
        \frac{||x_s-x_r||^{\gamma_d}}
        {\omega^{\beta_f}||E_1^0(x_s, x_r, \omega)||^{\beta_d}}


    .. todo::

        - Currently, low amplitudes are switched-off, because it is divided by
          data. Test if small offsets should also be switched off.
        - Include other data weighting functions.


    Parameters
    ----------
    gamma_d : float
        Offset weighting exponent.

    beta_d : float
        Data weighting exponent.

    beta_f : float
        Frequency weighting exponent.

    """

    def __init__(self, gamma_d=0.5, beta_d=1.0, beta_f=0.25):
        """Initialize new DataWeighting instance."""

        # Store values
        self.gamma_d = gamma_d
        self.beta_d = beta_d
        self.beta_f = beta_f

    def weights(self, data, rec, src, freq, min_amp=1e-25):
        """[PlMu08]_, equation 18.


        Parameters
        ----------
        data : ndarray
            CSEM data.

        rec, src : tupples
            Receiver and source coordinates.

        freq : float
            Frequency (Hz)

        min_amp : float
            Data with amplitudes below min_amp get zero data weighting (data
            weighting divides by the amplitude, which is a problem if the
            amplitude is zero).


        Returns
        -------
        data_weight : ndarray
            Data weights (size of number of receivers).

        """

        # Mute
        mute = np.abs(data) < 1e-15

        # Get offsets.
        locs = np.stack([rec[0]-src[0], rec[1]-src[1], rec[2]-src[2]])
        offsets = np.linalg.norm(locs, axis=0)

        # Compute the data weight.
        data_weight = offsets**self.gamma_d / (2*np.pi*freq)**self.beta_f
        data_weight /= np.sqrt(np.real(data*data.conj()))**self.beta_d

        # Set small amplitudes to zero.
        data_weight[mute] = 0

        return data_weight


class DepthWeighting:
    r"""Depth Weighting; Plessix and Mulder, equation 17.

    Very crude, needs improvement.

    .. math::

        D = \left[ z^{-\gamma_m} \exp\left(-\beta_m \frac{z-z_b}{\delta}
            \right)\right] \qquad (text{for} \quad z > z_b)

    Adds a Tukey-window style taper at the bottom.

    .. todo::

        - Include other depth weighting functions.
        - "Make it 3D":

            - `depths` should depend on x and y. At the moment, they don't.
            - `z_ref` should depend on x and y. At the moment, it doesn't.


    Parameters
    ----------
    grid : :class:`emg3d.utils.TensorMesh`-instance
        The computational grid of the resistivity model.

    z_ref : float
        Reference depth; normally sea-bottom in the marine case or surface
        in the land case.

    res_ref : float
        Reference resistivity for skin depth.

    gamma_m : float
        Linear model weighting exponent.

    beta_m : float
        Exponential model weighting exponent.

    epsilon_r : float
        Regularization term.

    taper : list of two floats
        Tukey-taper: [n_skind-start, n_skind-end]
        Default is [2, 3]

    """

    def __init__(self, grid, z_ref, res_ref, gamma_m=0.0, beta_m=1.0,
                 epsilon_r=1.0, taper=None):
        """Initialize new DepthWeighting instance."""

        # Store values
        self.z_ref = z_ref
        self.res_ref = res_ref
        self.gamma_m = gamma_m
        self.beta_m = beta_m
        self.epsilon_r = epsilon_r
        self.vnC = grid.vnC
        self.depths = grid.vectorCCz

        # Default: Taper off from 2*skind to 3*skind
        if taper is None:
            self.taper = [2, 3]
        else:
            self.taper = taper

    def weights(self, freq):
        """[PlMu08]_, equation 17.

        Depth weighting, used as a preconditioner, with a Tukey-style taper at
        the bottom; frequency-dependent.

        Parameters
        ----------
        freq : float
            Frequency (Hz)

        Returns
        -------
        depth_weight : ndarray
            Depth weights (size of number of receivers).

        """

        # Take skin-depth as reference depth.
        skind = 503.3*np.sqrt(self.res_ref/freq)

        # Depth weighting, Plessix and Mulder, equation 17.
        # There is a typo in the equation.
        z = -(self.depths-self.z_ref)/skind

        # Depth dependency.
        precond = 1/(z**-self.gamma_m * np.exp(-self.beta_m*z) +
                     self.epsilon_r)

        # Reset to zero everything above seafloor.
        precond[self.depths >= self.z_ref] = 0

        # Use a Tukey-window to fade-out in depth.
        # Let [ne, ns] = taper, then the Tukey window goes
        # from z_ref-ne*sd to z_ref-ns*sd.
        # - alpha is defined by taper:
        alpha = np.diff(self.taper)/self.taper[1]
        # - create a 101 pt filter:
        win = signal.tukey(101, alpha=alpha)
        # - we are only interested in the half below z_ref:
        win[50:] = 1
        # - and map it to the actual data:
        end = self.z_ref - self.taper[1]*skind
        x = end + np.arange(101)*self.taper[1]*skind/50
        f = interpolate.interp1d(x, win, kind='cubic',
                                 bounds_error=False, fill_value=(0.0, 1.0))
        # - apply it:
        precond *= f(self.depths)

        # Normalize and return.
        precond /= precond.max()
        return np.repeat(precond, self.vnC[0]*self.vnC[1])
