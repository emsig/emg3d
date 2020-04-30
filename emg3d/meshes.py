"""

:mod:`meshes` -- Discretization
===============================

Everything related to meshes appropriate for the multigrid solver.

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
from copy import deepcopy
from scipy import optimize

__all__ = ['TensorMesh', 'get_hx_h0', 'get_cell_numbers', 'get_stretched_h',
           'get_domain', 'get_hx']


class TensorMesh:
    """Rudimentary mesh for multigrid calculation.

    The tensor-mesh :class:`discretize.TensorMesh` is a powerful tool,
    including sophisticated mesh-generation possibilities in 1D, 2D, and 3D,
    plotting routines, and much more. However, in the multigrid solver we have
    to generate a mesh at each level, many times over and over again, and we
    only need a very limited set of attributes. This tensor-mesh class provides
    all required attributes. All attributes here are the same as their
    counterparts in :class:`discretize.TensorMesh` (both in name and value).

    .. warning::
        This is a slimmed-down version of :class:`discretize.TensorMesh`, meant
        principally for internal use by the multigrid modeller. It is highly
        recommended to use :class:`discretize.TensorMesh` to create the input
        meshes instead of this class. There are no input-checks carried out
        here, and there is only one accepted input format for `h` and `x0`.


    Parameters
    ----------
    h : list of three ndarrays
        Cell widths in [x, y, z] directions.

    x0 : ndarray of dimension (3, )
        Origin (x, y, z).

    """

    def __init__(self, h, x0):
        """Initialize the mesh."""
        self.x0 = x0

        # Width of cells.
        self.hx = h[0]
        self.hy = h[1]
        self.hz = h[2]

        # Cell related properties.
        self.nCx = int(self.hx.size)
        self.nCy = int(self.hy.size)
        self.nCz = int(self.hz.size)
        self.vnC = np.array([self.hx.size, self.hy.size, self.hz.size])
        self.nC = int(self.vnC.prod())
        self.vectorCCx = np.r_[0, self.hx[:-1].cumsum()]+self.hx*0.5+self.x0[0]
        self.vectorCCy = np.r_[0, self.hy[:-1].cumsum()]+self.hy*0.5+self.x0[1]
        self.vectorCCz = np.r_[0, self.hz[:-1].cumsum()]+self.hz*0.5+self.x0[2]

        # Node related properties.
        self.nNx = self.nCx + 1
        self.nNy = self.nCy + 1
        self.nNz = self.nCz + 1
        self.vnN = np.array([self.nNx, self.nNy, self.nNz], dtype=int)
        self.nN = int(self.vnN.prod())
        self.vectorNx = np.r_[0., self.hx.cumsum()] + self.x0[0]
        self.vectorNy = np.r_[0., self.hy.cumsum()] + self.x0[1]
        self.vectorNz = np.r_[0., self.hz.cumsum()] + self.x0[2]

        # Edge related properties.
        self.vnEx = np.array([self.nCx, self.nNy, self.nNz], dtype=int)
        self.vnEy = np.array([self.nNx, self.nCy, self.nNz], dtype=int)
        self.vnEz = np.array([self.nNx, self.nNy, self.nCz], dtype=int)
        self.nEx = int(self.vnEx.prod())
        self.nEy = int(self.vnEy.prod())
        self.nEz = int(self.vnEz.prod())
        self.vnE = np.array([self.nEx, self.nEy, self.nEz], dtype=int)
        self.nE = int(self.vnE.sum())

    def __repr__(self):
        """Simple representation."""
        return (f"TensorMesh: {self.nCx} x {self.nCy} x {self.nCz} "
                f"({self.nC:,})")

    def copy(self):
        """Return a copy of the TensorMesh."""
        return TensorMesh.from_dict(self.to_dict(True))

    def to_dict(self, copy=False):
        """Store the necessary information of the TensorMesh in a dict."""
        out = {'hx': self.hx, 'hy': self.hy, 'hz': self.hz, 'x0': self.x0,
               '__class__': self.__class__.__name__}
        if copy:
            return deepcopy(out)
        else:
            return out

    @classmethod
    def from_dict(cls, inp):
        """Convert dictionary into :class:`TensorMesh` instance.

        Parameters
        ----------
        inp : dict
            Dictionary as obtained from :func:`TensorMesh.to_dict`.
            The dictionary needs the keys `hx`, `hy`, `hz`, and `x0`.

        Returns
        -------
        obj : :class:`TensorMesh` instance

        """
        try:
            return cls(h=[inp['hx'], inp['hy'], inp['hz']], x0=inp['x0'])
        except KeyError as e:
            print(f"* ERROR   :: Variable {e} missing in `inp`.")
            raise

    @property
    def vol(self):
        """Construct cell volumes of the 3D model as 1D array."""
        if getattr(self, '_vol', None) is None:
            self._vol = (self.hx[None, None, :]*self.hy[None, :, None] *
                         self.hz[:, None, None]).ravel()
        return self._vol


def get_hx_h0(freq, res, domain, fixed=0., possible_nx=None, min_width=None,
              pps=3, alpha=None, max_domain=100000., raise_error=True, verb=1,
              return_info=False):
    r"""Return cell widths and origin for given parameters.

    Returns cell widths for the provided frequency, resistivity, domain extent,
    and other parameters using a flexible amount of cells. See input parameters
    for more details. A maximum of three hard/fixed boundaries can be provided
    (one of which is the grid center).

    The minimum cell width is calculated through :math:`\delta/\rm{pps}`, where
    the skin depth is given by :math:`\delta = 503.3 \sqrt{\rho/f}`, and the
    parameter `pps` stands for 'points-per-skindepth'. The minimum cell width
    can be restricted with the parameter `min_width`.

    The actual calculation domain adds a buffer zone around the (survey)
    domain. The thickness of the buffer is six times the skin depth. The field
    is basically zero after two wavelengths. A wavelength is
    :math:`2\pi\delta`, hence roughly 6 times the skin depth. Taking a factor 6
    gives therefore almost two wavelengths, as the field travels to the
    boundary and back. The actual buffer thickness can be steered with the
    `res` parameter.

    One has to take into account that the air is very resistive, which has to
    be considered not just in the vertical direction, but also in the
    horizontal directions, as the airwave will bounce back from the sides
    otherwise. In the marine case this issue reduces with increasing water
    depth.


    See Also
    --------
    get_stretched_h : Get `hx` for a fixed number `nx` and within a fixed
                      domain.


    Parameters
    ----------

    freq : float
        Frequency (Hz) to calculate the skin depth. The skin depth is a concept
        defined in the frequency domain. If a negative frequency is provided,
        it is assumed that the calculation is carried out in the Laplace
        domain. To calculate the skin depth, the value of `freq` is then
        multiplied by :math:`-2\pi`, to simulate the closest
        frequency-equivalent.

    res : float or list
        Resistivity (Ohm m) to calculate the skin depth. The skin depth is
        used to calculate the minimum cell width and the boundary thicknesses.
        Up to three resistivities can be provided:

        - float: Same resistivity for everything;
        - [min_width, boundaries];
        - [min_width, left boundary, right boundary].

    domain : list
        Contains the survey-domain limits [min, max]. The actual calculation
        domain consists of this domain plus a buffer zone around it, which
        depends on frequency and resistivity.

    fixed : list, optional
        Fixed boundaries, one, two, or maximum three values. The grid is
        centered around the first value. Hence it is the center location with
        the smallest cell. Two more fixed boundaries can be added, at most one
        on each side of the first one.
        Default is 0.

    possible_nx : list, optional
        List of possible numbers of cells. See :func:`get_cell_numbers`.
        Default is ``get_cell_numbers(500, 5, 3)``, which corresponds to
        [16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384].

    min_width : float, list or None, optional
        Minimum cell width restriction:

        - None : No restriction;
        - float : Fixed to this value, ignoring skin depth and `pps`.
        - list [min, max] : Lower and upper bounds.

        Default is None.

    pps : int, optional
        Points per skindepth; minimum cell width is calculated via
        `dmin = skindepth/pps`.
        Default = 3.

    alpha : list, optional
        Maximum alpha and step size to find a good alpha. The first value is
        the maximum alpha of the survey domain, the second value is the maximum
        alpha for the buffer zone, and the third value is the step size.
        Default = [1, 1.5, .01], hence no stretching within the survey domain
        and a maximum stretching of 1.5 in the buffer zone; step size is 0.01.

    max_domain : float, optional
        Maximum calculation domain from fixed[0] (usually source position).
        Default is 100,000.

    raise_error : bool, optional
        If True, an error is raised if no suitable grid is found. Otherwise it
        just prints a message and returns None's.
        Default is True.

    verb : int, optional
        Verbosity, 0 or 1.
        Default = 1.

    return_info : bool
        If True, a dictionary is returned with some grid info (min and max
        cell width and alpha).


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    x0 : float
        Origin of the mesh.

    info : dict
        Dictionary with mesh info; only if ``return_info=True``.

        Keys:

        - `dmin`: Minimum cell width;
        - `dmax`: Maximum cell width;
        - `amin`: Minimum alpha;
        - `amax`: Maximum alpha.

    """
    # Get variables with default lists:
    if alpha is None:
        alpha = [1, 1.5, 0.01]
    if possible_nx is None:
        possible_nx = get_cell_numbers(500, 5, 3)

    # Cast resistivity value(s).
    res = np.array(res, ndmin=1)
    if res.size == 1:
        res_arr = np.array([res[0], res[0], res[0]])
    elif res.size == 2:
        res_arr = np.array([res[0], res[1], res[1]])
    else:
        res_arr = np.array([res[0], res[1], res[2]])

    # Cast and check fixed.
    fixed = np.array(fixed, ndmin=1)
    if fixed.size > 2:

        # Check length.
        if fixed.size > 3:
            print("\n* ERROR   :: Maximum three fixed boundaries permitted.\n"
                  f"             Provided: {fixed.size}.")
            raise ValueError("Wrong input for fixed")

        # Sort second and third, so it doesn't matter how it was provided.
        fixed = np.array([fixed[0], max(fixed[1:]), min(fixed[1:])])

        # Check side.
        if np.sign(np.diff(fixed[:2])) == np.sign(np.diff(fixed[::2])):
            print("\n* ERROR   :: 2nd and 3rd fixed boundaries have to be "
                  "left and right of the first one.\n             "
                  f"Provided: [{fixed[0]}, {fixed[1]}, {fixed[2]}]")
            raise ValueError("Wrong input for fixed")

    # Calculate skin depth.
    skind = 503.3*np.sqrt(res_arr/abs(freq))
    if freq < 0:  # For Laplace-domain calculations.
        skind /= np.sqrt(2*np.pi)

    # Minimum cell width.
    dmin = skind[0]/pps
    if min_width is not None:  # Respect user input.
        min_width = np.array(min_width, ndmin=1)
        if min_width.size == 1:
            dmin = min_width
        else:
            dmin = np.clip(dmin, *min_width)

    # Survey domain; contains all sources and receivers.
    domain = np.array(domain, dtype=float)

    # Calculation domain; big enough to avoid boundary effects.
    # To avoid boundary effects we want the signal to travel two wavelengths
    # from the source to the boundary and back to the receiver.
    # => 2*pi*sd ~ 6.3*sd = one wavelength => signal is ~ 0.2 %.
    # Two wavelengths we can safely assume it is zero.
    #
    # The air does not follow the concept of skin depth, as it is a wave rather
    # than diffusion. For this is the factor `max_domain`, which restricts
    # the domain in each direction to this value from the center.

    # (a) Source to edges of domain.
    dist_in_domain = abs(domain - fixed[0])

    # (b) Two wavelengths.
    two_lambda = skind[1:]*4*np.pi

    # (c) Required buffer, additional to domain.
    dist_buff = np.max([np.zeros(2), (two_lambda - dist_in_domain)/2], axis=0)

    # (d) Add buffer to domain.
    calc_domain = np.array([domain[0]-dist_buff[0], domain[1]+dist_buff[1]])

    # (e) Restrict total domain to max_domain.
    calc_domain[0] = max(calc_domain[0], fixed[0]-max_domain)
    calc_domain[1] = min(calc_domain[1], fixed[0]+max_domain)

    # Initiate flag if terminated.
    finished = False

    # Initiate alpha variables for survey and calculation domains.
    sa, ca = 1.0, 1.0

    # Loop over possible cell numbers from small to big.
    for nx in np.unique(possible_nx):

        # Loop over possible alphas for domain.
        for sa in np.arange(1.0, alpha[0]+alpha[2]/2, alpha[2]):

            # Get current stretched grid cell sizes.
            thxl = dmin*sa**np.arange(nx)  # Left of origin.
            thxr = dmin*sa**np.arange(nx)  # Right of origin.

            # 0. Adjust stretching for fixed boundaries.
            if fixed.size > 1:  # Move mesh to first fixed boundary.
                t_nx = np.r_[fixed[0], fixed[0]+np.cumsum(thxr)]
                ii = np.argmin(abs(t_nx-fixed[1]))
                thxr *= abs(fixed[1]-fixed[0])/np.sum(thxr[:ii])

            if fixed.size > 2:  # Move mesh to second fixed boundary.
                t_nx = np.r_[fixed[0], fixed[0]-np.cumsum(thxl)]
                ii = np.argmin(abs(t_nx-fixed[2]))
                thxl *= abs(fixed[2]-fixed[0])/np.sum(thxl[:ii])

            # 1. Fill from center to left domain.
            nl = np.sum((fixed[0]-np.cumsum(thxl)) > domain[0])+1

            # 2. Fill from center to right domain.
            nr = np.sum((fixed[0]+np.cumsum(thxr)) < domain[1])+1

            # 3. Get remaining number of cells and check termination criteria.
            nsdc = nl+nr  # Number of domain cells.
            nx_remain = nx-nsdc

            # Not good, try next.
            if nx_remain <= 0:
                continue

            # Create the current hx-array.
            hx = np.r_[thxl[:nl][::-1], thxr[:nr]]
            hxo = np.r_[thxl[:nl][::-1], thxr[:nr]]

            # Get actual domain:
            asurv_domain = [fixed[0]-np.sum(thxl[:nl]),
                            fixed[0]+np.sum(thxr[:nr])]
            x0 = float(fixed[0]-np.sum(thxl[:nl]))

            # Get actual stretching (differs in case of fixed layers).
            sa_adj = np.max([hx[1:]/hx[:-1], hx[:-1]/hx[1:]])

            # Loop over possible alphas for calc_domain.
            for ca in np.arange(sa, alpha[1]+alpha[2]/2, alpha[2]):

                # 4. Fill to left calc_domain.
                thxl = hx[0]*ca**np.arange(1, nx_remain+1)
                nl = np.sum((asurv_domain[0]-np.cumsum(thxl)) >
                            calc_domain[0])+1

                # 5. Fill to right calc_domain.
                thxr = hx[-1]*ca**np.arange(1, nx_remain+1)
                nr = np.sum((asurv_domain[1]+np.cumsum(thxr)) <
                            calc_domain[1])+1

                # 6. Get remaining number of cells and check termination
                # criteria.
                ncdc = nl+nr  # Number of calc_domain cells.
                nx_remain2 = nx-nsdc-ncdc

                if nx_remain2 < 0:  # Not good, try next.
                    continue

                # Create hx-array.
                nl += int(np.floor(nx_remain2/2))  # If uneven, add one cell
                nr += int(np.ceil(nx_remain2/2))   # more on the right.
                hx = np.r_[thxl[:nl][::-1], hx, thxr[:nr]]

                # Calculate origin.
                x0 = float(asurv_domain[0]-np.sum(thxl[:nl]))

                # Mark it as finished and break out of the loop.
                finished = True
                break

            if finished:
                break

        if finished:
            break

    # Check finished and print info about found grid.
    if not finished:
        # Throw message if no solution was found.
        print("\n* ERROR   :: No suitable grid found; relax your criteria.\n")
        if raise_error:
            raise ArithmeticError("No grid found!")
        else:
            hx, x0 = None, None

    elif verb > 0:
        print(f"   Skin depth ", end="")
        if res.size == 1:
            print(f"         [m] : {skind[0]:.0f}")
        elif res.size == 2:
            print(f"(m/l-r)  [m] : {skind[0]:.0f} / {skind[1]:.0f}")
        else:
            print(f"(m/l/r)  [m] : {skind[0]:.0f} / {skind[1]:.0f} / "
                  f"{skind[2]:.0f}")
        print(f"   Survey domain       [m] : {domain[0]:.0f} - "
              f"{domain[1]:.0f}")
        print(f"   Calculation domain  [m] : {calc_domain[0]:.0f} - "
              f"{calc_domain[1]:.0f}")
        print(f"   Final extent        [m] : {x0:.0f} - "
              f"{x0+np.sum(hx):.0f}")
        extstr = f"   Min/max cell width  [m] : {min(hx):.0f} / "
        alstr = f"   Alpha survey"
        nrstr = "   Number of cells "
        if not np.isclose(sa, sa_adj):
            sastr = f"{sa:.3f} ({sa_adj:.3f})"
        else:
            sastr = f"{sa:.3f}"
        print(extstr+f"{max(hxo):.0f} / {max(hx):.0f}")
        print(alstr+f"/calc       : {sastr} / {ca:.3f}")
        print(nrstr+f"(s/c/r) : {nx} ({nsdc}/{ncdc}/{nx_remain2})")
        print()

    if return_info:
        if not fixed.size > 1:
            sa_adj = sa

        info = {'dmin': dmin,
                'dmax': np.nanmax(hx),
                'amin': np.nanmin([ca, sa, sa_adj]),
                'amax': np.nanmax([ca, sa, sa_adj])}

        return hx, x0, info
    else:
        return hx, x0


def get_cell_numbers(max_nr, max_prime=5, min_div=3):
    r"""Returns 'good' cell numbers for the multigrid method.

    'Good' cell numbers are numbers which can be divided by 2 as many times as
    possible. At the end there will be a low prime number.

    The function adds all numbers :math:`p 2^n \leq M` for :math:`p={2, 3, ...,
    p_\text{max}}` and :math:`n={n_\text{min}, n_\text{min}+1, ..., \infty}`;
    :math:`M, p_\text{max}, n_\text{min}` correspond to `max_nr`, `max_prime`,
    and `min_div`, respectively.


    Parameters
    ----------
    max_nr : int
        Maximum number of cells.

    max_prime : int
        Highest permitted prime number p for p*2^n. {2, 3, 5, 7} are good upper
        limits in order to avoid too big lowest grids in the multigrid method.
        Default is 5.

    min_div : int
        Minimum times the number can be divided by two.
        Default is 3.


    Returns
    -------
    numbers : array
        Array containing all possible cell numbers from lowest to highest.

    """
    # Primes till 20.
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19])

    # Sanity check; 19 is already ridiculously high.
    if max_prime > primes[-1]:
        print(f"* ERROR   :: Highest prime is {max_prime}, "
              "please use a value < 20.")
        raise ValueError("Highest prime too high")

    # Restrict to max_prime.
    primes = primes[primes <= max_prime]

    # Get possible values.
    # Currently restricted to prime*2**30 (for prime=2 => 1,073,741,824 cells).
    numbers = primes[:, None]*2**np.arange(min_div, 30)

    # Get unique values.
    numbers = np.unique(numbers)

    # Restrict to max_nr and return.
    return numbers[numbers <= max_nr]


def get_stretched_h(min_width, domain, nx, x0=0, x1=None, resp_domain=False):
    """Return cell widths for a stretched grid within the domain.

    Returns `nx` cell widths within `domain`, where the minimum cell width is
    `min_width`. The cells are not stretched within `x0` and `x1`, and outside
    uses a power-law stretching. The actual stretching factor and the number of
    cells left and right of `x0` and `x1` are find in a minimization process.

    The domain is not completely respected. The starting point of the domain
    is, but the endpoint of the domain might slightly shift (this is more
    likely the case for small `nx`, for big `nx` the shift should be small).
    The new endpoint can be obtained with ``domain[0]+np.sum(hx)``. If you want
    the domain to be respected absolutely, set ``resp_domain=True``. However,
    be aware that this will introduce one stretch-factor which is different
    from the other stretch factors, to accommodate the restriction. This
    one-off factor is between the left- and right-side of `x0`, or, if `x1` is
    provided, just after `x1`.


    See Also
    --------
    get_hx_x0 : Get `hx` and `x0` for a flexible number of `nx` with
                given bounds.


    Parameters
    ----------

    min_width : float
        Minimum cell width. If x1 is provided, the actual minimum cell width
        might be smaller than min_width.

    domain : list
        [start, end] of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. `x0` is restricted to `domain`.
        Default is 0.

    x1 : float
        If provided, then no stretching is applied between `x0` and `x1`. The
        non-stretched part starts at `x0` and stops at the first possible
        location at or after `x1`. `x1` is restricted to `domain`. This will
        min_width so that an integer number of cells fit within x0 and x1.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of `x0`, or, if `x1` is provided, just after `x1`.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """

    # Cast to arrays
    domain = np.array(domain, dtype=float)
    x0 = np.array(x0, dtype=float)
    x0 = np.clip(x0, *domain)  # Restrict to model domain
    min_width = np.array(min_width, dtype=float)
    if x1 is not None:
        x1 = np.array(x1, dtype=float)
        x1 = np.clip(x1, *domain)  # Restrict to model domain

    # If x1 is provided (a part is not stretched)
    if x1 is not None:

        # Store original values
        xlim_orig = domain.copy()
        nx_orig = int(nx)
        x0_orig = x0.copy()
        h_min_orig = min_width.copy()

        # Get number of non-stretched cells
        n_nos = int(np.ceil((x1-x0)/min_width))

        # Re-calculate min_width to fit with x0-x1-limits:
        min_width = (x1-x0)/n_nos

        # Subtract one cell, because the standard scheme provides one
        # min_width-cell.
        n_nos -= 1

        # Reset x0, because the first min_width comes from normal scheme
        x0 += min_width

        # Reset xmax for normal scheme
        domain[1] -= n_nos*min_width

        # Reset nx for normal scheme
        nx -= n_nos

        # If there are not enough points reset to standard procedure. The limit
        # of five is arbitrary. However, nx should be much bigger than five
        # anyways, otherwise stretched grid doesn't make sense.
        if nx <= 5:
            print("Warning :: Not enough points for non-stretched part,"
                  "ignoring therefore `x1`.")
            domain = xlim_orig
            nx = nx_orig
            x0 = x0_orig
            x1 = None
            min_width = h_min_orig

    # Get stretching factor (a = 1+alpha).
    if min_width == 0 or min_width > np.diff(domain)/nx:
        # If min_width is bigger than the domain-extent divided by nx, no
        # stretching is required at all.
        alpha = 0
    else:

        # Wrap _get_dx into a minimization function to call with fsolve.
        def find_alpha(alpha, min_width, args):
            """Find alpha such that min(hx) = min_width."""
            return min(get_hx(alpha, *args))/min_width-1

        # Search for best alpha, must be at least 0
        args = (domain, nx, x0)
        alpha = max(0, optimize.fsolve(find_alpha, 0.02, (min_width, args)))

    # With alpha get actual cell spacing with `resp_domain` to respect the
    # users decision.
    hx = get_hx(alpha, domain, nx, x0, resp_domain)

    # Add the non-stretched center if x1 is provided
    if x1 is not None:
        hx = np.r_[hx[: np.argmin(hx)], np.ones(n_nos)*min_width,
                   hx[np.argmin(hx):]]

    # Print warning min_width could not be respected.
    if abs(hx.min() - min_width) > 0.1:
        print(f"Warning :: Minimum cell width ({np.round(hx.min(), 2)} m) is "
              "below `min_width`, because `nx` is too big for `domain`.")

    return hx


def get_domain(x0=0, freq=1, res=0.3, limits=None, min_width=None,
               fact_min=0.2, fact_neg=5, fact_pos=None):
    r"""Get domain extent and minimum cell width as a function of skin depth.

    Returns the extent of the calculation domain and the minimum cell width as
    a multiple of the skin depth, with possible user restrictions on minimum
    calculation domain and range of possible minimum cell widths.

    .. math::

            \delta &= 503.3 \sqrt{\frac{\rho}{f}} , \\
            x_\text{start} &= x_0-k_\text{neg}\delta , \\
            x_\text{end} &= x_0+k_\text{pos}\delta , \\
            h_\text{min} &= k_\text{min} \delta .


    Parameters
    ----------

    x0 : float
        Center of the calculation domain. Normally the source location.
        Default is 0.

    freq : float
        Frequency (Hz) to calculate the skin depth. The skin depth is a concept
        defined in the frequency domain. If a negative frequency is provided,
        it is assumed that the calculation is carried out in the Laplace
        domain. To calculate the skin depth, the value of `freq` is then
        multiplied by :math:`-2\pi`, to simulate the closest
        frequency-equivalent.

        Default is 1 Hz.

    res : float, optional
        Resistivity (Ohm m) to calculate skin depth.
        Default is 0.3 Ohm m (sea water).

    limits : None or list
        [start, end] of model domain. This extent represents the minimum extent
        of the domain. The domain is therefore only adjusted if it has to reach
        outside of [start, end].
        Default is None.

    min_width : None, float, or list of two floats
        Minimum cell width is calculated as a function of skin depth:
        fact_min*sd. If `min_width` is a float, this is used. If a list of
        two values [min, max] are provided, they are used to restrain
        min_width. Default is None.

    fact_min, fact_neg, fact_pos : floats
        The skin depth is multiplied with these factors to estimate:

            - Minimum cell width (`fact_min`, default 0.2)
            - Domain-start (`fact_neg`, default 5), and
            - Domain-end (`fact_pos`, defaults to `fact_neg`).


    Returns
    -------

    h_min : float
        Minimum cell width.

    domain : list
        Start- and end-points of calculation domain.

    """

    # Set fact_pos to fact_neg if not provided.
    if fact_pos is None:
        fact_pos = fact_neg

    # Calculate the skin depth.
    skind = 503.3*np.sqrt(res/abs(freq))
    if freq < 0:  # For Laplace-domain calculations.
        skind /= np.sqrt(2*np.pi)

    # Estimate minimum cell width.
    h_min = fact_min*skind
    if min_width is not None:  # Respect user input.
        if np.array(min_width).size == 1:
            h_min = min_width
        else:
            h_min = np.clip(h_min, *min_width)

    # Estimate calculation domain.
    domain = [x0-fact_neg*skind, x0+fact_pos*skind]
    if limits is not None:  # Respect user input.
        domain = [min(limits[0], domain[0]), max(limits[1], domain[1])]

    return h_min, domain


def get_hx(alpha, domain, nx, x0, resp_domain=True):
    r"""Return cell widths for given input.

    Find the number of cells left and right of `x0`, `nl` and `nr`
    respectively, for the provided alpha. For this, we solve

    .. math::   \frac{x_\text{max}-x_0}{x_0-x_\text{min}} =
                \frac{a^{nr}-1}{a^{nl}-1}

    where :math:`a = 1+\alpha`.


    Parameters
    ----------

    alpha : float
        Stretching factor `a` is given by ``a=1+alpha``.

    domain : list
        [start, end] of model domain.

    nx : int
        Number of cells.

    x0 : float
        Center of the grid. `x0` is restricted to `domain`.

    resp_domain : bool
        If False (default), then the domain-end might shift slightly to assure
        that the same stretching factor is applied throughout. If set to True,
        however, the domain is respected absolutely. This will introduce one
        stretch-factor which is different from the other stretch factors, to
        accommodate the restriction. This one-off factor is between the left-
        and right-side of `x0`, or, if `x1` is provided, just after `x1`.


    Returns
    -------
    hx : ndarray
        Cell widths of mesh.

    """
    if alpha <= 0.:  # If alpha <= 0: equal spacing (no stretching at all)
        hx = np.ones(nx)*np.diff(np.squeeze(domain))/nx

    else:            # Get stretched hx
        a = alpha+1

        # Get hx depending if x0 is on the domain boundary or not.
        if np.isclose(x0, domain[0]) or np.isclose(x0, domain[1]):
            # Get al a's
            alr = np.diff(domain)*alpha/(a**nx-1)*a**np.arange(nx)
            if x0 == domain[1]:
                alr = alr[::-1]

            # Calculate differences
            hx = alr*np.diff(domain)/sum(alr)

        else:
            # Find number of elements left and right by solving:
            #     (xmax-x0)/(x0-xmin) = a**nr-1/(a**nl-1)
            nr = np.arange(2, nx+1)
            er = (domain[1]-x0)/(x0-domain[0]) - (a**nr[::-1]-1)/(a**nr-1)
            nl = np.argmin(abs(np.floor(er)))+1
            nr = nx-nl

            # Get all a's
            al = a**np.arange(nl-1, -1, -1)
            ar = a**np.arange(1, nr+1)

            # Calculate differences
            if resp_domain:
                # This version honours domain[0] and domain[1], but to achieve
                # this it introduces one stretch-factor which is different from
                # all the others between al to ar.
                hx = np.r_[al*(x0-domain[0])/sum(al),
                           ar*(domain[1]-x0)/sum(ar)]
            else:
                # This version moves domain[1], but each stretch-factor is
                # exactly the same.
                fact = (x0-domain[0])/sum(al)  # Take distance from al.
                hx = np.r_[al, ar]*fact

                # Note: this hx is equivalent as providing the following h
                # to TensorMesh:
                # h = [(min_width, nl-1, -a), (min_width, n_nos+1),
                #      (min_width, nr, a)]

    return hx
