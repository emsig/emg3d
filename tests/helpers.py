"""

Helpers
=======

These are some helper functions for the test suite.
"""
import numpy as np


def widths(ncore, npad, width, factor):
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


def compare_dicts(dict1, dict2, verb=False, **kwargs):
    """Return True if the two dicts `dict1` and `dict2` are the same.

    Private method, not foolproof. Useful for developing new extensions.

    If `verb=True`, it prints it key starting with the following legend:

      - True : Values are the same.
      - False : Values are not the same.
      - {1} : Key is only in dict1 present.
      - {2} : Key is only in dict2 present.

    Private keys (starting with an underscore) are ignored.


    Parameters
    ----------
    dict1, dict2 : dicts
        Dictionaries to compare.

    verb : bool
        If True, prints all keys and if they are the  same for that key.

    kwargs : dict
        For recursion.


    Returns
    -------
    same : bool
        True if dicts are the same, False otherwise.

    """
    # Get recursion kwargs.
    s = kwargs.pop('s', '')
    reverse = kwargs.pop('reverse', False)
    gsame = kwargs.pop('gsame', True)

    # Check if we are at the base level and in reverse mode or not.
    do_reverse = len(s) == 0 and reverse is False

    # Loop over key-value pairs.
    for key, value in dict1.items():

        # Recursion if value is dict and present in both dicts.
        if isinstance(value, dict) and key in dict2.keys():

            # Add current key to string.
            s += f"{key[:10]:11}> "

            # Recursion.
            compare_dicts(dict1[key], dict2[key], verb=verb, s=s,
                          reverse=reverse, gsame=gsame)

            # Remove current key.
            s = s[:-13]

        elif key.startswith('_'):  # Ignoring private keys.
            pass

        else:  # Do actual comparison.

            # Check if key in both dicts.
            if key in dict2.keys():

                # If reverse, the key has already been checked.
                if reverse is False:

                    # Compare.
                    same = np.all(value == dict2[key])

                    # Update global bool.
                    gsame *= same

                    if verb:
                        print(f"{bool(same)!s:^7}:: {s}{key}")

                    # Clean string.
                    s = len(s)*' '

            else:  # If only in one dict -> False.

                gsame = False

                if verb:
                    print(f"  {{{2 if reverse else 1}}}  :: {s}{key}")

    # Do the same reverse, do check for keys in dict2 which are not in dict1.
    if do_reverse:
        gsame = compare_dicts(dict2, dict1, verb, reverse=True, gsame=gsame)

    return gsame
