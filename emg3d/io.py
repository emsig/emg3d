"""
Utility functions for writing and reading data.
"""
# Copyright 2018-2021 The emsig community.
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

import os
import json
import warnings
from datetime import datetime

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from emg3d import meshes, utils

__all__ = ['save', 'load']


def save(fname, **kwargs):
    """Save simulations, surveys, meshes, models, fields, and more to disk.

    Serialize and save data to disk in different formats (see parameter
    description of ``fname`` for the supported file formats).

    Any other (non-emg3d) object can be added too, as long as it knows how to
    serialize itself.

    The serialized instances will be de-serialized if loaded with
    :func:`emg3d.io.load`.


    Parameters
    ----------
    fname : str
        File name with absolute or relative path including suffix, which
        defines the used data format. Implemented are currently:

        - ``.h5``: Uses h5py to store inputs to a hierarchical, compressed
          binary HDF5 file. Recommended file format, but requires the module
          ``h5py``.
        - ``.npz``: Uses numpy to store inputs to a flat, compressed binary
          file.
        - ``.json``: Uses json to store inputs to a hierarchical, plain text
          file.

    compression : {int, str}, default: 'gzip'
        Passed through to h5py.

    json_indent : {int, None}, default: 2
        Passed through to json.

    verb : int, default: 1
        Verbose if 1, if 0 silent; if -1 it returns the info as string instead
        of printing it.

    kwargs : optional
        Data to save using its key as name.

        Note that the provided data cannot contain the before described
        parameters as keys.


    Returns
    -------
    info : str, returned if verb<0
        Info-string.

    """
    # Get and remove optional kwargs.
    compression = kwargs.pop('compression', 'gzip')
    json_indent = kwargs.pop('json_indent', 2)
    verb = kwargs.pop('verb', 1)

    # Add meta-data to kwargs
    kwargs['_date'] = datetime.today().isoformat()
    kwargs['_version'] = f"emg3d v{utils.__version__}"
    kwargs['_format'] = "1.0"  # File format; version of emg3d when changed.

    # Get hierarchical dictionary with serialized and sorted KNOWN_CLASSES.
    data = _dict_serialize(kwargs)

    # Ensure fname is absolute.
    fname = os.path.abspath(fname)

    # Save NumPy.
    if fname.endswith('.npz'):
        np.savez_compressed(fname, **_dict_flatten(data))

    # Save HDF5
    elif fname.endswith('.h5'):
        _hdf5_dump(fname, data=data, compression=compression)

    # Save JSON
    elif fname.endswith('.json'):
        with open(fname, "w") as f:
            json.dump(_dict_dearray_decomp(data), f, indent=json_indent)

    # Unknown, throw error
    else:
        raise ValueError(f"Unknown extension '.{fname.split('.')[-1]}'.")

    # Print file info.
    info = (f"Data saved to «{fname}»\n[{kwargs['_version']} "
            f"(format {kwargs['_format']}) on {kwargs['_date']}].")
    if verb > 0:
        print(info)
    elif verb < 0:
        return info


def load(fname, **kwargs):
    """Load simulations, surveys, meshes, models, fields, and more from disk.

    Load data and de-serialize known instances.


    Parameters
    ----------
    fname : str
        File name with absolute or relative path including suffix, which
        defines the used data format. Implemented are currently:

        - ``'.npz'``: NumPy-binary;
        - ``'.h5'``: HDF5-binary (requires ``h5py``);
        - ``'.json'``: JSON plain text file.

    verb : int, default: 1
        Verbose if 1, if 0 silent; if -1 it returns the info as string instead
        of printing it.


    Returns
    -------
    out : dict
        A dictionary containing the data stored in ``fname``;

    info : str, returned if verb<0
        Info-string.

    """
    # Get kwargs.
    verb = kwargs.pop('verb', 1)
    # allow_pickle is undocumented, but kept, just in case...
    allow_pickle = kwargs.pop('allow_pickle', False)
    # Ensure no kwargs left.
    if kwargs:
        raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}.")

    # Ensure fname is absolute.
    fname = os.path.abspath(fname)

    # Load NumPy.
    if fname.endswith('.npz'):
        with np.load(fname, allow_pickle=allow_pickle) as dat:
            data = {key: dat[key] for key in dat.files}
            data = _dict_unflatten(data)  # Un-flatten

    # Load HDF5
    elif fname.endswith('.h5'):
        data = _hdf5_load(fname)

    # Load JSON
    elif fname.endswith('.json'):
        with open(fname, 'r') as f:
            data = json.load(f)
            data = _dict_array_comp(data)  # compose arrays / complex data

    # Unknown, throw error
    else:
        raise ValueError(f"Unknown extension '.{fname.split('.')[-1]}'.")

    # De-serialize data.
    _nonetype_to_none(data)
    _dict_deserialize(data)

    # Check if file was (supposedly) created by emg3d.
    info = f"Data loaded from «{fname}»"
    try:
        version = data['_version']
        date = data['_date']
        form = data['_format']

        # Print file info.
        info += f"\n[{version} (format {form}) on {date}]."

    except KeyError:
        info += "\n[version/format/date unknown; not created by emg3d]."

    if verb > 0:
        print(info)

    if verb < 0:
        data = (data, info)

    return data


def _dict_serialize(inp):
    """Serialize emg3d-classes and other objects in inp-dict.

    Returns a serialized dictionary <out> of <inp>, where all members of
    `emg3d.utils._KNOWN_CLASSES` are serialized with their respective
    `to_dict()` methods.

    Any other (non-emg3d) object can be added too, as long as it knows how to
    serialize itself.

    There are some limitations:

    1. Key names are converted to strings.
    2. None values are converted to 'NoneType'.
    3. TensorMesh instances from discretize will be stored as if they would be
       simpler emg3d.TensorMesh instances.


    Parameters
    ----------
    inp : dict
        Input dictionary to serialize.


    Returns
    -------
    out : dict
        Serialized <inp>-dict.

    """

    # Initiate output dictionary.
    out = {}

    # Loop over items.
    for key, value in inp.items():

        # Serialize known classes.
        if isinstance(value, tuple(utils._KNOWN_CLASSES.values())):

            # Workaround for discretize.TensorMesh (store as emg3d.TensorMesh)
            if hasattr(value, 'face_areas'):
                value = meshes.TensorMesh(value.h, value.origin)

            # Serialize.
            value = value.to_dict()

        # If value is a dict we use recursion
        if isinstance(value, dict):
            value = _dict_serialize(value)

        # Limitation 1: None -> 'NoneType'
        elif value is None:
            value = 'NoneType'

        # Store value
        # Limitation 2: Cast keys -> str(key)
        out[str(key)] = value

    return out


def _dict_deserialize(inp):
    """De-serialize emg3d-classes and other objects in inp-dict.

    De-serializes in-place dictionary <inp>, where all members of
    `emg3d.utils._KNOWN_CLASSES` are de-serialized with their respective
    `from_dict()` methods.


    Parameters
    ----------
    inp : dict
        Input dictionary to de-serialize.

    """

    # Loop over items.
    for key, value in inp.items():

        # If it is a dict, de-serialize if known class or recursion.
        if isinstance(value, dict):

            # If it has a __class__-key, de-serialize.
            if '__class__' in value.keys():

                # De-serialize, overwriting all the existing entries.
                try:
                    inst = utils._KNOWN_CLASSES[value['__class__']]
                    inp[key] = inst.from_dict(value)
                    continue

                except (AttributeError, KeyError, TypeError) as e:
                    # Gracefully fail.
                    msg = f"emg3d: Could not de-serialize <{key}>: {e}"
                    warnings.warn(msg, UserWarning)

            # In no __class__-key or de-serialization fails, use recursion.
            _dict_deserialize(value)


def _nonetype_to_none(inp):
    """Recursively replace side-effects in inp-dict from storing to disc.

    Changes:

    - Replaces ``'NoneType'`` by ``None``.
    - Casts back ``np.bool_`` to ``bool`` (because ``bool`` is converted to
      ``np.bool_`` for some file formats).

    """
    for k, v in inp.items():
        if isinstance(v, dict):
            _nonetype_to_none(v)
        elif isinstance(v, str) and v == 'NoneType':
            inp[k] = None
        elif isinstance(v, np.bool_):
            inp[k] = bool(v)
        elif isinstance(v, np.ndarray) and v.dtype == np.bool_:
            inp[k] = bool(np.squeeze(v))


def _dict_flatten(data):
    """Return flattened dict of input dict <data>.

    After https://codereview.stackexchange.com/revisions/21035/3


    Parameters
    ----------
    data : dict
        Input dict to flatten.


    Returns
    -------
    out : dict
        Flattened dict.

    """

    def expand(key, value):
        """Expand list."""

        if isinstance(value, dict):
            return [(key+'>'+k, v) for k, v in _dict_flatten(value).items()]
        else:
            return [(key, value)]

    return dict([item for k, v in data.items() for item in expand(k, v)])


def _dict_unflatten(data):
    """Return un-flattened dict of input dict <data>.

    After https://stackoverflow.com/a/6037657


    Parameters
    ----------
    data : dict
        Input dict to un-flatten.

    Returns
    -------
    out : dict
        Un-flattened dict.

    """

    # Initialize output dict.
    out = {}

    # Loop over items.
    for key, value in data.items():

        # Split the keys.
        parts = key.split(">")

        # Initiate tmp dict.
        tmp = out

        # Loop over key-parts.
        for part in parts[:-1]:

            # If sub-key does not exist yet, initiate sub-dict.
            if part not in tmp:
                tmp[part] = {}

            # Add value to subdict.
            tmp = tmp[part]

        # Convert numpy strings to str.
        if isinstance(value, np.ndarray) and value.dtype.type == np.str_:
            value = str(value)

        # Store actual value of this key.
        tmp[parts[-1]] = value

    return out


def _dict_dearray_decomp(data):
    """Return dict where arrays are replaced by lists, complex by real numbers.


    Parameters
    ----------
    data : dict
        Input dict to de-compose and de-array.


    Returns
    -------
    out : dict
        As input, but arrays are moved to lists, and complex number to real
        numbers like [real, imag].

    """

    # Output dict.
    out = {}

    # Loop over keys.
    for key, value in data.items():

        # Recursion.
        if isinstance(value, dict):
            value = _dict_dearray_decomp(value)

        # Decompose complex values.
        if np.iscomplexobj(value):
            key += '__complex'
            value = np.stack([np.asarray(value).real, np.asarray(value).imag])

        # Convert arrays to lists.
        if isinstance(value, np.ndarray):
            key += '__array-'+value.dtype.name
            value = value.tolist()

        # Store this key-value-pair.
        out[key] = value

    return out


def _dict_array_comp(data):
    """Return dict where lists/complex are moved back to arrays.


    Parameters
    ----------
    data : dict
        Input dict to compose.


    Returns
    -------
    out : dict
        As input, but lists are again arrays and complex data are complex
        again.

    """

    # Output dict.
    out = {}

    # Loop over keys.
    for key, value in data.items():

        # Recursion.
        if isinstance(value, dict):
            value = _dict_array_comp(value)

        # Get arrays back.
        if '__array' in key:
            arraytype = key.split('__')[-1]
            dtype = getattr(np, arraytype[6:])
            value = np.asarray(value, dtype=dtype, order='F')
            key = key.replace(key[-len(arraytype)-2:], '')

        # Compose complex numbers.
        if '__complex' in key:
            value = np.asarray(value)[0, ...] + 1j*np.asarray(value)[1, ...]
            key = key.replace('__complex', '')

        # Store this key-value-pair.
        out[key] = value

    return out


@utils._requires('h5py')
def _hdf5_dump(fname, data, compression):
    """Adds dictionary entries recursively to hdf5 file fname.


    Parameters
    ----------
    fname : str
        Absolute path/name of a HDF5-file, ending in .h5.
        (In recursion it is an HDF5-file handle).

    data : dict
        Dictionary containing the data.

    compression : {str, int}
        Passed through to h5py.

    """

    if isinstance(fname, str):
        with h5py.File(fname, "w") as h5file:
            _hdf5_dump(h5file, data, compression)

    else:
        # Loop over items.
        for key, value in data.items():

            # Use recursion if value is a dict, creating a new group.
            if isinstance(value, dict):
                _hdf5_dump(fname.create_group(key, track_order=True),
                           value, compression)

            elif np.ndim(value) > 0:  # Use compression where possible...
                fname.create_dataset(key, data=value, compression=compression)

            else:                    # else store without compression.
                fname.create_dataset(key, data=value)


@utils._requires('h5py')
def _hdf5_load(fname):
    """Return data from fname in a dict.


    Parameters
    ----------
    fname : file
        Absolute path/name of a HDF5-file, ending in .h5.


    Returns
    -------
    data : dict
        Dictionary containing the data.

    """

    if isinstance(fname, str):
        with h5py.File(fname, "r") as h5file:
            data = _hdf5_load(h5file)
        return data

    else:

        # Initiate dictionary.
        data = {}

        # Loop over items.
        for key, value in fname.items():

            # If it is a dataset add value to key, else use recursion.
            if isinstance(value, h5py._hl.dataset.Dataset):
                value = value[()]

                # h5py>=3.0 changed strings to byte strings.
                if isinstance(value, bytes):
                    data[key] = value.decode("utf-8")
                elif (isinstance(value, np.ndarray) and
                      value.dtype == 'object' and
                      isinstance(value[0], bytes)):
                    data[key] = [x.decode("utf-8") for x in value]
                else:
                    data[key] = value

            elif isinstance(value, h5py._hl.group.Group):
                data[key] = _hdf5_load(value)

        return data
