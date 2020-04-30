"""

:mod:`io` -- I/O utilities
==========================

Utility functions for writing and reading data.

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


import os
import numpy as np
from datetime import datetime

from emg3d.utils import fields, models, misc, meshes

try:
    import h5py
except ImportError:
    h5py = ("\n* ERROR   :: '.h5'-files require `h5py`."
            "\n             Install it via `pip install h5py` or"
            "\n             `conda install -c conda-forge h5py`.\n")


__all__ = ['save', 'load']


# Known classes to serialize and de-serialize.
KNOWN_CLASSES = {
    'Model': models.Model,
    'Field': fields.Field,
    'TensorMesh': meshes.TensorMesh,
}

def save(fname, backend="h5py", compression="gzip", **kwargs):
    """Save meshes, models, fields, and other data to disk.

    Serialize and save :class:`emg3d.meshes.TensorMesh`,
    :class:`emg3d.fields.Field`, and :class:`emg3d.models.Model` instances and
    add arbitrary other data, where instances of the same type are grouped
    together.

    The serialized instances will be de-serialized if loaded with :func:`load`.


    Parameters
    ----------
    fname : str
        File name.

    backend : str, optional
        Backend to use. Implemented are currently:

        - `h5py` (default): Uses `h5py` to store inputs to a hierarchical,
          compressed binary hdf5 file with the extension '.h5'. Recommended and
          default backend, but requires the module `h5py`. Use `numpy` if you
          don't want to install `h5py`.
        - `numpy`: Uses `numpy` to store inputs to a flat, compressed binary
          file with the extension '.npz'.

    compression : int or str, optional
        Passed through to h5py, default is 'gzip'.

    kwargs : Keyword arguments, optional
        Data to save using its key as name. The following instances will be
        properly serialized: :class:`emg3d.meshes.TensorMesh`,
        :class:`emg3d.fields.Field`, and :class:`emg3d.models.Model` and
        serialized again if loaded with :func:`load`. These instances are
        collected in their own group if h5py is used.

    """
    # Get absolute path.
    full_path = os.path.abspath(fname)

    # Add meta-data to kwargs
    kwargs['_date'] = datetime.today().isoformat()
    kwargs['_version'] = 'emg3d v' + misc.__version__
    kwargs['_format'] = '0.10.0'  # File format; version of emg3d when changed.

    # Get hierarchical dictionary with serialized and
    # sorted TensorMesh, Field, and Model instances.
    data = _dict_serialize(kwargs)

    # Save data depending on the backend.
    if backend == "numpy":

        # Add .npz if necessary.
        if not full_path.endswith('.npz'):
            full_path += '.npz'

        # Store flattened data.
        np.savez_compressed(full_path, **_dict_flatten(data))

    elif backend == "h5py":

        # Add .h5 if necessary.
        if not full_path.endswith('.h5'):
            full_path += '.h5'

        # Check if h5py is installed.
        if isinstance(h5py, str):
            print(h5py)
            raise ImportError("backend='h5py'")

        # Store data.
        with h5py.File(full_path, "w") as h5file:
            _hdf5_add_to(data, h5file, compression)

    else:
        raise NotImplementedError(f"Backend '{backend}' is not implemented.")


def load(fname, **kwargs):
    """Load meshes, models, fields, and other data from disk.

    Load and de-serialize :class:`emg3d.meshes.TensorMesh`,
    :class:`emg3d.fields.Field`, and :class:`emg3d.models.Model` instances and
    add arbitrary other data that were saved with :func:`save`.


    Parameters
    ----------
    fname : str
        File name including extension. Used backend depends on the file
        extensions:

        - '.npz': numpy-binary
        - '.h5': h5py-binary (needs `h5py`)

    verb : int
        If 1 (default) verbose, if 0 silent.


    Returns
    -------
    out : dict
        A dictionary containing the data stored in fname;
        :class:`emg3d.meshes.TensorMesh`, :class:`emg3d.fields.Field`, and
        :class:`emg3d.models.Model` instances are de-serialized and returned as
        instances.

    """
    # Get kwargs.
    verb = kwargs.pop('verb', 1)
    # allow_pickle is undocumented, but kept, just in case...
    allow_pickle = kwargs.pop('allow_pickle', False)
    # Ensure no kwargs left.
    if kwargs:
        raise TypeError(f"Unexpected **kwargs: {list(kwargs.keys())}")

    # Get absolute path.
    full_path = os.path.abspath(fname)

    # Load data depending on the file extension.
    if fname.endswith('npz'):

        # Load .npz into a flat dict.
        with np.load(full_path, allow_pickle=allow_pickle) as dat:
            data = {key: dat[key] for key in dat.files}

        # Un-flatten data.
        data = _dict_unflatten(data)

    elif fname.endswith('h5'):

        # Check if h5py is installed.
        if isinstance(h5py, str):
            print(h5py)
            raise ImportError("backend='h5py'")

        # Load data.
        with h5py.File(full_path, 'r') as h5file:
            data = _hdf5_get_from(h5file)

    else:
        ext = fname.split('.')[-1]
        raise NotImplementedError(f"Extension '.{ext}' is not implemented.")

    # De-serialize data.
    _dict_deserialize(data)

    # Check if file was (supposedly) created by emg3d.
    try:
        version = data['_version']
        date = data['_date']
        form = data['_format']

        # Print file info.
        if verb > 0:
            print(f"  Loaded file {full_path}")
            print(f"  -> Stored with {version} (format {form}) on {date}")

    except KeyError:
        if verb > 0:
            print(f"\n* NOTE    :: {full_path} was not created by emg3d.")

    return data


def _dict_serialize(inp, out=None, top=True):
    """Serialize TensorMesh, Field, and Model instances in dict.

    Returns a serialized dictionary <out> of <inp>, where all
    :class:`emg3d.meshes.TensorMesh`, :class:`emg3d.fields.Field`, and
    :class:`emg3d.models.Model` instances have been serialized.

    These instances are additionally grouped together in dictionaries, and all
    other stuff is put into 'Data'.

    There are some limitations:

    1. Key names are converted to strings.
    2. None values are converted to 'NoneType'.
    3. TensorMesh instances from discretize will be stored as if they would be
       simpler emg3d-meshes.


    Parameters
    ----------
    inp : dict
        Input dictionary to serialize.

    out : dict
        Output dictionary; created if not provided.

    top : bool
        Used for recursion.


    Returns
    -------
    out : dict
        Serialized <inp>-dict.

    """

    # Initiate output dictionary if not provided.
    if out is None:
        output = True
        out = {}
    else:
        output = False

    # Loop over items.
    for key, value in inp.items():

        # Limitation 1: Cast keys to string
        if not isinstance(key, str):
            key = str(key)

        # Take care of the following instances (if we are in the
        # top-directory they get their own category):
        if (isinstance(value, tuple(KNOWN_CLASSES.values())) or
                hasattr(value, 'x0')):

            # Name of the instance
            name = value.__class__.__name__

            # Workaround for discretize.TensorMesh -> stored as if TensorMesh.
            if hasattr(value, 'to_dict'):
                to_dict = value.to_dict()
            else:

                try:
                    to_dict = {'hx': value.hx, 'hy': value.hy, 'hz': value.hz,
                               'x0': value.x0, '__class__': name}
                except AttributeError:  # Gracefully fail.
                    print(f"* WARNING :: Could not serialize <{key}>")
                    continue

            # If we are in the top-directory put them in their own category.
            if top:
                value = {key: to_dict}
                key = name
            else:
                value = to_dict

        elif top:
            if key.startswith('_'):  # Store meta-data in top-level...
                out[key] = value
                continue
            else:                    # ...rest falls into Data/.
                value = {key: value}
                key = 'Data'

        # Initiate if necessary.
        if key not in out.keys():
            out[key] = {}

        # If value is a dict use recursion, else store.
        if isinstance(value, dict):
            _dict_serialize(value, out[key], False)
        else:
            # Limitation 2: None
            if value is None:
                out[key] = 'NoneType'
            else:
                out[key] = value

    # Return if it wasn't provided.
    if output:
        return out


def _dict_deserialize(inp):
    """De-serialize TensorMesh, Field, and Model instances in dict.

    De-serializes in-place dictionary <inp>, where all
    :class:`emg3d.meshes.TensorMesh`, :class:`emg3d.fields.Field`, and
    :class:`emg3d.models.Model` instances have been de-serialized. It also
    converts back 'NoneType'-strings to None.


    Parameters
    ----------
    inp : dict
        Input dictionary to de-serialize.

    """

    # Loop over items.
    for key, value in inp.items():

        # Analyze if it is a dict, else ignore (check for 'NoneType').
        if isinstance(value, dict):

            # If it has a __class__-key, de-serialize.
            if '__class__' in value.keys():

                for k2, v2 in value.items():
                    if isinstance(v2, str) and v2 == 'NoneType':
                        value[k2] = None

                # De-serialize, overwriting all the existing entries.
                try:
                    inst = KNOWN_CLASSES[value['__class__']]
                    inp[key] = inst.from_dict(value)
                    continue

                except (AttributeError, KeyError):  # Gracefully fail.
                    print(f"* WARNING :: Could not de-serialize <{key}>")

            # In no __class__-key or de-serialization fails, use recursion.
            _dict_deserialize(value)

        elif isinstance(value, str) and value == 'NoneType':
            inp[key] = None


def _dict_flatten(data):
    """Return flattened dict of input dict <data>.

    After https://codereview.stackexchange.com/revisions/21035/3


    Parameters
    ----------
    data : dict
        Input dict to flatten

    Returns
    -------
    fdata : dict
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
        Input dict to un-flatten

    Returns
    -------
    udata : dict
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

            # If subkey does not exist yet, initiate subdict.
            if part not in tmp:
                tmp[part] = {}

            # Add value to subdict.
            tmp = tmp[part]

        # Convert numpy strings to str.
        if '<U' in str(value.dtype):
            value = str(value)

        # Store actual value of this key.
        tmp[parts[-1]] = value

    return out


def _hdf5_add_to(data, h5file, compression):
    """Adds dictionary entries recursively to h5.


    Parameters
    ----------
    data : dict
        Dictionary containing the data.

    h5file : file
        Opened by h5py.

    compression : str or int
        Passed through to h5py.

    """

    # Loop over items.
    for key, value in data.items():

        # Use recursion if value is a dict, creating a new group.
        if isinstance(value, dict):
            _hdf5_add_to(value, h5file.create_group(key), compression)

        elif np.ndim(value) > 0:  # Use compression where possible...

            # There are issues with lists of strings.
            # This should be handled better.
            try:
                h5file.create_dataset(key, data=value, compression=compression)
            except:
                h5file.create_dataset(key, data=np.string_(value),
                                      compression=compression)

        else:                    # else store without compression.
            h5file.create_dataset(key, data=value)


def _hdf5_get_from(h5file):
    """Return data from h5file in a dictionary.


    Parameters
    ----------
    h5file : file
        Opened by h5py.


    Returns
    -------
    data : dict
        Dictionary containing the data.

    """
    # Initiate dictionary.
    data = {}

    # Loop over items.
    for key, value in h5file.items():

        # If it is a dataset add value to key, else use recursion to dig in.
        if isinstance(value, h5py._hl.dataset.Dataset):
            data[key] = value[()]

        elif isinstance(value, h5py._hl.group.Group):
            data[key] = _hdf5_get_from(value)

    return data
