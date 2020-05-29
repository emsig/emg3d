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
import json
import shelve
import warnings
import numpy as np
from datetime import datetime

from emg3d import fields, models, utils, meshes, survey

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
    'Survey': survey.Survey,
    'Dipole': survey.Dipole,
    'SourceField': fields.SourceField,
    'TensorMesh': meshes.TensorMesh,
}


def data_write(fname, keys, values, path='data', exists=0):
    """DEPRECATED; USE :func:`save`.


    Parameters
    ----------
    fname : str
        File name.

    keys : str or list of str
        Name(s) of the values to store in file.

    values : anything
        Values to store with keys in file.

    path : str, optional
        Absolute or relative path where to store. Default is 'data'.

    exists : int, optional
        Flag how to act if a shelve with the given name already exists:

        - < 0: Delete existing shelve.
        - 0 (default): Do nothing (print that it exists).
        - > 0: Append to existing shelve.

    """
    # Issue warning
    mesg = ("\n    The use of `data_write` and `data_read` is deprecated.\n"
            "    These function will be removed before v1.0.\n"
            "    Use `emg3d.save` and `emg3d.load` instead.")
    warnings.warn(mesg, DeprecationWarning)

    # Get absolute path, create if it doesn't exist.
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, fname)

    # Check if shelve exists.
    bak_exists = os.path.isfile(full_path+".bak")
    dat_exists = os.path.isfile(full_path+".dat")
    dir_exists = os.path.isfile(full_path+".dir")
    if any([bak_exists, dat_exists, dir_exists]):
        print("   > File exists, ", end="")
        if exists == 0:
            print("NOT SAVING THE DATA.")
            return
        elif exists > 0:
            print("appending to it", end='')
        else:
            print("overwriting it.")
            for extension in ["dat", "bak", "dir"]:
                try:
                    os.remove(full_path+"."+extension)
                except FileNotFoundError:
                    pass

    # Cast into list.
    if not isinstance(keys, (list, tuple)):
        keys = [keys, ]
        values = [values, ]

    # Shelve it.
    with shelve.open(full_path) as db:

        # If appending, print the keys which will be overwritten.
        if exists > 0:
            over = [j for j in keys if any(i == j for i in list(db.keys()))]
            if len(over) > 0:
                print(" (overwriting existing key(s) "+f"{over}"[1:-1]+").")
            else:
                print(".")

        # Writing it to the shelve.
        for i, key in enumerate(keys):

            # If the parameter is a TensorMesh instance, we set the volume
            # None. This saves space, and it will simply be reconstructed if
            # required.
            if type(values[i]).__name__ == 'TensorMesh':
                if hasattr(values[i], '_vol'):
                    delattr(values[i], '_vol')

            db[key] = values[i]


def data_read(fname, keys=None, path="data"):
    """DEPRECATED; USE :func:`load`.


    Parameters
    ----------
    fname : str
        File name.

    keys : str, list of str, or None; optional
        Name(s) of the values to get from file. If None, returns everything as
        a dict. Default is None.

    path : str, optional
        Absolute or relative path where fname is stored. Default is 'data'.


    Returns
    -------
    out : values or dict
        Requested value(s) or dict containing everything if keys=None.

    """
    # Issue warning
    mesg = ("\n    The use of `data_write` and `data_read` is deprecated.\n"
            "    These functions will be removed before v1.0.\n"
            "    Use `save` and `load` instead.")
    warnings.warn(mesg, DeprecationWarning)

    # Get absolute path.
    path = os.path.abspath(path)
    full_path = os.path.join(path, fname)

    # Check if shelve exists.
    for extension in [".dat", ".bak", ".dir"]:
        if not os.path.isfile(full_path+extension):
            print(f"   > File <{full_path+extension}> does not exist.")
            if isinstance(keys, (list, tuple)):
                return len(keys)*(None, )
            else:
                return None

    # Get it from shelve.
    with shelve.open(path+"/"+fname) as db:
        if keys is None:                           # None
            out = dict()
            for key, item in db.items():
                out[key] = item
            return out

        elif not isinstance(keys, (list, tuple)):  # single parameter
            return db[keys]

        else:                                      # lists/tuples of parameters
            out = []
            for key in keys:
                out.append(db[key])
            return out


def save(fname, backend="h5", compression="gzip", **kwargs):
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

        - `h5` (default): Uses `h5py` to store inputs to a hierarchical,
          compressed binary hdf5 file with the extension '.h5'. Recommended and
          default backend, but requires the module `h5py`. Use `npz` or `json`
          if you don't want to install `h5py`.
        - `npz`: Uses `numpy` to store inputs to a flat, compressed binary
          file with the extension '.npz'.
        - `json`: Uses `json` to store inputs to a hierarchical, plain '.json'
          file with the extension '.json'.

    compression : int or str, optional
        Passed through to h5py, default is 'gzip'.

    json_indent : int or None
        Passed through to json, default is 2.

    kwargs : Keyword arguments, optional
        Data to save using its key as name. The following instances will be
        properly serialized: :class:`emg3d.meshes.TensorMesh`,
        :class:`emg3d.fields.Field`, and :class:`emg3d.models.Model` and
        serialized again if loaded with :func:`load`. These instances are
        collected in their own group if h5py is used.

        Note that the provided data cannot contain the before described
        parameters as keys.

    """
    # Get and remove optional kwargs.
    json_indent = kwargs.pop('json_indent', 2)

    # Get absolute path.
    full_path = os.path.abspath(fname)

    # Add meta-data to kwargs
    kwargs['_date'] = datetime.today().isoformat()
    kwargs['_version'] = 'emg3d v' + utils.__version__
    kwargs['_format'] = '0.11.1'  # File format; version of emg3d when changed.

    # Get hierarchical dictionary with serialized and
    # sorted TensorMesh, Field, and Model instances.
    data = _dict_serialize(kwargs)

    # Deprecated backends.
    if backend == 'numpy':
        mesg = ("\n    The use of `backend='numpy'` is deprecated and will\n"
                "    be removed. Use `backend='npz'` instead.")
        warnings.warn(mesg, DeprecationWarning)
        backend = 'npz'
    elif backend == 'h5py':
        mesg = ("\n    The use of `backend='h5py'` is deprecated and will\n"
                "    be removed. Use `backend='h5'` instead.")
        warnings.warn(mesg, DeprecationWarning)
        backend = 'h5'

    # Add file-ending if necessary.
    if not full_path.endswith('.'+backend):
        full_path += '.'+backend

    # Save data depending on the backend.
    if backend == "npz":

        # Convert hierarchical dict to a flat dict.
        data = _dict_flatten(data)

        # Store flattened data.
        np.savez_compressed(full_path, **data)

    elif backend == "h5":

        # Check if h5py is installed.
        if isinstance(h5py, str):
            print(h5py)
            raise ImportError("backend='h5'")

        # Store data.
        with h5py.File(full_path, "w") as h5file:
            _hdf5_add_to(data, h5file, compression)

    elif backend == "json":

        # Move arrays to lists and decompose complex data.
        data = _dict_dearray_decomp(data)

        # Store hierarchical data.
        with open(full_path, "w") as f:
            json.dump(data, f, indent=json_indent)

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
        - '.json': json

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
    if fname.endswith('.npz'):

        # Load .npz into a flat dict.
        with np.load(full_path, allow_pickle=allow_pickle) as dat:
            data = {key: dat[key] for key in dat.files}

        # Un-flatten data.
        data = _dict_unflatten(data)

    elif fname.endswith('.h5'):

        # Check if h5py is installed.
        if isinstance(h5py, str):
            print(h5py)
            raise ImportError("backend='h5'")

        # Load data.
        with h5py.File(full_path, 'r') as h5file:
            data = _hdf5_get_from(h5file)

    elif fname.endswith('.json'):

        with open(full_path, 'r') as f:
            data = json.load(f)

        # Move lists back to arrays and compose complex data.
        data = _dict_array_comp(data)

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
        Input dict to flatten.


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
        Input dict to un-flatten.

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
        if '<U' in str(np.asarray(value).dtype):
            value = str(value)

        # Store actual value of this key.
        tmp[parts[-1]] = value

    return out


def _dict_dearray_decomp(data):
    """Return dict where arrays are replaced by lists, complex by real numbers.


    Parameters
    ----------
    data : dict
        Input dict to decompose.


    Returns
    -------
    ddata : dict
        As input, but arrays are moved to lists, and complex number to real
        numbers like [real, imag].

    """

    # Output dict.
    ddata = {}

    # Loop over keys.
    for key, value in data.items():

        # Recursion.
        if isinstance(value, dict):
            value = _dict_dearray_decomp(value)

        # Test if complex.
        if np.iscomplexobj(value):
            key += '__complex'
            value = np.stack([np.asarray(value).real, np.asarray(value).imag])

        # Convert to lists if no arrays wanted.
        if isinstance(value, np.ndarray):
            key += '__array-'+value.dtype.name
            value = value.tolist()

        # Store this key-value-pair.
        ddata[key] = value

    return ddata


def _dict_array_comp(data):
    """Return dict where lists/complex are moved back to arrays.


    Parameters
    ----------
    data : dict
        Input dict to compose.


    Returns
    -------
    ddata : dict
        As input, but lists are again arrays and complex data are complex
        again.

    """

    # Output dict.
    ddata = {}

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
        ddata[key] = value

    return ddata


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
            h5file.create_dataset(key, data=value, compression=compression)

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


def _compare_dicts(dict1, dict2, verb=False, **kwargs):
    """Return True if the two dicts `dict1` and `dict2` are the same.

    Private method, not foolproof. Useful for developing new backends.

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
            _compare_dicts(dict1[key], dict2[key], verb=verb, s=s,
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
        gsame = _compare_dicts(dict2, dict1, verb, reverse=True, gsame=gsame)

    return gsame
