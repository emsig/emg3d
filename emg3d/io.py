"""
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
from datetime import datetime

import numpy as np

try:
    import h5py
except ImportError:
    h5py = ("'.h5'-files require `h5py`. Install it via\n"
            "`pip install h5py` or `conda install -c conda-forge h5py`.")

from emg3d import fields, maps, models, utils, meshes, surveys, simulations

__all__ = ['save', 'load']

# Known classes to serialize and de-serialize.
KNOWN_CLASSES = {
    '_Map': maps._Map,
    'Model': models.Model,
    'Field': fields.Field,
    'Survey': surveys.Survey,
    'Dipole': surveys.Dipole,
    'TensorMesh': meshes.TensorMesh,
    'SourceField': fields.SourceField,
    'Simulation': simulations.Simulation,
}


def save(fname, **kwargs):
    """Save surveys, meshes, models, fields, and more to disk.

    Serialize and save data to disk in different formats (see parameter
    description of `fname` for the supported file formats). The main
    emg3d-classes (type `emg3d.io.KNOWN_CLASSES` to get a list) can be
    collected in corresponding root-folders by setting `collect_classes=True`.

    Any other (non-emg3d) object can be added too, as long as it knows how to
    serialize itself.

    The serialized instances will be de-serialized if loaded with :func:`load`.


    Parameters
    ----------
    fname : str
        File name inclusive ending, which defines the used data format.
        Implemented are currently:

        - `.h5`: Uses `h5py` to store inputs to a hierarchical, compressed
          binary hdf5 file. Recommended file format, but requires the module
          `h5py`.
        - `.npz`: Uses `numpy` to store inputs to a flat, compressed binary
          file.
        - `.json`: Uses `json` to store inputs to a hierarchical, plain text
          file.

    compression : int or str, optional
        Passed through to h5py, default is 'gzip'.

    json_indent : int or None
        Passed through to json, default is 2.

    collect_classes : bool
        If True, input data is collected in folders for the principal
        emg3d-classes (type `emg3d.io.KNOWN_CLASSES` to get a list) and
        everything else collected in a `Data`-folder. Defaults to False.

    verb : int
        If 1 (default) verbose, if 0 silent.

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
    compression = kwargs.pop('compression', 'gzip')
    json_indent = kwargs.pop('json_indent', 2)
    collect_classes = kwargs.pop('collect_classes', False)
    verb = kwargs.pop('verb', 1)

    # Get absolute path.
    full_path = os.path.abspath(fname)

    # Add meta-data to kwargs
    kwargs['_date'] = datetime.today().isoformat()
    kwargs['_version'] = 'emg3d v' + utils.__version__
    kwargs['_format'] = '0.13.0'  # File format; version of emg3d when changed.

    # Get hierarchical dictionary with serialized and
    # sorted TensorMesh, Field, and Model instances.
    data = _dict_serialize(kwargs, collect_classes=collect_classes)

    # Save data depending on the extension.
    if full_path.endswith('.npz'):

        # Convert hierarchical dict to a flat dict.
        data = _dict_flatten(data)

        # Store flattened data.
        np.savez_compressed(full_path, **data)

    elif full_path.endswith('.h5'):

        # Check if h5py is installed.
        if isinstance(h5py, str):
            raise ImportError(h5py)

        # Store data.
        with h5py.File(full_path, "w") as h5file:
            _hdf5_add_to(data, h5file, compression)

    elif full_path.endswith('.json'):

        # Move arrays to lists and decompose complex data.
        data = _dict_dearray_decomp(data)

        # Store hierarchical data.
        with open(full_path, "w") as f:
            json.dump(data, f, indent=json_indent)

    else:
        ext = full_path.split('.')[-1]
        raise ValueError(f"Unknown extension '.{ext}'.")

    # Print file info.
    if verb > 0:
        print(f"Data saved to «{full_path}»\n[{kwargs['_version']} "
              f"(format {kwargs['_format']}) on {kwargs['_date']}].")


def load(fname, **kwargs):
    """Load meshes, models, fields, and other data from disk.

    Load and de-serialize :class:`emg3d.meshes.TensorMesh`,
    :class:`emg3d.fields.Field`, and :class:`emg3d.models.Model` instances and
    add arbitrary other data that were saved with :func:`save`.


    Parameters
    ----------
    fname : str
        File name including extension. Possibilities:

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
    if full_path.endswith('.npz'):

        # Load .npz into a flat dict.
        with np.load(full_path, allow_pickle=allow_pickle) as dat:
            data = {key: dat[key] for key in dat.files}

        # Un-flatten data.
        data = _dict_unflatten(data)

    elif full_path.endswith('.h5'):

        # Check if h5py is installed.
        if isinstance(h5py, str):
            raise ImportError(h5py)

        # Load data.
        with h5py.File(full_path, 'r') as h5file:
            data = _hdf5_get_from(h5file)

    elif full_path.endswith('.json'):

        with open(full_path, 'r') as f:
            data = json.load(f)

        # Move lists back to arrays and compose complex data.
        data = _dict_array_comp(data)

    else:
        ext = full_path.split('.')[-1]
        raise ValueError(f"Unknown extension '.{ext}'.")

    # De-serialize data.
    _dict_deserialize(data)

    # Check if file was (supposedly) created by emg3d.
    if verb > 0:
        print(f"Data loaded from «{full_path}»")
    try:
        version = data['_version']
        date = data['_date']
        form = data['_format']

        # Print file info.
        if verb > 0:
            print(f"[{version} (format {form}) on {date}].")

    except KeyError:
        if verb > 0:
            print("[version/format/date unknown; not created by emg3d].")

    return data


def _dict_serialize(inp, out=None, collect_classes=False):
    """Serialize emg3d-classes and other objects in inp-dict.

    Returns a serialized dictionary <out> of <inp>, where all members of
    `emg3d.io.KNOWN_CLASSES` are serialized with their respective `to_dict()`
    methods. These instances are additionally grouped together in dictionaries,
    and all other stuff is put into 'Data' if `collect_classes=True`.

    Any other (non-emg3d) object can be added too, as long as it knows how to
    serialize itself.

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

    collect_classes : bool
        If True, input data is collected in folders for the principal
        emg3d-classes (type `emg3d.io.KNOWN_CLASSES` to get a list) and
        everything else collected in a `Data`-folder. Default is False.


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

        # Take care of the following instances
        # (if we are in the root-directory they get their own category):
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
                except AttributeError as e:  # Gracefully fail.
                    print(f"* WARNING :: Could not serialize <{key}>.\n"
                          f"             {e}")
                    continue

            # If we are in the root-directory put them in their own category.
            # `collect_classes` can only be True in root-directory, as it is
            # set to False in recursion.
            if collect_classes:
                value = {key: to_dict}
                key = name
            else:
                value = to_dict

        elif collect_classes:
            # `collect_classes` can only be True in root-directory, as it is
            # set to False in recursion.
            if key.startswith('_'):  # Store meta-data in root-level...
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
            _dict_serialize(value, out[key], collect_classes=False)
        else:
            # Limitation 2: None
            if value is None:
                out[key] = 'NoneType'
            else:
                out[key] = value

    # Return if it wasn't provided.
    if output:
        return out


def _dict_deserialize(inp, first_call=True):
    """De-serialize emg3d-classes and other objects in inp-dict.

    De-serializes in-place dictionary <inp>, where all members of
    `emg3d.io.KNOWN_CLASSES` are de-serialized with their respective
    `from_dict()` methods. It also converts back `'NoneType'`-strings to
    `None`, and `np.bool_` to `bool`.


    Parameters
    ----------
    inp : dict
        Input dictionary to de-serialize.

    """

    # Recursively replace `'NoneType'` by `None` and `np.bool_` by `bool`.
    if first_call:
        _nonetype_to_none(inp)

    # Loop over items.
    for key, value in inp.items():

        # If it is a dict, deserialize if KNOWN_CLASS or recursion.
        if isinstance(value, dict):

            # If it has a __class__-key, de-serialize.
            if '__class__' in value.keys():

                # De-serialize, overwriting all the existing entries.
                try:
                    inst = KNOWN_CLASSES[value['__class__']]
                    inp[key] = inst.from_dict(value)
                    continue

                except (NotImplementedError, AttributeError, KeyError) as e:
                    # Gracefully fail.
                    print(f"* WARNING :: Could not de-serialize <{key}>.\n"
                          f"             {e}")

            # In no __class__-key or de-serialization fails, use recursion.
            _dict_deserialize(value, False)


def _nonetype_to_none(inp):
    """Recursively replace side-effects in inp-dict from storing to disc.

    Changes:

    - Replace `NoneType'` by `None`.
    - `np.bool_` are cast back to `bool` (because `bool` is converted to
      `np.bool_` for some file formats).

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

            # h5py>=3.0 changed strings to byte strings.
            if isinstance(data[key], bytes):
                data[key] = data[key].decode("utf-8")

        elif isinstance(value, h5py._hl.group.Group):
            data[key] = _hdf5_get_from(value)

    return data


def _compare_dicts(dict1, dict2, verb=False, **kwargs):
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
