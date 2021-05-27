.. _io-persistence:

I/O & Persistence
=================


Saving and loading
------------------

emg3d has functions to store data to disk and to read data from disk, in
different file formats. Currently three file formats are supported, each
with its own advantages and disadvantages:

- ``.h5``:
  Uses h5py to store inputs to a hierarchical, compressed binary HDF5 file.
  **Recommended file format.**

  - Advantage: Widely used, compressed file format, which can be read and
    written in many programs.
  - Disadvantage: You have to install ``h5py``.


- ``.npz``:
  Uses numpy to store inputs to a flat, compressed binary file.

  - Advantage: No extra installation is required, and the outputs are
    compressed.
  - Disadvantage: Only useful within the Python ecosystem.


- ``.json``:
  Uses json to store inputs to a hierarchical, plain text file.

  - Advantage: No extra installation is required, and the output is a plain
    text file that can be viewed in any editor; good for developing and
    debugging.
  - Disadvantage: Not compressed (files can become huge).


Example
~~~~~~~

You should be able to save and load everything you do in emg3d with these
functions. Please have a look at the API of :func:`emg3d.io.save` and
:func:`emg3d.io.load`. But in a nutshell, the first argument is a string
containing the relative or absolute path, file name, and the appropriate suffix
indicating the file format. Afterwards it is simply a ``name=value`` list,
where the name can be anything, and the value must be an existing variable.
(There are a few more options, see the API.)

.. ipython::
  :verbatim:

  In [1]: emg3d.save(
     ...:     '/path/to/filename.ending',
     ...:     inp_model=model1,
     ...:     out_model=model2,
     ...:     survey=survey,
     ...:     efield=efield,
     ...: )
  Out[1]: Data saved to «/path/to/filename.ending»

When you load such a file it will give you a dictionary containing as keys the
names you have defined:

.. ipython::
  :verbatim:

  In [1]: data = emg3d.load('/path/to/filename.ending')
  Out[1]: Data loaded from «/path/to/filename.ending»

  In [2]: data.keys()
  Out[2]: dict_keys(['_date', '_format', '_version', 'efield', 'inp_model', 'out_model', 'survey'])

In addition to the variables you have defined there are a few other, "private"
(starting with an underscore) variables such as the date, format, and version
of emg3d with which the archive was created.


``{to;from}_file``
~~~~~~~~~~~~~~~~~~

The two classes :class:`emg3d.surveys.Survey` and
:class:`emg3d.simulations.Simulation` have ``to_file`` and ``from_file``
methods, which are basically wrappers around the saving and loading functions.
They can be used in the following way:

Storing to disk

.. ipython::
  :verbatim:

  In [1]: my_survey.to_file('mydata.h5')


and loading from disk

.. ipython::
  :verbatim:

  In [1]: my_survey = emg3d.Survey.from_file('mydata.h5')




Serialization
-------------

The following are advanced information if you want to read data created with
emg3d outside of Python or if you want to create data outside of Python which
you can read subsequently with emg3d. As a pure end-user of emg3d you can
ignore this section.

Here a few info with regards to the (de-)serialization used in emg3d.

- When invoking ``emg3d.save('filename.ending', a=a, b=something, foo=bar)``,
  the data is collected in a dict ``{'a': a, 'b': something, 'foo': bar}``.
- Afterwards the dict is serialized. Instances of emg3d
  (:class:`emg3d.meshes.TensorMesh`, :class:`emg3d.fields.Field`,
  :class:`emg3d.surveys.Survey`, :class:`emg3d.simulations.Simulation`) have
  ``to_dict`` and ``from_dict`` methods to (de-)serialize themselves. These are
  used when saving and loading them. In principal emg3d can save everything
  that is either serialized already or is present in
  ``emg3d.utils._KNOWN_CLASSES``. You can define your own classes which have
  ``{to;from}_dict`` methods, and add them to the known classes with the
  decorator ``@utils._known_class``.

  - Things which are done when serializing and undone when de-serializing:

    - ``None`` is saved as a string ``'NoneType'``.

  - Things done when serializing:

    - Dictionary key names are converted to strings
    - Grids generated with discretize are stored as if they were created using
      emg3d.

  - Things done when de-serializing:

    - ``np.bool_`` is returned as ``bool``.




These first two points are always carried out. After this it depends on the
file format, as different file formats have different limitations.



- ``.h5``:
  Each nesting level creates a new data set.


- ``.npz``:
  The serialized dict is converted into a flattened dict, where the keys are
  separated with ``'>'``.

- ``.json``:

  - NumPy-arrays are turned into lists, where ``'__array-'`` plus the ``dtype``
    are added to the key.
  - Complex numbers are stacked, real values followed by imaginary values;
    ``__complex`` is added to the key.
