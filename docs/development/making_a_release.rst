Making a release
================

.. todo::

    Rework this for version 1.

1. Update ``CHANGELOG.rst``.

2. Push it to GitHub, create a release tagging it.

3. Tagging it on GitHub will automatically deploy it to PyPi, which in turn
   will create a PR for the conda-forge `feedstock
   <https://github.com/conda-forge/emg3d-feedstock>`_. Merge that PR.

4. Check that:

  - `PyPi <https://pypi.org/project/emg3d>`_ deployed;
  - `conda-forge <https://anaconda.org/conda-forge/emg3d>`_ deployed;
  - `Zenodo <https://doi.org/10.5281/zenodo.3229006>`_ minted a DOI;
  - `emg3d.emsig.xyz <https://emg3d.emsig.xyz>`_ created a tagged version.


Useful things
-------------

- If there were changes to README, check it with::

       python setup.py --long-description | rst2html.py --no-raw > index.html

- If unsure, test it first on testpypi (requires ~/.pypirc)::

       ~/anaconda3/bin/twine upload dist/* -r testpypi

- If unsure, test the test-pypi for conda if the skeleton builds::

       conda skeleton pypi --pypi-url https://test.pypi.io/pypi/ emg3d

- If it fails, you might have to install ``python3-setuptools``::

       sudo apt install python3-setuptools


CI
--

Automatic bits
``````````````

- Testing on Github Actions includes:

  - Tests using ``pytest``
  - Linting / code style with ``pytest-flake8``
  - Ensure all http(s)-links work (``sphinx linkcheck``)

- Line-coverage with ``pytest-cov`` on `Coveralls
  <https://coveralls.io/github/emsig/emg3d>`_
- Code-quality on `Codacy
  <https://app.codacy.com/manual/prisae/emg3d/dashboard>`_
- Manual on `ReadTheDocs <https://emg3d.emsig.xyz/en/latest>`_
- DOI minting on `Zenodo <https://doi.org/10.5281/zenodo.3229006>`_

Manual things
`````````````

- Benchmarks with `Airspeed Velocity <https://emsig.xyz/emg3d-asv>`_
  (``asv``)
- Gallery in `emg3d-gallery <https://emsig.xyz/emg3d-gallery>`_
  (``sphinx-gallery``)

Automatically deploys if tagged
```````````````````````````````

- `PyPi <https://pypi.org/project/emg3d>`_
- `conda -c conda-forge <https://anaconda.org/conda-forge/emg3d>`_
