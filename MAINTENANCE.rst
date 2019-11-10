Maintainers Guide
=================

Releases of ``emg3d`` are currently done manually. This is the 'recipe'.


Making a release
----------------

1. Update ``CHANGELOG.rst``.

2. Push it to GitHub, create a release tagging it.

3. Tagging it on GitHub will automatically deploy it to PyPi, which in turn
   will create a PR for the conda-forge `feedstock
   <https://github.com/conda-forge/emg3d-feedstock>`_. Merge that PR.

4. Release notes edits: (1) get and add the `Zenodo-DOI
   <https://doi.org/10.5281/zenodo.3229006>`_; (b) add the readthedocs badge,
   you might have to trigger a build first.


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

- Testing on `Travis <https://travis-ci.org/empymod/emg3d>`_, includes:

  - Tests using ``pytest``
  - Linting / code style with ``pytest-flake8``
  - Ensure all http(s)-links work (``sphinx linkcheck``)

- Line-coverage with ``pytest-cov`` on `Coveralls
  <https://coveralls.io/github/empymod/emg3d>`_
- Code-quality on `Codacy
  <https://app.codacy.com/manual/prisae/emg3d/dashboard>`_
- Manual on `ReadTheDocs <https://emg3d.readthedocs.io/en/latest>`_
- DOI minting on `Zenodo <https://doi.org/10.5281/zenodo.3229006>`_
- Benchmarks with `Airspeed Velocity <https://empymod.github.io/emg3d-asv>`_
  (``asv``) [currently manually]
- Examples in `emg3d-examples <https://github.com/empymod/emg3d-examples>`_
  [currently manually] => should move to a sphinx-gallery instance (`#45
  <https://github.com/empymod/emg3d/issues/45>`_)
- Automatically deploys if tagged:

  - `PyPi <https://pypi.org/project/emg3d>`_
  - `conda -c conda-forge <https://anaconda.org/conda-forge/emg3d>`_
