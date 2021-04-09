Maintenance
===========

Quick overview / QC
-------------------

- .. image:: https://github.com/emsig/emg3d/actions/workflows/linux.yml/badge.svg
     :target: https://github.com/emsig/emg3d/actions/workflows/linux.yml
     :alt: GitHub Actions linux
  .. image:: https://github.com/emsig/emg3d/actions/workflows/macos_windows.yml/badge.svg
     :target: https://github.com/emsig/emg3d/actions/workflows/macos_windows.yml
     :alt: GitHub Actions macos & windows
  .. image:: https://github.com/emsig/emg3d/actions/workflows/linkcheck.yml/badge.svg
     :target: https://github.com/emsig/emg3d/actions/workflows/linkcheck.yml
     :alt: GitHub Actions linkcheck
  .. image:: https://readthedocs.org/projects/emg3d/badge/?version=latest
     :target: https://emg3d.emsig.xyz/en/latest
     :alt: Documentation Status

  Ensure CI and docs are passing.

- .. image:: https://img.shields.io/pypi/v/emg3d.svg
     :target: https://pypi.python.org/pypi/emg3d
     :alt: PyPI
  .. image:: https://img.shields.io/conda/v/conda-forge/emg3d.svg
     :target: https://anaconda.org/conda-forge/emg3d
     :alt: conda-forge

  Ensure latest version is deployed on PyPI and conda.

- .. image:: https://coveralls.io/repos/github/emsig/emg3d/badge.svg?branch=master
     :target: https://coveralls.io/github/emsig/emg3d?branch=master
     :alt: Coveralls
  .. image:: https://app.codacy.com/project/badge/Grade/0412e617e8cd42fea05303fe490b09b5
     :target: https://www.codacy.com/gh/emsig/emg3d/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=emsig/emg3d&amp;utm_campaign=Badge_Grade
     :alt: Codacy

  Check CI coverage and code quality is good.

- .. image:: https://img.shields.io/badge/benchmark-asv-blue.svg?style=flat
     :target: https://emsig.xyz/emg3d-asv
     :alt: Airspeed Velocity

  Check Benchmarks are run up to the latest version.

- .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3229006.svg
     :target: https://doi.org/10.5281/zenodo.3229006
     :alt: Zenodo DOI

  Check Zenodo is linking to the latest release.


Info from ReadTheDocs
---------------------

.. ipython::

    In [1]: import emg3d
       ...: emg3d.Report(
       ...:     ['sphinx', 'numpydoc', 'ipykernel', 'sphinx_numfig',
       ...:      'sphinx_automodapi', 'pydata_sphinx_theme']
       ...: )
