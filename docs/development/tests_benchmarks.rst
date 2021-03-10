Tests and Benchmarks
====================

The modeller comes with a test suite using ``pytest``. If you want to run the
tests, just install ``pytest`` and run it within the ``emg3d``-top-directory.

.. code-block:: console

    > pytest --cov=emg3d --flake8

It should run all tests successfully. Please let us know if not!

Note that installations of ``em3gd`` via conda or pip do not have the
test-suite included. To run the test-suite you must download ``emg3d`` from
GitHub.

There is also a benchmark suite using *airspeed velocity*, located in the
`emsig/emg3d-asv <https://github.com/emsig/emg3d-asv>`_-repository. The results
of my machine can be found in the `emsig/emg3d-bench
<https://github.com/emsig/emg3d-bench>`_, its rendered version at
`emsig.github.io/emg3d-asv <https://emsig.github.io/emg3d-asv>`_.
