Credits
#######

This project was started by **Dieter Werthmüller** at TUD as part of the
*Gitaro.JIM* project (see below).
Dieter would like to thank his past and current employers who allowed and allow
him to maintain and further develop the code after the initial funding ended,
namely:

- 2018-2021: `Delft University of Technology <https://www.tudelft.nl>`_;
  through the *Gitaro.JIM* project (till 05/2021, emg3d v1.0.0), funded by
  `MarTERA <https://www.martera.eu>`_ as part of Horizon 2020, a funding scheme
  of the European Research Area.
- 2021-2024: `Delft University of Technology <https://www.tudelft.nl>`_ through
  the `Delphi Consortium <https://www.delphi-consortium.com>`_.
- 2021-2022: `TERRASYS Geophysics GmbH & Co. KG
  <https://www.terrasysgeo.com>`_.
- 2024-today: `ETH Zurich <https://ethz.ch>`_ through the group `Geothermal
  Energy and Geofluids <https://geg.ethz.ch>`_.

For a list of code contributors see
https://github.com/emsig/emg3d/graphs/contributors.

There are various contributors who improved emg3d not through code commits but
through discussions and help on Slack at
`SWUNG <https://softwareunderground.org>`_ and at
`SimPEG <https://simpeg.xyz>`_,
thanks to both communities. Special thanks to
`@jokva <https://github.com/jokva>`_ (general),
`@banesullivan <https://github.com/banesullivan>`_ (visualization),
`@joferkington <https://github.com/joferkington>`_ (interpolation),
`@jcapriot <https://github.com/jcapriot>`_ (volume averaging), and
`@sgkang <https://github.com/sgkang>`_ (inversion).


Historical credits
------------------

The core of *emg3d* is a complete rewrite and redesign of the multigrid code by
**Wim A. Mulder** ([Muld06]_, [Muld07]_, [Muld08]_, [MuWS08]_), developed at
Shell and at TU Delft. Various authors contributed to the original code,
amongst others, **Tom Jönsthövel** ([JoOM06]_; improved solver for strongly
stretched grids), **Marwan Wirianto** ([WiMS10]_, [WiMS11]_; computation of
time-domain data), and **Evert C. Slob** ([SlHM10]_; analytical solutions). The
original code was written in Matlab, where the most time- and memory-consuming
parts were sped up through mex-files (written in C). It included multigrid with
or without BiCGSTAB, VTI resistivity, semicoarsening, and line relaxation; the
number of cells had to be powers of two, and coarsening was done only until the
first dimension was at two cells. As such it corresponded roughly to ``emg3d
v0.3.0``.

See the *References* section in the manual for the full citations and a more
extensive list.
