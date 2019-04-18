Credits
#######

This project was started by **Dieter Werthmüller**. Every contributor will be
listed here and is considered to be part of *The emg3d Developers*:

- `Dieter Werthmüller <https://github.com/prisae>`_


History and authors of the original code
----------------------------------------

The core of *emg3d* is a complete rewrite and redesign of the multigrid code by
**Wim A. Mulder** ([Muld06]_, [Muld07]_, [Muld08]_, [MuWS08]_), developed at
Shell and at TU Delft. Various authors contributed to the original code,
amongst others, **Tom Jönsthövel** ([JoOM06]_; improved solver for strongly
stretched grids), **Marwan Wirianto** ([WiMS10]_, [WiMS11]_; computation of
time-domain data), and **Evert C. Slob** ([SlHM10]_; analytical solutions). The
original code was written in Matlab, where the most time- and memory-consuming
parts were sped up through mex-files (written in C). It included multigrid with
or without BiCGSTAB, VTI resistivity, semi coarsening, and linerelaxation; the
number of cells had to be powers of two, and coarsening was done only until the
first dimension was at two cells. As such it corresponds more or less to *emg3d
v0.3.0* (with VTI instead of tri-axial resistivity).

See the :doc:`references` in the manual for the full citations and a more
extensive list.


.. note::

    This software was initially (till 05/2021) developed at *Delft University
    of Technology* (https://www.tudelft.nl) within the **Gitaro.JIM** project
    funded through MarTERA as part of Horizon 2020, a funding scheme of the
    European Research Area (ERA-NET Cofund, https://www.martera.eu).
