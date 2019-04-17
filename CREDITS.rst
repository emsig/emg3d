Credits
#######

- **Wim A. Mulder**
- **Evert C. Slob**
- **Marwan Wirianto**
- **Tom Jönsthövel**

History and authors of the original code
----------------------------------------

The core of `emg3d` is a complete rewrite of the multigrid code by Wim Mulder
([Muld06]_, [Muld07]_, [Muld08]_, [MuWS08]_), developed at Shell and at TU
Delft. Various authors contributed to the original code, amongst others, Tom
Jönsthövel ([JoOM06]_; improved solver for strongly stretched grids) and Marwan
Wirianto ([WiMS10]_, [WiMS11]_; computation of time-domain data). The original
code was written in Matlab, where the most time- and memory-consuming parts
were sped up through mex-files (written in C).

See the :doc:`references` in the manual for the full citations and a more
extensive list.


.. note::

    This software was initially (till 05/2021) developed at *Delft University
    of Technology* (https://www.tudelft.nl) within the **Gitaro.JIM** project
    funded through MarTERA as part of Horizon 2020, a funding scheme of the
    European Research Area (ERA-NET Cofund, https://www.martera.eu).
