---
title: "emg3d: A multigrid solver for 3D electromagnetic diffusion"
tags:
  - python
  - geophysics
  - CSEM
  - multigrid
  - modelling
  - electromagnetic
authors:
 - name: Dieter Werthmüller
   orcid: 0000-0002-8575-2484
   affiliation: 1
 - name: Wim A. Mulder
   orcid: 0000-0001-7020-9297
   affiliation: "1, 2"
 - name: Evert C. Slob
   orcid: 0000-0002-4529-1134
   affiliation: 1
affiliations:
 - name: TU Delft, Building 23, Stevinweg 1, 5048 CN Delft, NL
   index: 1
 - name: Shell Global Solutions International BV, Grasweg 31, 1031 HW Amsterdam, NL
   index: 2
date: 03 May 2019
bibliography: paper.bib
---

# Summary

Controlled-source electromagnetic (CSEM) surveys are a common geophysical
investigation tool in the search for, amongst other, groundwater, hydrocarbons,
and minerals. The numerical modelling of CSEM data requires the solution of the
Maxwell equations. These can be simplified in the particular case of CSEM, as
the frequencies used in surveys are usually sufficiently low to ignore any
displacement currents. A diffusive problem remains, which resulting system of
equations is given in the frequency domain by

$$ \eta \mathbf{E} - \nabla \times \mu_r^{-1} \nabla \times \mathbf{E} =
    -\mathrm{i}\omega\mu_0\mathbf{J}_s \, ,$$

where $\eta = \mathrm{i}\omega \mu_0(\sigma - \mathrm{i}\omega\varepsilon)$.

Various open-source codes exist to model CSEM responses for a layered Earth,
e.g., *DIPOLE1D* [@Key:2009] and *empymod* [@Werthmuller:2017], and for a
two-dimensional Earth, e.g., *MARE2DEM* [@Key:2011]. Open-source modellers
for a three-dimensional (3D) Earth only recently became available, notably
*SimPEG* [@Cockett:2015]; *PETGEM* [@Castillo:2018]; and *custEM*
[@Rochlitz:2019]. SimPEG is a framework that not only includes CSEM but also
other geophysical methods and can model them on various types of regular grids.
It currently uses mostly the direct solver *PARDISO*. PETGEM and custEM use
finite elements with the *FEniCS* solver. All three codes require substantial
memory and are not easily run on a laptop for models with several millions of
cells.

Mulder [@Mulder:2006] has shown that the multigrid method [@Briggs:2000], which
reduces memory requirement substantially, works fine for diffusive CSEM
problems. This was later also confirmed by others [@Jaysaval:2016]. However, the
multigrid CSEM codes of these publications are proprietary. The code *emg3d* is
a multigrid solver for 3D CSEM diffusion with tri-axial electrical anisotropy
using a staggered grid [@Mulder:2006]; [@Mulder:2007].
It can act as a solver on its own, or be used as a preconditioner for various
Krylov subspace methods included in the scientific Python library (SciPy).
Multigrid solvers can struggle to converge with strong grid-stretching or
strong anisotropy. We implemented *semicoarsening* (coarsening the grid only in
some coordinate directions) and *line relaxation* (solving for the values of a
whole gridline simultaneously in some directions) inside the multigrid
technique to deal with these issues [@Jonsthovel:2006].

The code is written completely in Python using the NumPy/SciPy-stack, where the
most time- and memory-consuming parts are sped up through jitted
numba-functions. It can currently be used as a stand-alone modeller or as a
solver in conjunction with the SimPEG-framework.


# Acknowledgements
This code was developed within the Gitaro.JIM project funded through MarTERA as
part of Horizon 2020, a funding scheme of the European Research Area (ERA-NET
Cofund, https://www.martera.eu).

# References
