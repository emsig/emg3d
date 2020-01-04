"""
Parameter tests
===============

The modeller ``emg3d`` has quite a few parameters which can influence the speed
of a calculation. It can be difficult to estimate which is the best setting. In
the case that speed is of utmost importance, and a lot of similar models are
going to be calculated (e.g. for inversions), it might be worth to do some
input parameter testing.

**IMPORTANT:** None of the conclusions you can draw from these figures are
applicable to other models. What is faster depends on your input. Influence has
particularly the degree of anisotropy and of model stretching. These are simply
examples that you can adjust for your problem at hand.

"""
import emg3d
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


###############################################################################
def plotit(infos, labels):
    """Simple plotting routine for the tests."""

    plt.figure()

    # Loop over infos.
    for i, info in enumerate(infos):
        plt.plot(info['runtime_at_cycle'],
                 info['error_at_cycle']/info1['ref_error'],
                 '.-', label=labels[i])

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Rel. Error $(-)$')
    plt.yscale('log')

    plt.show()


###############################################################################

# Survey
zwater = 1000                  # Water depth.
src = [0, 0, 50-zwater, 0, 0]  # Source at origin, 50 m above seafloor.
freq = 1.0                     # Frequency (Hz).

# Mesh
ginp = {'min_width': 100, 'verb': 0}
xx, x0 = emg3d.utils.get_hx_h0(
    freq=freq, res=[0.3, 1.], fixed=src[0], domain=[-1000, 5000], **ginp)
yy, y0 = emg3d.utils.get_hx_h0(
    freq=freq, res=[0.3, 1.], fixed=src[1], domain=[-500, 500], **ginp)
zz, z0 = emg3d.utils.get_hx_h0(
    freq=freq, res=[0.3, 1., 0.3], domain=[-2500, 0],
    fixed=[-1000, 0, -2100], **ginp)
grid = emg3d.utils.TensorMesh([xx, yy, zz], x0=np.array([x0, y0, z0]))
print(grid)

# Source-field
sfield = emg3d.utils.get_source_field(grid, src=src, freq=freq)

# Create a simple marine model for the tests.

# Layered_background
res_x = 1e8*np.ones(grid.vnC)              # Air
res_x[:, :, grid.vectorCCz <= 0] = 0.3     # Water
res_x[:, :, grid.vectorCCz <= -1000] = 1.  # Background

# Target
xt = np.nonzero((grid.vectorCCx >= -500) & (grid.vectorCCx <= 5000))[0]
yt = np.nonzero((grid.vectorCCy >= -1000) & (grid.vectorCCy <= 1000))[0]
zt = np.nonzero((grid.vectorCCz >= -2100) & (grid.vectorCCz <= -1800))[0]
res_x[xt[0]:xt[-1]+1, yt[0]:yt[-1]+1, zt[0]:zt[-1]+1] = 100

# Create a model instance
model_iso = emg3d.utils.Model(grid, res_x)

###############################################################################
# Test 1: F, W, and V MG cycles
# -----------------------------

inp = {'grid': grid, 'model': model_iso, 'sfield': sfield,
       'verb': 1, 'return_info': True}

_, info1 = emg3d.solve(cycle='F', **inp)
_, info2 = emg3d.solve(cycle='W', **inp)
_, info3 = emg3d.solve(cycle='V', **inp)

plotit([info1, info2, info3], ['F-cycle', 'W-cycle', 'V-cycle'])

###############################################################################
# Test 2: semicoarsening, line-relaxation
# ---------------------------------------

inp = {'grid': grid, 'model': model_iso, 'sfield': sfield,
       'verb': 1, 'return_info': True}

_, info1 = emg3d.solve(**inp)
_, info2 = emg3d.solve(semicoarsening=True, **inp)
_, info3 = emg3d.solve(linerelaxation=True, **inp)
_, info4 = emg3d.solve(semicoarsening=True, linerelaxation=True, **inp)

plotit([info1, info2, info3, info4], ['MG', 'MG+SC', 'MG+LR', 'MG+SC+LR'])

###############################################################################
# Test 3: MG and BiCGstab
# -----------------------

inp = {'grid': grid, 'model': model_iso, 'sfield': sfield,
       'semicoarsening': True, 'verb': 1, 'return_info': True, 'maxit': 500}

_, info1 = emg3d.solve(cycle='F', sslsolver=False, **inp)
_, info2 = emg3d.solve(cycle='F', sslsolver=True, **inp)
_, info3 = emg3d.solve(cycle=None, sslsolver=True, **inp)

plotit([info1, info2, info3], ['MG', 'MG+BiCGStab', 'BiCGStab'])

###############################################################################
# Test 4: `nu_init`, `nu_pre`, `nu_coarse`, `nu_post`
# ---------------------------------------------------

inp = {'grid': grid, 'model': model_iso, 'sfield': sfield,
       'semicoarsening': True, 'verb': 1, 'return_info': True}

_, info1 = emg3d.solve(**inp)
_, info2 = emg3d.solve(nu_pre=0, **inp)
_, info3 = emg3d.solve(nu_post=0, **inp)
_, info4 = emg3d.solve(nu_init=2, **inp)

plotit([info1, info2, info3, info4],
       ['{0,2,1,2} (default)', '{0,0,1,2}', '{0,2,1,0}', '{2,1,2,1}'])

###############################################################################

emg3d.Report()
