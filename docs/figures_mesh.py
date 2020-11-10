import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt

# Use xkcd-style figures.
plt.xkcd()

# Some settings
fs = 14

# # # (A) Figure with survey and computational domains, buffer.           # # #
fig, ax = plt.subplots(1, 1, figsize=(11, 7))

# Plot domains.
dinp1 = {'fc': 'none', 'zorder': 2}
dc = patches.Rectangle((0, 0), 100, 60, color='.9')
dcf = patches.Rectangle((0, 0), 100, 60, ec='C0', **dinp1)
ds = patches.Rectangle((15, 10), 70, 40, fc='w')
dsf = patches.Rectangle((15, 10), 70, 40, ec='C1', **dinp1)
for d in [dc, dcf, ds, dsf]:
    ax.add_patch(d)
dinp2 = {'verticalalignment': 'center', 'zorder': 2}
ax.text(60, 60, r'Computational domain $D_c$', c='C0', **dinp2)
ax.text(60, 50, r'Survey domain $D_s$', c='C1', **dinp2)

# plot seasurface, seafloor, receivers, source.
x = np.arange(101)
y1 = np.sin(x/np.pi)-np.arange(x.size)/10+43
ax.plot(x, y1, '.8', zorder=1)
ax.plot(x[15:-15], y1[15:-15], '.4', zorder=1)
si = [30, 40, 50, 60, 70]
for i in si:
    ax.plot(x[i], y1[i]+4, 'C3*')
ri = np.arange(4, 16)*5
for i in ri:
    ax.plot(x[i+2], y1[i+2]+1, 'C0v')

# Subsurface.
y2 = np.sin(x/5)-((np.arange(x.size)-x.size/2)/20)**2+13+np.arange(x.size)/10
y3 = np.sin(x/10)-((np.arange(x.size)-x.size/2)/40)**3+28-np.arange(x.size)/300
y4 = np.min([y3, np.arange(101)/10+22], axis=0)
subinp1 = {'c': '0.8', 'zorder': 1}
subinp2 = {'c': '0.4', 'zorder': 1}
ax.plot(x, y2, **subinp1)
ax.plot(x, y4, **subinp1)
ax.plot(x, y3, **subinp1)
ax.plot(x[15:-15], y2[15:-15], **subinp2)
ax.plot(x[15:-15], y4[15:-15], **subinp2)
ax.plot(x[15:-15], y3[15:-15], **subinp2)

# Lambdas.
aprops = {'head_width': 2, 'head_length': 3,
          'length_includes_head': True, 'color': 'C2'}
tprops = {'fontsize': fs, 'c': 'C2', 'verticalalignment': 'center'}
ax.arrow(50, 10, 0, -10, **aprops)
ax.text(51, 5, r'$\lambda(f, \sigma_{z-})$', **tprops)
ax.arrow(50, 50, 0, 10, **aprops, zorder=10)
ax.text(51, 55, r'$\lambda(f, \sigma_{z+})$', **tprops)
ax.arrow(15, 30, -15, 0, **aprops, zorder=10)
ax.text(3, 32, r'$\lambda(f, \sigma_{x-})$', **tprops)
ax.arrow(85, 30, 15, 0, **aprops, zorder=10)
ax.text(88, 32, r'$\lambda(f, \sigma_{x+})$', **tprops)
ax.text(5, 5, 'Buffer', c='.5', fontsize=16)

# Axis
ax.arrow(0, 0, 10, 0, head_width=2, head_length=3, fc='k', ec='k', zorder=10)
ax.arrow(0, 0, 0, 10, head_width=2, head_length=3, fc='k', ec='k', zorder=10)
ax.text(15, 0, r'$x$', fontsize=fs, verticalalignment='center', zorder=10)
ax.text(0, 15, r'$z$', fontsize=fs, horizontalalignment='center', zorder=10)

ax.set_axis_off()
ax.set_xlim([-5, 105])
ax.set_ylim([-5, 65])

fig.savefig('_static/construct_mesh.png', bbox_inches='tight', pad_inches=0)
fig.show()

# # # (B) Figure with survey and computational domains, buffer.           # # #
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

# Plot domains
ax.plot([35, 35], [16, 30], 'C0')
ax.plot([30, 30], [0, 14], 'C0')
ax.plot([20, 20], [0, 14], 'C1')
ax.plot([20, 20], [16, 30], 'C1')

# Plot center
ax.plot([5, 5], [6, 23], 'C3*', zorder=11)

# lambdas
aprops2 = {'head_width': 2, 'head_length': 3, 'zorder': 10,
           'length_includes_head': True, 'color': 'C2'}

ax.arrow(20, 22, 15, 1, **aprops2)
ax.arrow(35, 24, -15, 1, **aprops2)

ax.arrow(5, 6, 25, 1, **aprops2)
ax.arrow(30, 8, -10, 1, **aprops2)

ax.text(20, 31, r'$D_s$', c='C1', horizontalalignment='center', fontsize=fs)
ax.text(20, -3, r'$D_s$', c='C1', horizontalalignment='center', fontsize=fs)
ax.text(35, 31, r'$D_c$', c='C0', horizontalalignment='center', fontsize=fs)
ax.text(30, -3, r'$D_c$', c='C0', horizontalalignment='center', fontsize=fs)
ax.text(5, 31, r'$center$', c='C3', horizontalalignment='center', fontsize=fs)
ax.text(5, -3, r'$center$', c='C3', horizontalalignment='center', fontsize=fs)

ax.text(25, 15, r'$B$', verticalalignment='center')
ax.text(40, 15, r"$\lambda'=\lambda_{fact}\ \lambda(f, \sigma)$",
        verticalalignment='center')
aprops3 = {**aprops2, 'color': 'k'}
ax.arrow(29, 17, -9, 0, **aprops3)
ax.arrow(26, 17, 9, 0, **aprops3)
ax.arrow(27, 13, -7, 0, **aprops3)
ax.arrow(23, 13, 7, 0, **aprops3)

ax.text(0, 25, '(I) False')
ax.text(40, 23, r"$B=\lambda' \leq B_{max}$")
ax.text(0, 9, '(II) True')
ax.text(37, 5, r"$B=(2\lambda'-|D_s-center|)/2$")
ax.text(39, 1, r'$\leq B_{max}-|D_s-center|$')

ax.set_axis_off()
ax.set_xlim([-5, 75])
ax.set_ylim([-10, 40])

fig.savefig('_static/construct_mesh2.png', bbox_inches='tight', pad_inches=0)
fig.show()
