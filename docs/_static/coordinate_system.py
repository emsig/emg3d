import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from emg3d.electrodes import rotation


class Arrow3D(FancyArrowPatch):
    """https://stackoverflow.com/a/29188796"""

    def __init__(self, xs, ys, zs):
        FancyArrowPatch.__init__(
                self, (0, 0), (0, 0), mutation_scale=20, lw=1.5,
                arrowstyle='-|>', color='.2', zorder=10)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# Create figure
fig = plt.figure(figsize=(8, 5))

# Right-handed  system
ax = fig.add_subplot(111, projection='3d', facecolor='w')
ax.axis('off')

# Point centered around (0, 0, 0);
# distances in positive directions.
P = (6, 8, 5)

# Coordinate system
# The first three are not visible, but for the aspect ratio of the plot.
ax.plot([-2, 8], [0, 0], [0, 0], c='w')
ax.plot([0, 0], [-2, 12], [0, 0], c='w')
ax.plot([0, 0], [0, 0], [-1.5, 6], c='w')
ax.add_artist(Arrow3D([-2, 10], [0, 0], [0, 0]))
ax.add_artist(Arrow3D([0, 0], [-2, 12], [0, 0]))
ax.add_artist(Arrow3D([0, 0], [0, 0], [-2, 5]))

fact = 5

# Theta
azm = np.arcsin(P[1]/np.sqrt(P[0]**2+P[1]**2))
lazm = np.linspace(0, azm, 31)
rot1 = rotation(lazm, lazm*0, deg=False)
ax.plot(*(rot1*fact), lw=2, c='C2', solid_capstyle='round')
ax.text(5.2, 1.5, 0, r"$\theta$", color='C2', fontsize=14)

# Theta and Phi
elevation = np.pi/2-np.arcsin(
        np.sqrt(P[0]**2+P[1]**2)/np.sqrt(P[0]**2+P[1]**2+P[2]**2))
# print(f"theta = {np.rad2deg(azm):.1f}°; phi = {np.rad2deg(elevation):.1f}°")
lelevation = np.linspace(0, elevation, 31)

rot2 = rotation(azm, lelevation, deg=False)
ax.plot(*(rot2*fact), c='C0', lw=2, zorder=11, solid_capstyle='round')
ax.text(3.3, 3.5, 1.5, r"$\varphi$", color='C0', fontsize=14)

# Helper lines
ax.plot([0, P[0]], [P[1], P[1]], [0, 0], ':', c='.8')
ax.plot([P[0], P[0]], [0, P[1]], [0, 0], ':', c='.8')
ax.plot([P[0], P[0]], [P[1], P[1]], [0, P[2]], '--', c='.6')
ax.plot([0, P[0]], [0, P[1]], [0, 0], '--', c='.6')

# Annotate it
ax.text(11, -3.5, 0, r'$x$ / Easting')
ax.text(0.2, 11, 0.5, r'$y$ / Northing')
ax.text(0, 0, 5.2, r'$z$ / Elevation')

# Resulting trajectory
ax.plot([0, P[0]], [0, P[1]], [0, P[2]], 'C3', lw=2, zorder=12,
        solid_capstyle='round')

ax.view_init(azim=-50, elev=10)
ax.dist = 6
plt.title('RHS coordinate system', y=1)
plt.tight_layout()
plt.savefig('./coordinate_system.svg', bbox_inches='tight', pad_inches=0.2)
plt.show()
