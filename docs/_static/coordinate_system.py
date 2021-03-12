import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    """https://stackoverflow.com/a/29188796"""

    def __init__(self, xs, ys, zs):
        FancyArrowPatch.__init__(
                self, (0, 0), (0, 0), mutation_scale=20, lw=1.5,
                arrowstyle='-|>', color='.2', zorder=100)
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
P = (6, 10, -4)

# Coordinate system
# The first three are not visible, but for the aspect ratio of the plot.
ax.plot([-2, 12], [0, 0], [0, 0], c='w')
ax.plot([0, 0], [-2, 12], [0, 0], c='w')
ax.plot([0, 0], [0, 0], [-7, 7], c='w')
ax.add_artist(Arrow3D([-2, 12], [0, 0], [0, 0]))
ax.add_artist(Arrow3D([0, 0], [-2, 16], [0, 0]))
ax.add_artist(Arrow3D([0, 0], [0, 0], [-5, 4]))
ax.plot([0, P[0]], [P[1], P[1]], [P[2], P[2]], ':', c='.8')
ax.plot([0, P[0]], [0, 0], [P[2], P[2]], ':', c='.8')
ax.plot([P[0], P[0]], [0, P[1]], [P[2], P[2]], ':', c='.8')
ax.plot([0, 0], [0, P[1]], [P[2], P[2]], ':', c='.8')
ax.plot([P[0], P[0]], [P[1], P[1]], [0, P[2]], '--', c='.6')

# Annotate it
ax.text(9, 3, 0, r'$x$ / Easting')
ax.text(0, 10, 1, r'$y$ / Northing')
ax.text(-1, 0, 4.5, r'$z$ / Elevation')

# Helper lines
ax.plot([0, P[0]], [0, P[1]], [0, 0], '--', c='.6')

fact = 7

# Theta
azm = np.arcsin(P[1]/np.sqrt(P[0]**2+P[1]**2))
lazm = np.linspace(0, azm, 31)
ax.plot(np.cos(lazm)*fact, np.sin(lazm)*fact, 0, c='C2')
ax.text(6.8, 2.5, 0, r"$\theta$", color='C2', fontsize=14)

# Phi
dip = np.pi/2-np.arcsin(
        np.sqrt(P[0]**2+P[1]**2)/np.sqrt(P[0]**2+P[1]**2+P[2]**2))
# print(f"theta = {np.rad2deg(azm):.1f}°; phi = {np.rad2deg(dip):.1f}°")
ldip = np.linspace(0, dip, 31)
ax.plot(np.cos(azm)*np.cos(ldip)*fact,
        np.sin(azm)*np.cos(ldip)*fact, -np.sin(ldip)*fact, c='C0')
ax.text(4.5, 4, -1.5, r"$\varphi$", color='C0', fontsize=14)

# Resulting trajectory
ax.plot([-P[0], P[0]], [-P[1], P[1]], [-P[2], P[2]], 'C3', lw=2)

ax.view_init(azim=-70, elev=10)
ax.dist = 6
plt.title('RHS coordinate system', y=0.9)
plt.tight_layout()
plt.savefig('./coordinate_system.png', bbox_inches='tight', pad_inches=0)
plt.show()
