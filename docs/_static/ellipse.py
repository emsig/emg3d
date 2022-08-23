import numpy as np
import matplotlib.pyplot as plt

# Note: This is not the full script, and just for explanatory purposes.
#       See emg3d.maps.ellipse_indices for more details.

p0 = np.array([1500, 1400])     # Point 1
p1 = np.array([5500, 800])      # Point 2
radius = 1000                   # Radius
cx, cy = (p0 + p1) / 2          # Center coordinates
dx, dy = (p1 - p0) / 2          # Adjacent and opposite sides
dxy = np.linalg.norm([dx, dy])  # c: linear eccentricity
major = dxy + radius            # a: semi-major axis
minor = 0.8*major               # b: semi-minor axis
cos, sin = dx/dxy, dy/dxy       # Angles

# Parametrized ellipse
t = np.linspace(0, 2*np.pi, 101)
cost, sint = np.cos(t), np.sin(t)
x = cx + major*cos*cost - minor*sin*sint
y = cy + major*sin*cost + minor*cos*sint

# Vertices
vr = [cx + major*cos, cy + major*sin]
vl = [cx - major*cos, cy - major*sin]
vu = [cx - minor*sin, cy + minor*cos]

fig, ax = plt.subplots(constrained_layout=True)

# Draw a, b, c, and r
ax.plot([cx, vr[0]], [cy, vr[1]], 'C4')
ax.annotate(r'$a=\max(f c, c + r)$', (cx+500, cy-550), c='C4', rotation=-9)
ax.plot([cx, vu[0]], [cy, vu[1]], 'C3')
ax.annotate(r'$b=\max(m a, r)$', (cx+150, cy+200), c='C3', rotation=81)
ax.plot([p0[0], cx], [p0[1], cy], 'C0')
ax.annotate(r'$c=||p_1-p_0||/2$', (p0[0]+600, p0[1]-480), c='C0', rotation=-9)
ax.plot([vl[0], p0[0]], [vl[1], p0[1]], 'C5')
ax.annotate(r'$r$', (vl[0]+300, vl[1]-250), c='C5', rotation=-9)

# Draw the two points and their circles with radius
for p in [p0, p1]:
    ax.plot(*p, 'o', c='k', ms=8)
    ax.plot(p[0] + radius*cost, p[1] + radius*sint, '.7')

ax.plot(x, y, 'C1', lw=2)  # Draw the ellipse
ax.axis('equal')     # Ensure square axes
ax.axis('off')
fig.savefig('./ellipse.svg', bbox_inches='tight')
plt.show()
