# JOSS figure
# Python file to recreate the figure in the JOSS article paper.md.
import numpy as np
import matplotlib.pyplot as plt

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8

# Data is from the notebooks `4a_RAM-requirements.ipynb` and `4b_Runtime.ipynb`
# in the repo https://github.com/empymod/emg3d-examples at commit 1a1a658c23
# (2019-06-05). It was run at the TU Delft server Texel on one thread. From
# `cat /proc/cpuinfo`: Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
nsizes_mem = np.array([64, 96, 128, 192, 256, 384, 512, 768, 1024])
mem = np.array([212, 379, 678, 1821, 4035, 13194, 30928, 103691, 245368])
nsizes_cpu = np.array([32, 48, 64, 96, 128, 192, 256, 384])
cpu = np.array([1.3, 4.5, 10.6, 36.3, 90.6, 311.9, 758.7, 2727.8])


def focus_on_data(ax):
    """Reduce and hide figure elements to focus on data."""
    ax.grid(axis='y', color='grey', linestyle='-', linewidth=1, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


# Figure
fig = plt.figure(figsize=(6, 2.5))

# Runtime subplot
ax1 = plt.subplot(121)
plt.title('(a) Time', fontsize=10)
plt.loglog(nsizes_cpu**3/1e6, cpu, '.-')
plt.ylabel('Runtime (s)', fontsize=8)
plt.xticks([1e-1, 1e0, 1e1, 1e2], ('0.1', '1', '10', '100'))
plt.yticks([1e0, 1e1, 1e2, 1e3], ('1', '10', '100', '1000'))
focus_on_data(ax1)

# Memory subplot
ax2 = plt.subplot(122)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.title('(b) Memory', fontsize=10)
plt.loglog(nsizes_mem**3/1e6, mem/1e3, '.-')
plt.ylabel('RAM (GB)', fontsize=8)
plt.xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ('0.1', '1', '10', '100', '1000'))
plt.yticks([1e-1, 1e0, 1e1, 1e2], ('0.1', '1', '10', '100'))
focus_on_data(ax2)

# Combined x-label
fig.text(0.5, -0.05, 'Number of cells (in millions)', ha='center', fontsize=8)

# Save the figure
plt.savefig('paper.png', bbox_inches='tight')
