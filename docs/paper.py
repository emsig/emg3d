# JOSS figure
import numpy as np
import matplotlib.pyplot as plt

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8

# Data is from the notebooks `4a_RAM-requirements.ipynb` and `4b_Runtime.ipynb`
# in the repo https://github.com/empymod/emg3d-examples.
nsizes_mem = np.array([64, 96, 128, 192, 256, 384, 512, 768, 1024])
mem = np.array([212, 379, 678, 1821, 4035, 13194, 30928, 103691, 245368])
nsizes_cpu = np.array([32, 48, 64, 96, 128, 192, 256, 384])
cpu = np.array([1.3, 4.5, 10.6, 36.3, 90.6, 311.9, 758.7, 2727.8])

# Figure
plt.figure(figsize=(8, 3))

ax1 = plt.subplot(121)
plt.title('Time')
plt.loglog(nsizes_cpu**3/1e6, cpu, '.-', label='runtime')
plt.xlabel('Number of cells (in millions)')
plt.ylabel('Runtime (s)')
plt.xticks([1e-1, 1e0, 1e1, 1e2], ('0.1', '1', '10', '100'))
plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4], ('1', '10', '100', '1000', '10000'))

# Show horizontal gridlines, switch off frame
ax1.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax2 = plt.subplot(122)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.title('Memory')
plt.loglog(nsizes_mem**3/1e6, mem/1e3, '.-', zorder=10, label='MG full RAM')
plt.xlabel('Number of cells (in millions)')
plt.ylabel('RAM (GB)')
plt.xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ('0.1', '1', '10', '100', '1000'))
plt.yticks([1e-1, 1e0, 1e1, 1e2], ('0.1', '1', '10', '100'))

# Show horizontal gridlines, switch off frame
ax2.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.savefig('paper.png', bbox_inches='tight')
plt.show()
