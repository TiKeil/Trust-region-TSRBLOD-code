import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import tikzplotlib
# mpl.rcParams['figure.figsize'] = (8.0, 6.0)
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['savefig.dpi'] = 300
# mpl.rcParams['figure.subplot.bottom'] = .1 
N = 20

dims = np.array([
9, 9, 8, 9, 12, 9, 9, 6, 7, 9, 9, 9, 9, 9, 10, 12, 9, 9, 8, 8, 9, 9, 9, 9, 12, 10, 9, 6, 8, 9, 9, 9, 9, 9, 12, 12, 9, 9, 8, 6, 9, 9, 9, 9, 12, 11, 9, 9, 9, 9, 9, 9, 9, 9, 12, 12, 9, 9, 8, 7, 9, 9, 9, 9, 12, 12, 9, 9, 9, 12, 11, 12, 12, 12, 12, 12, 12, 9, 9, 10, 9, 9, 9, 12, 15, 15, 11, 11, 11, 13, 13, 12, 12, 12, 15, 15, 15, 12, 11, 12, 9, 9, 9, 12, 15, 15, 12, 12, 12, 15, 12, 9, 9, 12, 13, 15, 15, 12, 12, 13, 9, 10, 9, 9, 13, 12, 11, 9, 10, 12, 9, 9, 8, 8, 12, 14, 12, 10, 10, 10, 9, 9, 9, 11, 12, 11, 9, 6, 8, 9, 6, 6, 6, 7, 10, 12, 9, 9, 9, 9, 10, 10, 8, 9, 13, 12, 9, 6, 8, 9, 6, 6, 4, 7, 13, 12, 11, 10, 10, 9, 9, 10, 10, 12, 15, 15, 11, 9, 9, 12, 12, 9, 9, 12, 14, 15, 14, 13, 12, 11, 11, 12, 12, 15, 15, 15, 10, 9, 9, 12, 12, 9, 9, 11, 15, 15, 15, 13, 12, 12, 9, 9, 9, 11, 13, 12, 6, 6, 6, 6, 6, 4, 6, 9, 9, 13, 12, 9, 9, 9, 9, 9, 9, 11, 12, 13, 6, 6, 6, 6, 6, 3, 6, 8, 9, 12, 12, 9, 9, 9, 9, 9, 9, 11, 12, 12, 6, 6, 6, 8, 8, 8, 8, 10, 11, 13, 12, 11, 11, 9, 12, 13, 12, 13, 15, 15, 9, 9, 9, 12, 12, 10, 9, 12, 15, 15, 15, 15, 12, 13, 10, 12, 12, 14, 15, 15, 12, 9, 12, 12, 12, 12, 12, 12, 15, 15, 15, 14, 13, 12, 9, 9, 9, 9, 12, 9, 9, 6, 9, 10, 12, 12, 12, 12, 15, 13, 10, 9, 10, 9, 9, 9, 9, 9, 12, 9, 7, 6, 9, 9, 12, 11, 9, 9, 12, 10, 9, 9, 9, 9, 9, 9, 10, 9, 12, 9, 7, 7, 9, 9, 11, 9, 9, 9, 11, 9, 9, 9, 7, 6, 9, 9, 9, 9, 10, 9, 7, 6, 9, 9, 9, 9, 9, 9, 10, 9, 9, 7, 8, 8

])
dims_ = dims.reshape((N, N), order='F').T
subdomains = len(dims_)

min_size = np.min(dims)
max_size = np.max(dims)
print('sizes of the local reduced bases (min/max): {}/{}'.format(min_size, max_size))
# plt.matshow(dims_, cmap=cm.get_cmap('Purples', max_size - min_size + 1))
plt.imshow(dims_,
           origin='lower_left',
           interpolation='none', cmap=cm.get_cmap('Purples', max_size - min_size + 1),
           vmax=min_size, vmin=max_size)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(1, N, 2))
ax.set_yticks(np.arange(1, N, 2))

# Labels for major ticks
ax.set_xticklabels(np.arange(2, N+1, 2), fontsize=14)
ax.set_yticklabels(np.arange(2, N+1, 2), fontsize=14)

# Minor ticks
ax.set_xticks(np.arange(-.5, N, 1), minor=True)
ax.set_yticks(np.arange(-.5, N, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('basis_sizes.png', bbox_inches='tight')
# tikzplotlib.save('optional_enrichment.tex')

plt.figure()
dims = np.array([
2,  2,  4,  4,  4,  4,  4,  6,  4,  4,  4,  4,  6,  4,  4,  4,  4,
        4,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  4,  4,  4,  4,  6,  4,
        4,  4,  4,  4,  2,  2,  4,  4,  8,  8,  8,  8,  8, 12,  8,  8,  8,
        8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,  8,  8,  8,  8, 12,
        8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,  8,  8,
        8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,
        8,  8,  8,  8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,
        4,  4,  4,  8,  8,  8,  8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,
        8,  8,  4,  4,  6,  6, 12, 12, 12, 12, 12, 18, 12, 12, 12, 12, 18,
       12, 12, 12, 12, 12,  6,  6,  4,  4,  8,  8,  8,  8,  8, 12,  8,  8,
        8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,  8,  8,  8,  8,
       12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,  8,
        8,  8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,
        4,  8,  8,  8,  8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,
        4,  4,  6,  6, 12, 12, 12, 12, 12, 18, 12, 12, 12, 12, 18, 12, 12,
       12, 12, 12,  6,  6,  4,  4,  8,  8,  8,  8,  8, 12,  8,  8,  8,  8,
       12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,  8,  8,  8,  8, 12,  8,
        8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,  8,  8,  8,
        8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,  4,  4,  8,
        8,  8,  8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,  8,  4,  4,
        4,  4,  8,  8,  8,  8,  8, 12,  8,  8,  8,  8, 12,  8,  8,  8,  8,
        8,  4,  4,  2,  2,  4,  4,  4,  4,  4,  6,  4,  4,  4,  4,  6,  4,
        4,  4,  4,  4,  2,  2,  2,  2,  4,  4,  4,  4,  4,  6,  4,  4,  4,
        4,  6,  4,  4,  4,  4,  4,  2,  2
])
dims_ = dims.reshape((N, N), order='F').T
subdomains = len(dims_)

min_size = np.min(dims)
max_size = np.max(dims)
print('sizes of the local reduced bases (min/max): {}/{}'.format(min_size, max_size))

# patch 1
p1 = [-0.5, 3.5]
p2 = [3.5, 3.5]
p3 = [3.5, -0.5]
p4 = [-0.5, -0.5]
plt.plot(p1, p2, color="red", linewidth=2)
plt.plot(p4, p3, color="red", linewidth=4)
plt.plot(p1, p4, color="red", linewidth=4)
plt.plot(p2, p1, color="red", linewidth=2)

# patch 2
p1 = [8.5, 15.5]
p2 = [15.5, 15.5]
p3 = [15.5, 8.5]
p4 = [8.5, 8.5]
plt.plot(p1, p2, color="r", linewidth=2)
plt.plot(p4, p3, color="r", linewidth=2)
plt.plot(p1, p4, color="r", linewidth=2)
plt.plot(p2, p1, color="r", linewidth=2)
plt.imshow(dims_,
           origin='lower_left',
           interpolation='none', cmap=cm.get_cmap('Purples', max_size - min_size + 1),
           vmax=min_size, vmin=max_size)

ax = plt.gca();

# Major ticks
ax.set_xticks(np.arange(4.5, N, 5))
ax.set_yticks(np.arange(4.5, N, 5))

# Labels for major ticks
# ax.set_xticklabels(np.arange(4, N+1, 5))
# ax.set_yticklabels(np.arange(4, N+1, 5))
ax.set_xticklabels([])
ax.set_yticklabels([])

# Minor ticks
ax.set_xticks(np.arange(-.5, N, 1), minor=True)
ax.set_yticks(np.arange(-.5, N, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
ax.grid(which='major', color='g', linestyle='--', linewidth=2)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
# tikzplotlib.save('affine_decomposition.tex')
plt.tight_layout()
plt.savefig('affine_decomp.png', bbox_inches='tight')
plt.show()
