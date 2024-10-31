import emg3d
import numpy as np
import ctypes as ct
from numpy.testing import assert_allclose
import numpy.ctypeslib as npct


c_doublep = ct.POINTER(ct.c_double)
C_lib = npct.load_library("./emg3d/solve.so", ".")


def solve(amat, bvec):
    C_lib.solve(amat.ctypes.data_as(c_doublep), bvec.ctypes.data_as(c_doublep))


# Create complex symmetric matrix A.
avec = np.zeros(36, dtype=np.complex128)
avec[::6] = np.array([100+100j, 1, 1, 1, 2, 40+3j])
avec[1:-6:6] = np.array([2, 2, 2+10j, 3, 3+6j])
avec[2:-12:6] = np.array([3j, 10+10j, 4, 4])
avec[3:-18:6] = np.array([4, 5, 2])
avec[4:-24:6] = np.array([5, 6])

# Create solution vector x
x = np.array([1.+1j, 1, 1j, 2+1j, 1+2j, 3+3.j])

# Re-arrange to full (symmetric) other numpy solvers.
amat = np.zeros((6, 6), dtype=avec.dtype)
for i in range(6):
    for j in range(i+1):
        amat[i, j] = avec[i+5*j]
        amat[j, i] = avec[i+5*j]

# Compute b = A x
b = amat@x

# 1. Check with numpy
# Ensure that our dummy-linear-equation system works fine.
xnp = np.linalg.solve(amat, b)                 # Solve A x = b
assert_allclose(x, xnp)                        # Check

# 2. Check current
xnb = b.copy()
emg3d.core.solve(avec.copy(), xnb)             # Solve A x = b
assert_allclose(x, xnb)                        # Check

# 2. Check new
xc = b.copy()
solve(avec.copy(), xc)                         # Solve A x = b
assert_allclose(x, xc)                         # Check

print(f"Solution : {x}")
print(f"Numpy    : {xnp}")
print(f"Numba    : {xnb}")
print(f"C        : {xc}")
