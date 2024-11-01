import emg3d
import numpy as np

from emg3d import _ccore

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

atol = 0
rtol = 1e-12

def compare(a, b, atol, rtol, title):
    same = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    error = np.abs(a - b)
    if not same:
        print(f"  $$ {title} :: {['❌', '✅'][same]}")
        print(f"     => True ***")
        print(a)
        print(f"     => Code ***")
        print(b)
        print()
        print(f"    Data {['are NOT', 'ARE'][same]} the same "
            f"(given rtol={rtol}, atol={atol})\n"
            f"    max abs error: {np.max(error):.4g}; max rel "
            f"error: {np.max(error/np.abs(a)):.4g}\n")

# 1. Check with numpy
# Ensure that our dummy-linear-equation system works fine.
xnp = np.linalg.solve(amat, b)                 # Solve A x = b
compare(x, xnp, atol, rtol, 'numpy')

# 2. Check current
xnb = b.copy()
emg3d.core.solve(avec.copy(), xnb)             # Solve A x = b
compare(x, xnb, atol, rtol, 'numba')

# 2. Check new
xc = b.copy()
_ccore.solve(avec.copy(), xc)            # Solve A x = b
compare(x, xc, atol, rtol, 'C')

print(f"Solution : {x}")
print(f"Numpy    : {np.array2string(xnp, precision=2)}")
print(f"Numba    : {np.array2string(xnb, precision=2)}")
print(f"C        : {np.array2string(xc, precision=2)}")
