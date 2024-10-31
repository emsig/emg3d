#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<complex.h>

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

/* Solve A x = b using a non-standard Cholesky factorisation.

    Solve the system A x = b using a non-standard Cholesky factorisation
    without pivoting for a symmetric, complex matrix A tailored to the problem
    of the multigrid solver. The matrix A (``amat``) is an array of length 6*n,
    containing the main diagonal and the first five lower off-diagonals
    (ordered so that the first element of the main diagonal is followed by the
    first elements of the off diagonals, then the second elements and so on).
    The vector ``bvec`` has length b.

    The solution is placed in b (``bvec``), and A (``amat``) is replaced by its
    decomposition.

    1. Non-standard Cholesky factorisation.

        From [Muld07]_: «We use a non-standard Cholesky factorisation. The
        standard factorisation factors a Hermitian matrix A into L L^H, where L
        is a lower triangular matrix and L^H its complex conjugate transpose.
        In our case, the discretisation is based on the Finite Integration
        Technique ([Weil77]_) and provides a matrix A that is complex-valued
        and symmetric: A = A^T, where the superscript T denotes the transpose.
        The line relaxation scheme takes a matrix B that is a subset of A along
        the line. B is a complex symmetric band matrix with eleven diagonals.
        The non-standard Cholesky factorisation factors the matrix B into L
        L^T. Because of the symmetry, only the main diagonal and five lower
        diagonal elements of B need to be computed. The Cholesky factorisation
        replaces this matrix by L, containing six diagonals, after which the
        line relaxation can be carried out by simple back-substitution.»

        :math:`A = L D L^T` factorisation without pivoting:

        .. math::

            D(j) &= A(j,j)-\sum_{k=1}^{j-1} L(j,k)^2 D(k),\ j=1,..,n ;\\
            L(i,j) &= \frac{1}{D(j)}
                     \left[A(i,j)-\sum_{k=1}^{j-1} L(i,k)L(j,k)D(k)\right],
                     \ i=j+1,..,n .

        A and L are in this case arrays, where :math:`A(i, j) \rightarrow
        A(i+5j)`.

    2. Solve A x = b.
        Solve A x = b, given L which is the result from the factorisation in
        the first step (and stored in A), hence, solve L x = b, where x is
        stored in b:

        .. math::

            b(j) = b(j) - \sum_{k=1}^{j-1} L(j,k) x(k), j = 2,..,n .

    The result is equivalent with simply using :func:`numpy.linalg.solve`, but
    faster for the particular use-case of this code.

    Note that in this custom solver there is no pivoting, and the diagonals of
    the matrix cannot be zero.


    Parameters
    ----------
    amat : ndarray
        Banded matrix A provided as a vector of length 6*n, containing main
        diagonal plus first five lower diagonals.

    bvec : ndarray
        Right-hand-side vector b of length n.
*/

void core( int n, double complex* amat, double complex* bvec)
{
    int i, j, k;
    double complex d, h;

    // Number of unknowns
    //n = len(bvec)

    // Pre-allocate h
    //h = np.zeros(1, dtype=amat.dtype)[0]

    // 1. Get L from non-standard Cholesky L D L^T factorisation

    // First element (i = j = 0). Warning: Diagonals of amat cannot be 0!
    d = 1.0/amat[0];

    // Multiply to other elements of first column (j = 0)
    for ( i=1; i<MIN(n,6); i++ ) { 
        amat[i] *= d;
    }
    //for ( i=0; i<n*6; i++ ) fprintf(stderr,"amat[%d]=%f %f\n", i, crealf(amat[i]), cimagf(amat[i]) );

    // Other columns (1 to n)
    for ( j=1; j<n; j++ ) { 

        h = 0.0;
        for ( k=MAX(0,j-5); k<j; k++ ) { 
            h += amat[j+5*k]*amat[j+5*k]*amat[6*k];
        }

        amat[6*j] -= h;

        // Warning: Diagonals of amat cannot be 0!
        d = 1./amat[6*j];

        // Off-diagonals, rows i > j
        for ( i=j+1; i<MIN(n,j+6); i++ ) { 
            h = 0.0;
            for ( k=MAX(0,i-5); k<j; k++ ) { 
                h += amat[i+5*k]*amat[j+5*k]*amat[6*k];
            }
            amat[i+5*j] -= h;
            amat[i+5*j] *= d;
        }
    }

    // Replace diagonal by 1/D
    amat[6*(n-1)] = d;  // Last one is still around
    for ( j=n-2; j>-1; j-- ) { 
    //for j in range(n-2, -1, -1):
        amat[6*j] = 1./amat[6*j];
    }

    // 2. Solve A x = b

    // All elements except first column
    for ( j=1; j<n; j++ ) { 
        h = 0.;  // Reset h
        for ( k=MAX(0,j-5); k<j; k++ ) { 
            h += amat[j+5*k]*bvec[k];
        }
        bvec[j] -= h;
    }

    // Divide by diagonal; A[j, j] (hence A[6j]) contains 1/D[j]
    for ( j=0; j<n; j++ ) { 
        bvec[j] *= amat[6*j];
    }

    // Solve L^T x = b, x stored in b, L is 1 on diagonal
    for ( j=n-2; j>-1; j-- ) { 
        h = 0.;  // Reset h
        for ( k=j+1; k<MIN(n,j+6); k++ ) { 
            h += amat[k+5*j]*bvec[k];
        }
        bvec[j] -= h;
    }

    return;
}

