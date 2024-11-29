#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <complex.h>

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

void solve( int n, double complex* amat, double complex* rhs);

void gauss_seidel(double complex *ex, double complex *ey, double complex *ez,
                  double complex *sx, double complex *sy, double complex *sz,
                  double complex *eta_x, double complex *eta_y, double complex *eta_z,
                  double *zeta, double *hx, double *hy, double *hz,
                  int nx, int ny, int nz, int nu) 
{
    int ix, iy, iz, ixh, iyh, izh;
    int nxy, ny1, nx1, nz1, nzy1;
    int ixm, iym, izm;
    int ixp, iyp, izp;
    double complex st[6];
    double complex rhs[6];
    //double complex *tex, *tey, *tez, *tsx, *tsy, *tsz, *teta_x, *teta_y, *teta_z, *tzeta;

    // Pre-allocating `A` for the six * nzy edges attached to one node
    double complex amat[36];

    fprintf(stderr,"nx=%d ny=%d nz=%d\n", nx, ny, nz);

    nxy = nx*ny;
    nx1 = (nx+1);
    ny1 = (ny+1);
    nz1 = (nz+1);
    nzy1 = nz*ny1;
    // Get half of the inverse widths
    double *kx = (double *)malloc(nx * sizeof(double));
    double *ky = (double *)malloc(ny * sizeof(double));
    double *kz = (double *)malloc(nz * sizeof(double));
    for (int i = 0; i < nx; i++) kx[i] = 0.5 / hx[i];
    for (int i = 0; i < ny; i++) ky[i] = 0.5 / hy[i];
    for (int i = 0; i < nz; i++) kz[i] = 0.5 / hz[i];

    for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
        //fprintf(stderr,"[kx=%f ky=%f kz=%f\n",kx[i], ky[i], kz[i]);
        fprintf(stderr,"ey[%d %d %d]=%e %ej \n",k, j, i, crealf(ey[i+j*nx1+k*nx1*ny]), cimagf(ey[i+j*nx1+k*nx1*ny]) );
    }
    }
    }

    // Direction-switch for Gauss-Seidel
    int iback = 0;

/* Tranpose arrays for the moment 
    tex = malloc(nx*ny*nz * sizeof(double complex));
    tey = malloc(nx*ny*nz * sizeof(double complex));
    tez = malloc(nx*ny*nz * sizeof(double complex));
    tsx = malloc(nx*ny*nz * sizeof(double complex));
    tsy = malloc(nx*ny*nz * sizeof(double complex));
    tsz = malloc(nx*ny*nz * sizeof(double complex));
    teta_x = malloc(nx*ny*nz * sizeof(double complex));
    teta_y = malloc(nx*ny*nz * sizeof(double complex));
    teta_z = malloc(nx*ny*nz * sizeof(double complex));
    tzeta = malloc(nx*ny*nz * sizeof(double complex));
*/

    // Smoothing steps
    for (int n = 0; n < nu; n++) {
        // Direction of Gauss-Seidel ordering; 0=forward, 1=backward
        iback = 1 - iback;

        // Loop over cells, keeping boundaries fixed; x-fastest, then y, z.
        for (izh = 1; izh < nz; izh++) {
            iz = iback ? nz - izh : izh;
            izm = iz - 1;
            izp = iz + 1;

            for (iyh = 1; iyh < ny; iyh++) {
                iy = iback ? ny - iyh : iyh;
                iym = iy - 1;
                iyp = iy + 1;

                for (ixh = 1; ixh < nx; ixh++) {
                    ix = iback ? nx - ixh : ixh;
                    ixm = ix - 1;
                    ixp = ix + 1;

                    // Averaging of 1/mu_r
                    double mzyLxm = ky[iym] * (zeta[ixm + iym * nx + iz* nxy] +
                                               zeta[ixm + iym * nx + izm* nxy]);
        //fprintf(stderr,"%d %d %d %f  %f \n",iz, iy, ix, mzyLxm, zeta[ix + iy*nx + iz*nx*ny]);
        //fprintf(stderr,"%d %d %d %f\n",izh, iyh, ixh, crealf(mzyLxm), cimagf(mzyLxm));
                    double mzyRxm = ky[iy] *  (zeta[ixm + iy  * nx + iz * nxy] +
                                               zeta[ixm + iy  * nx + izm* nxy]);
                    double myzLxm = kz[izm] * (zeta[ixm + iy  * nx + izm* nxy] +
                                               zeta[ixm + iym * nx + izm* nxy]);
                    double myzRxm = kz[iz] *  (zeta[ixm + iy  * nx + iz * nxy] +
                                               zeta[ixm + iym * nx + iz * nxy]);
                    double mzyLxp = ky[iym] * (zeta[ix  + iym * nx + iz * nxy] +
                                               zeta[ix  + iym * nx + izm* nxy]);
                    double mzyRxp = ky[iy] *  (zeta[ix  + iy  * nx + iz * nxy] +
                                               zeta[ix  + iy  * nx + izm* nxy]);
                    double myzLxp = kz[izm] * (zeta[ix  + iy  * nx + izm* nxy] +
                                               zeta[ix  + iym * nx + izm* nxy]);
                    double myzRxp = kz[iz] *  (zeta[ix  + iy  * nx + iz * nxy] +
                                               zeta[ix  + iym * nx + iz * nxy]);
 
                    double mzxLym = kx[ixm] * (zeta[ixm + iym * nx + iz * nxy] +
                                               zeta[ixm + iym * nx + izm* nxy]);
                    double mzxRym = kx[ix] *  (zeta[ix  + iym * nx + iz * nxy] +
                                               zeta[ix  + iym * nx + izm* nxy]);
                    double mxzLym = kz[izm] * (zeta[ix  + iym * nx + izm* nxy] +
                                               zeta[ixm + iym * nx + izm* nxy]);
                    double mxzRym = kz[iz] *  (zeta[ix  + iym * nx + iz * nxy] +
                                               zeta[ixm + iym * nx + iz * nxy]);
                    double mzxLyp = kx[ixm] * (zeta[ixm + iy  * nx + iz * nxy] +
                                               zeta[ixm + iy  * nx + izm* nxy]);
                    double mzxRyp = kx[ix] *  (zeta[ix  + iy  * nx + iz * nxy] +
                                               zeta[ix  + iy  * nx + izm* nxy]);
                    double mxzLyp = kz[izm] * (zeta[ix  + iy  * nx + izm* nxy] +
                                               zeta[ixm + iy  * nx + izm* nxy]);
                    double mxzRyp = kz[iz] *  (zeta[ix  + iy  * nx + iz * nxy] +
                                               zeta[ixm + iy  * nx + iz * nxy]);

                    double myxLzm = kx[ixm] * (zeta[ixm + iy  * nx + izm* nxy] +
                                               zeta[ixm + iym * nx + izm* nxy]);
                    double myxRzm = kx[ix] *  (zeta[ix  + iy  * nx + izm* nxy] +
                                               zeta[ix  + iym * nx + izm* nxy]);
                    double mxyLzm = ky[iym] * (zeta[ix  + iym * nx + izm* nxy] +
                                               zeta[ixm + iym * nx + izm* nxy]);
                    double mxyRzm = ky[iy] *  (zeta[ix  + iy  * nx + izm* nxy] +
                                               zeta[ixm + iy  * nx + izm* nxy]);
                    double myxLzp = kx[ixm] * (zeta[ixm + iy  * nx + iz * nxy] +
                                               zeta[ixm + iym * nx + iz * nxy]);
                    double myxRzp = kx[ix] *  (zeta[ix  + iy  * nx + iz * nxy] +
                                               zeta[ix  + iym * nx + iz * nxy]);
                    double mxyLzp = ky[iym] * (zeta[ix  + iym * nx + iz * nxy] +
                                               zeta[ixm + iym * nx + iz * nxy]);
                    double mxyRzp = ky[iy] *  (zeta[ix  + iy  * nx + iz * nxy] +
                                               zeta[ixm + iy  * nx + iz * nxy]);

                    // Diagonal elements
                    st[0] = (eta_x[ixm + iy  * nx + iz * nxy] +
                             eta_x[ixm + iy  * nx + izm* nxy] +
                             eta_x[ixm + iym * nx + iz * nxy] +
                             eta_x[ixm + iym * nx + izm* nxy]);
                    st[1] = (eta_x[ix  + iy  * nx + iz * nxy] +
                             eta_x[ix  + iy  * nx + izm* nxy] +
                             eta_x[ix  + iym * nx + iz * nxy] +
                             eta_x[ix  + iym * nx + izm* nxy]);
                    st[2] = (eta_y[ix  + iym * nx + iz * nxy] +
                             eta_y[ix  + iym * nx + izm* nxy] +
                             eta_y[ixm + iym * nx + iz * nxy] +
                             eta_y[ixm + iym * nx + izm* nxy]);
                    st[3] = (eta_y[ix  + iy  * nx + iz * nxy] +
                             eta_y[ix  + iy  * nx + izm * nxy] +
                             eta_y[ixm + iy  * nx + iz * nxy] +
                             eta_y[ixm + iy  * nx + izm* nxy]);
                    st[4] = (eta_z[ix  + iy  * nx + izm* nxy] +
                             eta_z[ix  + iym * nx + izm* nxy] +
                             eta_z[ixm + iy  * nx + izm* nxy] +
                             eta_z[ixm + iym * nx + izm* nxy]);
                    st[5] = (eta_z[ix  + iy  * nx + iz * nxy] +
                             eta_z[ix  + iym * nx + iz * nxy] +
                             eta_z[ixm + iy  * nx + iz * nxy] +
                             eta_z[ixm + iym * nx + iz * nxy]);

/*
                    fprintf(stderr,"[st %f %fj %f %fj %f %fj\n",crealf(st[0]), cimagf(st[0]), 
                                                            crealf(st[1]), cimagf(st[1]), 
                                                            crealf(st[2]), cimagf(st[2])); 
                    fprintf(stderr,"st %f %fj %f %fj %f %fj]\n",crealf(st[3]), cimagf(st[3]), 
                                                            crealf(st[4]), cimagf(st[4]), 
                                                            crealf(st[5]), cimagf(st[5])); 
*/

                    memset(&amat[0],0,sizeof(double complex)*36);
                    for (int k = 0; k < 6; k++) {
                        amat[6 * k] = -st[k] * 0.25;  // Fill diagonal elements
                    }

                    // Complete diagonals
                    // A is symmetric and curl-curl part is real-valued
                    amat[0]  += mzyRxm / hy[iy] + mzyLxm / hy[iym];
                    amat[0]  += myzRxm / hz[iz] + myzLxm / hz[izm];
                    amat[6]  += mzyRxp / hy[iy] + mzyLxp / hy[iym];
                    amat[6]  += myzRxp / hz[iz] + myzLxp / hz[izm];
                    amat[12] += mzxRym / hx[ix] + mzxLym / hx[ixm];
                    amat[12] += mxzRym / hz[iz] + mxzLym / hz[izm];
                    amat[18] += mzxRyp / hx[ix] + mzxLyp / hx[ixm];
                    amat[18] += mxzRyp / hz[iz] + mxzLyp / hz[izm];
                    amat[24] += myxRzm / hx[ix] + myxLzm / hx[ixm];
                    amat[24] += mxyRzm / hy[iy] + mxyLzm / hy[iym];
                    amat[30] += myxRzp / hx[ix] + myxLzp / hx[ixm];
                    amat[30] += mxyRzp / hy[iy] + mxyLzp / hy[iym];

                    // Off-diagonal elements
                    // Upper triangle not needed and not set.
                    // The elements
                    //   [1, 0] (1); [3, 2] (13); and [5, 4] (21)
                    // are all zero.
                    amat[2] = -mzyLxm / hx[ixm];
                    amat[3] = mzyRxm / hx[ixm];
                    amat[4] = -myzLxm / hx[ixm];
                    amat[5] = myzRxm / hx[ixm];
                    amat[7] = mzyLxp / hx[ix];
                    amat[8] = -mzyRxp / hx[ix];
                    amat[9] = myzLxp / hx[ix];
                    amat[10] = -myzRxp / hx[ix];
                    amat[14] = -mxzLym / hy[iym];
                    amat[15] = mxzRym / hy[iym];
                    amat[19] = mxzLyp / hy[iy];
                    amat[20] = -mxzRyp / hy[iy];

                    // Fill residual (b - Ux^{(k)})
                    // Note: rhs is NOT the full residual at this point

                    // Get the 6 edges for ix, iy, and iz
                    rhs[0] = sx[ixm + iy*nx   + iz*nx*ny1];
                    rhs[1] = sx[ix  + iy*nx   + iz*nx*ny1];
                    rhs[2] = sy[ix  + iym*nx1 + iz*nx1*ny];
                    rhs[3] = sy[ix  + iy*nx1  + iz*nx1*ny];
                    rhs[4] = sz[ix  + iy*nx1  + izm*nx1*ny1];
                    rhs[5] = sz[ix  + iy*nx1  + iz*nx1*ny1];
/*
                    fprintf(stderr," %d %d %d rhs=[%.3e %.3ej %.3e %.3ej %.3e %.3ej ",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), 
                                                            crealf(rhs[1]), cimagf(rhs[1]), 
                                                            crealf(rhs[2]), cimagf(rhs[2])); 
                    fprintf(stderr,"%.3e %.3ej %.3e %.3ej %.3e %.3ej]\n",crealf(rhs[3]), cimagf(rhs[3]), 
                                                            crealf(rhs[4]), cimagf(rhs[4]), 
                                                            crealf(rhs[5]), cimagf(rhs[5])); 
*/

                    // Residual updates
                    rhs[0] += mzyRxm * (ey[ixm + iy*nx1  + iz * nx1*ny] / hx[ixm] +
                                        ex[ixm + iyp*nx  + iz * nx*ny1] / hy[iy]);
                    rhs[0] += mzyLxm *(-ey[ixm + iym*nx1 + iz * nx1*ny] / hx[ixm] +
                                        ex[ixm + iym*nx  + iz * nx*ny1] / hy[iym]);
                    rhs[0] += myzRxm * (ez[ixm + iy*nx1  + iz * nx1*ny1] / hx[ixm] +
                                        ex[ixm + iy*nx   + izp* nx*ny1] / hz[iz]);
                    rhs[0] += myzLxm *(-ez[ixm + iy*nx1  + izm* nx1*ny1] / hx[ixm] +
                                        ex[ixm + iy*nx   + izm* nx*ny1] / hz[izm]);
                    //fprintf(stderr,"Crhs[%d %d %d][0]=%e %ej m=%e %e %e %e\n",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), mzyRxm, mzyLxm, myzRxm, myzLxm, crealf(ey[ixm + iy*nx1  + iz * nx1*ny]) );

                    rhs[1] += mzyRxp *(-ey[ixp + iy*nx1  + iz * nx1*ny] / hx[ix] +
                                        ex[ix  + iyp*nx  + iz * nx*ny1] / hy[iy]);
                    rhs[1] += mzyLxp * (ey[ixp + iym*nx1 + iz * nx1*ny] / hx[ix] +
                                        ex[ix  + iym*nx  + iz * nx*ny1] / hy[iym]);
                    rhs[1] += myzRxp *(-ez[ixp + iy*nx1  + iz * nx1*ny1] / hx[ix] +
                                        ex[ix  + iy*nx   + izp* nx*ny1] / hz[iz]);
                    rhs[1] += myzLxp * (ez[ixp + iy*nx1  + izm* nx1*ny1] / hx[ix] +
                                        ex[ix  + iy*nx   + izm* nx*ny1] / hz[izm]);

                    rhs[2] += mzxRym * (ey[ixp + iym*nx1 + iz * nx1*ny] / hx[ix] +
                                        ex[ix  + iym*nx  + iz * nx*ny1] / hy[iym]);
                    rhs[2] += mzxLym * (ey[ixm + iym*nx1 + iz * nx1*ny] / hx[ixm] -
                                        ex[ixm + iym*nx  + iz * nx*ny1] / hy[iym]);
                    rhs[2] += mxzRym * (ez[ix  + iym*nx1 + iz * nx1*ny1] / hy[iym] +
                                        ey[ix  + iym*nx1 + izp* nx1*ny] / hz[iz]);
                    rhs[2] += mxzLym *(-ez[ix  + iym*nx1 + izm* nx1*ny1] / hy[iym] +
                                        ey[ix  + iym*nx1 + izm* nx1*ny] / hz[izm]);


                    rhs[3] += mzxRyp * (ey[ixp + iy*nx1  + iz * nx1*ny] / hx[ix] -
                                        ex[ix  + iyp*nx  + iz * nx*ny1] / hy[iy]);
                    rhs[3] += mzxLyp * (ey[ixm + iy*nx1  + iz * nx1*ny] / hx[ixm] +
                                        ex[ixm + iyp*nx  + iz * nx*ny1] / hy[iy]);
                    rhs[3] += mxzRyp *(-ez[ix  + iyp*nx1 + iz * nx1*ny1] / hy[iy] +
                                        ey[ix  + iy*nx1  + izp* nx1*ny] / hz[iz]);
                    rhs[3] += mxzLyp * (ez[ix  + iyp*nx1 + izm* nx1*ny1] / hy[iy] +
                                        ey[ix  + iy*nx1  + izm* nx1*ny] / hz[izm]);

                    rhs[4] += myxRzm * (ez[ixp + iy*nx1  + izm* nx1*ny1] / hx[ix] +
                                        ex[ix  + iy*nx   + izm* nx*ny1] / hz[izm]);
                    rhs[4] += myxLzm * (ez[ixm + iy*nx1  + izm* nx1*ny1] / hx[ixm] -
                                        ex[ixm + iy*nx   + izm* nx*ny1] / hz[izm]);
                    rhs[4] += mxyRzm * (ez[ix  + iyp*nx1 + izm* nx1*ny1] / hy[iy] +
                                        ey[ix  + iy*nx1  + izm* nx1*ny] / hz[izm]);
                    rhs[4] += mxyLzm * (ez[ix  + iym*nx1 + izm* nx1*ny1] / hy[iym] -
                                        ey[ix  + iym*nx1 + izm* nx1*ny] / hz[izm]);
 
                    rhs[5] += myxRzp * (ez[ixp + iy*nx1  + iz * nx1*ny1] / hx[ix] +
                                        ex[ix  + iy*nx   + iz * nx*ny1] / hz[iz]);
                    rhs[5] += myxLzp * (ez[ixm + iy*nx1  + iz * nx1*ny1] / hx[ixm] -
                                        ex[ixm + iy*nx   + iz * nx*ny1] / hz[iz]);
                    rhs[5] += mxyRzp *(-ez[ix  + iy*nx1  + izp* nx1*ny1] / hy[iy] +
                                        ey[ix  + iy*nx1  + iz * nx1*ny] / hz[iz]);
                    rhs[5] += mxyLzp * (ez[ix  + iy*nx1  + izm* nx1*ny1] / hy[iy] +
                                        ey[ix  + iy*nx1  + iz * nx1*ny] / hz[iz]);

                    // Solve the linear system A x = b
                    solve(6, amat, rhs);

                    // Update e-field (here we could apply damping weights)
                    ex[ixm + iy*nx   + iz *nx*ny1] = rhs[0];
                    ex[ix  + iy*nx   + iz *nx*ny1] = rhs[1];
                    ey[ix  + iym*nx1 + iz *nx1*ny] = rhs[2];
                    ey[ix  + iy*nx1  + iz *nx1*ny] = rhs[3];
                    ez[ix  + iy*nx1  + izm* nx1*ny1] = rhs[4];
                    ez[ix  + iy*nx1  + iz * nx1*ny1] = rhs[5];
                }
            }
        }
    }

    // Free allocated memory
    free(kx);
    free(ky);
    free(kz);
    return;
}

