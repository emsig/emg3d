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
    int nxy, ny1, nx1, nz1, nyz;
    int ixm, iym, izm;
    int ixp, iyp, izp;
    double complex st[6];
    double complex rhs[6];
    //double complex *tex, *tey, *tez, *tsx, *tsy, *tsz, *teta_x, *teta_y, *teta_z, *tzeta;

    // Pre-allocating `A` for the six * nzy edges attached to one node
    double complex amat[36];

    fprintf(stderr,"nx=%d ny=%d nz=%d\n", nx, ny, nz);


    nxy = nx*ny;
    nyz = ny*nz;
    nx1 = (nx+1);
    ny1 = (ny+1);
    nz1 = (nz+1);
    // Get half of the inverse widths
    double *kx = (double *)malloc(nx * sizeof(double));
    double *ky = (double *)malloc(ny * sizeof(double));
    double *kz = (double *)malloc(nz * sizeof(double));
    for (int i = 0; i < nx; i++) kx[i] = 0.5 / hx[i];
    for (int i = 0; i < ny; i++) ky[i] = 0.5 / hy[i];
    for (int i = 0; i < nz; i++) kz[i] = 0.5 / hz[i];

/*
iz=5; iy=4; ix=4;
                    fprintf(stderr," T %d %d %d ex=%.3e %.3ej\n",iz, iy, ix, creal(ex[ix*ny1*nz1+iy*nz1+iz]), cimag(ex[ix*ny1*nz1+iy*nz1+iz]) );
                    fprintf(stderr," T %d %d %d ey=%.3e %.3ej\n",iz, iy, ix, creal(ey[ix*ny*nz1+iy*nz1+iz]), cimag(ey[ix*ny*nz1+iy*nz1+iz]) );
                    fprintf(stderr," T %d %d %d ez=%.3e %.3ej\n",iz, iy, ix, creal(ez[ix*ny1*nz+iy*nz+iz]), cimag(ez[ix*ny1*nz+iy*nz+iz]) );

    fprintf(stderr," %d %d %d ex=%.3e %.3ej\n",iz, iy, ix, creal(ex[ix+iy*nx+iz*nx*ny1]), cimag(ex[ix+iy*nx+iz*nx*ny1]) );
    fprintf(stderr," %d %d %d ey=%.3e %.3ej\n",iz, iy, ix, creal(ey[ix+iy*nx1+iz*nx1*ny]), cimag(ey[ix+iy*nx1+iz*nx1*ny]) );
    fprintf(stderr," %d %d %d ez=%.3e %.3ej\n",iz, iy, ix, creal(ez[ix+iy*nx1+iz*nx1*ny1]), cimag(ez[ix+iy*nx1+iz*nx1*ny1]) );
    fprintf(stderr," T %d %d %d ex=%.3e %.3ej\n",iz, iy, ix, creal(ex[ix*ny1*nz1+iy*nz1+iz]), cimag(ex[ix*ny1*nz1+iy*nz1+iz]) );
    fprintf(stderr," T %d %d %d ey=%.3e %.3ej\n",iz, iy, ix, creal(ey[ix*ny*nz1+iy*nz1+iz]), cimag(ey[ix*ny*nz1+iy*nz1+iz]) );
    fprintf(stderr," T %d %d %d ez=%.3e %.3ej\n",iz, iy, ix, creal(ez[ix*ny1*nz+iy*nz+iz]), cimag(ez[ix*ny1*nz+iy*nz+iz]) );
*/

//    for (int i = 0; i < nx; i++) fprintf(stderr,"kx=%f ", kx[i]);
//    for (int j = 0; j < ny; j++) fprintf(stderr,"ky=%f ", ky[j]);
//    for (int k = 0; k < nz; k++) fprintf(stderr,"kz=%f ", kz[k]);

/*
    for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
//        fprintf(stderr,"[kx=%f ky=%f kz=%f\n",kx[i], ky[i], kz[i]);
        fprintf(stderr,"ey[%d %d %d]=%e %ej \n",k, j, i, crealf(ey[i+j*nx1+k*nx1*ny]), cimagf(ey[i+j*nx1+k*nx1*ny]) );
    }
    }
    }
*/

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
                    //mzyLxm = ky[iym]*(zeta[ixm, iym, iz] + zeta[ixm, iym, izm])
                    double mzyLxm = ky[iym] * (zeta[ixm*nyz + iym * nz + iz] +
                                               zeta[ixm*nyz + iym * nz + izm]);
        //fprintf(stderr,"%d %d %d mzyLxm %f zeta %f \n",iz, iy, ix, mzyLxm, zeta[ix*nyz + iy*nz + iz]);
        //fprintf(stderr,"%d %d %d %f\n",izh, iyh, ixh, crealf(mzyLxm), cimagf(mzyLxm));
                    double mzyRxm = ky[iy] *  (zeta[ixm*nyz + iy  * nz + iz ] +
                                               zeta[ixm*nyz + iy  * nz + izm]);
                    double myzLxm = kz[izm] * (zeta[ixm*nyz + iy  * nz + izm] +
                                               zeta[ixm*nyz + iym * nz + izm]);
                    double myzRxm = kz[iz] *  (zeta[ixm*nyz + iy  * nz + iz ] +
                                               zeta[ixm*nyz + iym * nz + iz ]);
                    double mzyLxp = ky[iym] * (zeta[ix*nyz  + iym * nz + iz ] +
                                               zeta[ix*nyz  + iym * nz + izm]);
                    double mzyRxp = ky[iy] *  (zeta[ix*nyz  + iy  * nz + iz ] +
                                               zeta[ix*nyz  + iy  * nz + izm]);
                    double myzLxp = kz[izm] * (zeta[ix*nyz  + iy  * nz + izm] +
                                               zeta[ix*nyz  + iym * nz + izm]);
                    double myzRxp = kz[iz] *  (zeta[ix*nyz  + iy  * nz + iz ] +
                                               zeta[ix*nyz  + iym * nz + iz ]);
 
                    double mzxLym = kx[ixm] * (zeta[ixm*nyz + iym * nz + iz ] +
                                               zeta[ixm*nyz + iym * nz + izm]);
                    double mzxRym = kx[ix] *  (zeta[ix*nyz  + iym * nz + iz ] +
                                               zeta[ix*nyz  + iym * nz + izm]);
                    double mxzLym = kz[izm] * (zeta[ix*nyz  + iym * nz + izm] +
                                               zeta[ixm*nyz + iym * nz + izm]);
                    double mxzRym = kz[iz] *  (zeta[ix*nyz  + iym * nz + iz ] +
                                               zeta[ixm*nyz + iym * nz + iz ]);
                    double mzxLyp = kx[ixm] * (zeta[ixm*nyz + iy  * nz + iz ] +
                                               zeta[ixm*nyz + iy  * nz + izm]);
                    double mzxRyp = kx[ix] *  (zeta[ix*nyz  + iy  * nz + iz ] +
                                               zeta[ix*nyz  + iy  * nz + izm]);
                    double mxzLyp = kz[izm] * (zeta[ix*nyz  + iy  * nz + izm] +
                                               zeta[ixm*nyz + iy  * nz + izm]);
                    double mxzRyp = kz[iz] *  (zeta[ix*nyz  + iy  * nz + iz ] +
                                               zeta[ixm*nyz + iy  * nz + iz ]);

                    double myxLzm = kx[ixm] * (zeta[ixm*nyz + iy  * nz + izm] +
                                               zeta[ixm*nyz + iym * nz + izm]);
                    double myxRzm = kx[ix] *  (zeta[ix*nyz  + iy  * nz + izm] +
                                               zeta[ix*nyz  + iym * nz + izm]);
                    double mxyLzm = ky[iym] * (zeta[ix*nyz  + iym * nz + izm] +
                                               zeta[ixm*nyz + iym * nz + izm]);
                    double mxyRzm = ky[iy] *  (zeta[ix*nyz  + iy  * nz + izm] +
                                               zeta[ixm*nyz + iy  * nz + izm]);
                    double myxLzp = kx[ixm] * (zeta[ixm*nyz + iy  * nz + iz ] +
                                               zeta[ixm*nyz + iym * nz + iz ]);
                    double myxRzp = kx[ix] *  (zeta[ix*nyz  + iy  * nz + iz ] +
                                               zeta[ix*nyz  + iym * nz + iz ]);
                    double mxyLzp = ky[iym] * (zeta[ix*nyz  + iym * nz + iz ] +
                                               zeta[ixm*nyz + iym * nz + iz ]);
                    double mxyRzp = ky[iy] *  (zeta[ix*nyz  + iy  * nz + iz ] +
                                               zeta[ixm*nyz + iy  * nz + iz ]);

/*
                     st[0] = (eta_x[ixm + iy * nx + iz * nxy] +
                              eta_x[ixm + iy * nx + izm * nxy] +
                              eta_x[ixm + iym * nx + iz * nxy] +
                              eta_x[ixm + iym * nx + izm * nxy]);
                     st[1] = (eta_x[ix + iy * nx + iz * nxy] +
                              eta_x[ix + iy * nx + izm * nxy] +
                              eta_x[ix + iym * nx + iz * nxy] +
                              eta_x[ix + iym * nx + izm * nxy]);
                     st[2] = (eta_y[ix + iym * nx + iz * ny] +
                              eta_y[ix + iym * nx + izm * nz] +
                              eta_y[ixm + iym * nx + iz * nz] +
                              eta_y[ixm + iym * nx + izm * nz]);
                     st[3] = (eta_y[ix + iy * nx + iz * nz] +
                              eta_y[ix + iy * nx + izm * nz] +
                              eta_y[ixm + iy * nx + iz * nz] +
                              eta_y[ixm + iy * nx + izm * nz]);
                     st[4] = (eta_z[ix + iy * nx + izm * nz] +
                              eta_z[ix + iym * nx + izm * nz] +
                              eta_z[ixm + iy * nx + izm * nz] +
                              eta_z[ixm + iym * nx + izm * nz]);
                     st[5] = (eta_z[ix + iy * nx + iz * nz] +
                              eta_z[ix + iym * nx + iz * nz] +
                              eta_z[ixm + iy * nx + iz * nz] +
                              eta_z[ixm + iym * nx + iz * nz]);
*/

                    // Diagonal elements
                    st[0] = (eta_x[ixm*nyz + iy  * nz + iz ] +
                             eta_x[ixm*nyz + iy  * nz + izm] +
                             eta_x[ixm*nyz + iym * nz + iz ] +
                             eta_x[ixm*nyz + iym * nz + izm]);
                    st[1] = (eta_x[ix*nyz  + iy  * nz + iz ] +
                             eta_x[ix*nyz  + iy  * nz + izm] +
                             eta_x[ix*nyz  + iym * nz + iz ] +
                             eta_x[ix*nyz  + iym * nz + izm]);
                    st[2] = (eta_y[ix*nyz  + iym * nz + iz ] +
                             eta_y[ix*nyz  + iym * nz + izm] +
                             eta_y[ixm*nyz + iym * nz + iz ] +
                             eta_y[ixm*nyz + iym * nz + izm]);
                    st[3] = (eta_y[ix*nyz  + iy  * nz + iz ] +
                             eta_y[ix*nyz  + iy  * nz + izm] +
                             eta_y[ixm*nyz + iy  * nz + iz ] +
                             eta_y[ixm*nyz + iy  * nz + izm]);
                    st[4] = (eta_z[ix*nyz  + iy  * nz + izm] +
                             eta_z[ix*nyz  + iym * nz + izm] +
                             eta_z[ixm*nyz + iy  * nz + izm] +
                             eta_z[ixm*nyz + iym * nz + izm]);
                    st[5] = (eta_z[ix*nyz  + iy  * nz + iz ] +
                             eta_z[ix*nyz  + iym * nz + iz ] +
                             eta_z[ixm*nyz + iy  * nz + iz ] +
                             eta_z[ixm*nyz + iym * nz + iz ]);
/*

fprintf(stderr,"[eta_x-1 %f %fj \n",eta_x[ix*nyz  + iy*nz + iz ]);
fprintf(stderr,"[eta_x-2 %f %fj \n",eta_x[ix  + iy*nx + iz*nxy ]);
*/

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
                    //fprintf(stderr,"[amat %d %f %fj \n",k,crealf(amat[6*k]), cimagf(amat[6*k]));
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
                    //fprintf(stderr,"[amat [0]%f %fj \n",crealf(amat[6*0]), cimagf(amat[6*0]));

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

//                    for (int k = 0; k < 36; k++) {
//                    fprintf(stderr,"[amat %d %f %fj \n",k,crealf(amat[k]), cimagf(amat[k]));
//                    }
                    // Fill residual (b - Ux^{(k)})
                    // Note: rhs is NOT the full residual at this point

                    // Get the 6 edges for ix, iy, and iz
                    rhs[0] = sx[ixm*nz1*ny1 + iy*nz1  + iz];
                    rhs[1] = sx[ix*nz1*ny1  + iy*nz1  + iz];
                    rhs[2] = sy[ix*nz1*ny   + iym*nz1 + iz];
                    rhs[3] = sy[ix*nz1*ny   + iy*nz1  + iz];
                    rhs[4] = sz[ix*nz*ny1   + iy*nz   + izm];
                    rhs[5] = sz[ix*nz*ny1   + iy*nz   + iz];

/*
                    fprintf(stderr," %d %d %d sx=%.3e %.3ej\n",iz, iy, ix, creal(sx[ix+iy*nx+iz*nx*ny1]), cimag(sx[ix+iy*nx+iz*nx*ny1]) );
                    fprintf(stderr," T %d %d %d sx=%.3e %.3ej\n",iz, iy, ix, creal(sx[ix*ny1*nz1+iy*nz1+iz]), cimag(sx[ix*ny1*nz1+iy*nz1+iz]) );
                    fprintf(stderr," %d %d %d rhs=[%.3e %.3ej %.3e %.3ej %.3e %.3ej ",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), 
                                                            crealf(rhs[1]), cimagf(rhs[1]), 
                                                            crealf(rhs[2]), cimagf(rhs[2])); 
                    fprintf(stderr,"%.3e %.3ej %.3e %.3ej %.3e %.3ej]\n",crealf(rhs[3]), cimagf(rhs[3]), 
                                                            crealf(rhs[4]), cimagf(rhs[4]), 
                                                            crealf(rhs[5]), cimagf(rhs[5])); 

                    fprintf(stderr," %d %d %d ex=%.3e %.3ej\n",iz, iy, ix, creal(ex[ix+iy*nx+iz*nx*ny1]), cimag(ex[ix+iy*nx+iz*nx*ny1]) );
                    fprintf(stderr," %d %d %d ey=%.3e %.3ej\n",iz, iy, ix, creal(ey[ix+iy*nx1+iz*nx1*ny]), cimag(ey[ix+iy*nx1+iz*nx1*ny]) );
                    fprintf(stderr," %d %d %d ez=%.3e %.3ej\n",iz, iy, ix, creal(ez[ix+iy*nx1+iz*nx1*ny1]), cimag(ez[ix+iy*nx1+iz*nx1*ny1]) );
*/
/*
                    fprintf(stderr," T %d %d %d ex=%.3e %.3ej\n",iz, iy, ix, creal(ex[ix*ny1*nz1+iy*nz1+iz]), cimag(ex[ix*ny1*nz1+iy*nz1+iz]) );
                    fprintf(stderr," T %d %d %d ey=%.3e %.3ej\n",iz, iy, ix, creal(ey[ix*ny*nz1+iy*nz1+iz]), cimag(ey[ix*ny*nz1+iy*nz1+iz]) );
                    fprintf(stderr," T %d %d %d ez=%.3e %.3ej\n",iz, iy, ix, creal(ez[ix*ny1*nz+iy*nz+iz]), cimag(ez[ix*ny1*nz+iy*nz+iz]) );
*/
                    // Residual updates
                    rhs[0] += mzyRxm * (ey[ixm*ny *nz1 + iy *nz1 + iz ] / hx[ixm] +
                                        ex[ixm*ny1*nz1 + iyp*nz1 + iz ] / hy[iy]);
                    rhs[0] += mzyLxm *(-ey[ixm*ny *nz1 + iym*nz1 + iz ] / hx[ixm] +
                                        ex[ixm*ny1*nz1 + iym*nz1 + iz ] / hy[iym]);
                    rhs[0] += myzRxm * (ez[ixm*ny1*nz  + iy *nz  + iz ] / hx[ixm] +
                                        ex[ixm*ny1*nz1 + iy *nz1 + izp] / hz[iz]);
                    rhs[0] += myzLxm *(-ez[ixm*ny1*nz  + iy *nz  + izm] / hx[ixm] +
                                        ex[ixm*ny1*nz1 + iy *nz1 + izm] / hz[izm]);

                    //fprintf(stderr,"Crhs[%d %d %d][0]=%e %ej m=%e %e %e %e\n",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), mzyRxm, mzyLxm, myzRxm, myzLxm, crealf(ey[ixm + iy*nx1  + iz * nx1*ny]) );
                    //fprintf(stderr,"Crhs[%d %d %d][0]=%e %ej m=%e %e %e %e\n",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), mzyRxm, mzyLxm, myzRxm, myzLxm );

                    rhs[1] += mzyRxp *(-ey[ixp*ny *nz1 + iy *nz1 + iz ] / hx[ix] +
                                        ex[ix *ny1*nz1 + iyp*nz1 + iz ] / hy[iy]);
                    //fprintf(stderr,"Crhs[%d %d %d][1].1=%e %ej \n",iz, iy, ix, crealf(rhs[1]), cimagf(rhs[1]), mzyRxp);
                    rhs[1] += mzyLxp * (ey[ixp*ny *nz1 + iym*nz1 + iz ] / hx[ix] +
                                        ex[ix *ny1*nz1 + iym*nz1 + iz ] / hy[iym]);
                    //fprintf(stderr,"Crhs[%d %d %d][1].2=%e %ej \n",iz, iy, ix, crealf(rhs[1]), cimagf(rhs[1]), mzyLxp);
                    rhs[1] += myzRxp *(-ez[ixp*ny1*nz  + iy *nz  + iz ] / hx[ix] +
                                        ex[ix *ny1*nz1 + iy *nz1 + izp] / hz[iz]);
                    //fprintf(stderr,"Crhs[%d %d %d][1].3=%e %ej \n",iz, iy, ix, crealf(rhs[1]), cimagf(rhs[1]), myzRxp);
                    rhs[1] += myzLxp * (ez[ixp*ny1*nz  + iy *nz  + izm] / hx[ix] +
                                        ex[ix *ny1*nz1 + iy *nz1 + izm] / hz[izm]);
                    //fprintf(stderr,"Crhs[%d %d %d][1].4=%e %ej \n",iz, iy, ix, crealf(rhs[1]), cimagf(rhs[1]), myzLxp);

/*
                    fprintf(stderr,"Crhs[%d %d %d][1]=%e %ej m=%e %e %e %e\n",iz, iy, ix, crealf(rhs[1]), cimagf(rhs[1]), mzyRxp, mzyLxp, myzRxp, myzLxp );
                    fprintf(stderr," T %d %d %d ex=%.3e %.3ej\n",iz, iy, ix, creal(ex[ix*ny1*nz1+iy*nz1+iz]), cimag(ex[ix*ny1*nz1+iy*nz1+iz]) );
                    fprintf(stderr," T %d %d %d ey=%.3e %.3ej\n",iz, iy, ix, creal(ey[ix*ny*nz1+iy*nz1+iz]), cimag(ey[ix*ny*nz1+iy*nz1+iz]) );
                    fprintf(stderr," T %d %d %d ez=%.3e %.3ej\n",iz, iy, ix, creal(ez[ix*ny1*nz+iy*nz+iz]), cimag(ez[ix*ny1*nz+iy*nz+iz]) );
*/
 
                    rhs[2] += mzxRym * (ey[ixp*ny *nz1 + iym*nz1 + iz ] / hx[ix] +
                                        ex[ix *ny1*nz1 + iym*nz1 + iz ] / hy[iym]);
                    rhs[2] += mzxLym * (ey[ixm*ny *nz1 + iym*nz1 + iz ] / hx[ixm] -
                                        ex[ixm*ny1*nz1 + iym*nz1 + iz ] / hy[iym]);
                    rhs[2] += mxzRym * (ez[ix *ny1*nz  + iym*nz  + iz ] / hy[iym] +
                                        ey[ix *ny *nz1 + iym*nz1 + izp] / hz[iz]);
                    rhs[2] += mxzLym *(-ez[ix *ny1*nz  + iym*nz  + izm] / hy[iym] +
                                        ey[ix *ny *nz1 + iym*nz1 + izm] / hz[izm]);


                    rhs[3] += mzxRyp * (ey[ixp*ny *nz1 + iy*nz1  + iz ] / hx[ix] -
                                        ex[ix *ny1*nz1 + iyp*nz1 + iz ] / hy[iy]);
                    rhs[3] += mzxLyp * (ey[ixm*ny *nz1 + iy*nz1  + iz ] / hx[ixm] +
                                        ex[ixm*ny1*nz1 + iyp*nz1 + iz ] / hy[iy]);
                    rhs[3] += mxzRyp *(-ez[ix *ny1*nz  + iyp*nz  + iz ] / hy[iy] +
                                        ey[ix *ny *nz1 + iy*nz1  + izp] / hz[iz]);
                    rhs[3] += mxzLyp * (ez[ix *ny1*nz  + iyp*nz  + izm] / hy[iy] +
                                        ey[ix *ny *nz1 + iy*nz1  + izm] / hz[izm]);

                    rhs[4] += myxRzm * (ez[ixp*ny1*nz  + iy*nz   + izm] / hx[ix] +
                                        ex[ix *ny1*nz1 + iy*nz1  + izm] / hz[izm]);
                    rhs[4] += myxLzm * (ez[ixm*ny1*nz  + iy*nz   + izm] / hx[ixm] -
                                        ex[ixm*ny1*nz1 + iy*nz1  + izm] / hz[izm]);
                    rhs[4] += mxyRzm * (ez[ix *ny1*nz  + iyp*nz  + izm] / hy[iy] +
                                        ey[ix *ny *nz1 + iy*nz1  + izm] / hz[izm]);
                    rhs[4] += mxyLzm * (ez[ix *ny1*nz  + iym*nz  + izm] / hy[iym] -
                                        ey[ix *ny *nz1 + iym*nz1 + izm] / hz[izm]);
 
                    rhs[5] += myxRzp * (ez[ixp*ny1*nz  + iy*nz   + iz ] / hx[ix] -
                                        ex[ix *ny1*nz1 + iy*nz1  + izp] / hz[iz]);
                    rhs[5] += myxLzp * (ez[ixm*ny1*nz  + iy*nz   + iz ] / hx[ixm] +
                                        ex[ixm*ny1*nz1 + iy*nz1  + izp] / hz[iz]);
                    rhs[5] += mxyRzp * (ez[ix *ny1*nz  + iyp*nz  + iz ] / hy[iy] -
                                        ey[ix *ny *nz1 + iy*nz1  + izp] / hz[iz]);
                    rhs[5] += mxyLzp * (ez[ix *ny1*nz  + iym*nz  + iz ] / hy[iym] +
                                        ey[ix *ny *nz1 + iym*nz1 + izp] / hz[iz]);

/*
                    fprintf(stderr," %d %d %d rhs2=[%.3e %.3ej %.3e %.3ej %.3e %.3ej ",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), 
                                                            crealf(rhs[1]), cimagf(rhs[1]), 
                                                            crealf(rhs[2]), cimagf(rhs[2])); 
                    fprintf(stderr,"%.3e %.3ej %.3e %.3ej %.3e %.3ej]\n",crealf(rhs[3]), cimagf(rhs[3]), 
                                                            crealf(rhs[4]), cimagf(rhs[4]), 
                                                            crealf(rhs[5]), cimagf(rhs[5])); 
*/
                    // Solve the linear system A x = b
                    solve(6, amat, rhs);
		    
		    /*
                    fprintf(stderr," %d %d %d rhs3=[%.3e %.3ej %.3e %.3ej %.3e %.3ej ",iz, iy, ix, crealf(rhs[0]), cimagf(rhs[0]), 
                                                            crealf(rhs[1]), cimagf(rhs[1]), 
                                                            crealf(rhs[2]), cimagf(rhs[2])); 
                    fprintf(stderr,"%.3e %.3ej %.3e %.3ej %.3e %.3ej]\n",crealf(rhs[3]), cimagf(rhs[3]), 
                                                            crealf(rhs[4]), cimagf(rhs[4]), 
                                                            crealf(rhs[5]), cimagf(rhs[5])); 
		    */

                    // Update e-field (here we could apply damping weights)
                    ex[ixm*ny1*nz1 + iy*nz1  + iz ] = rhs[0];
                    ex[ix*ny1*nz1  + iy*nz1  + iz ] = rhs[1];
                    ey[ix*ny*nz1   + iym*nz1 + iz ] = rhs[2];
                    ey[ix*ny*nz1   + iy*nz1  + iz ] = rhs[3];
                    ez[ix*ny1*nz   + iy*nz   + izm] = rhs[4];
                    ez[ix*ny1*nz   + iy*nz   + iz ] = rhs[5];
                }
            }
        }
    }
    FILE *fp;
    fp = fopen("Cex.bin", "w+");
    fwrite(ex, sizeof(double complex), nx*ny1*nz1, fp);
    fclose(fp);
    fp = fopen("Cey.bin", "w+");
    fwrite(ey, sizeof(double complex), nx1*ny*nz1, fp);
    fclose(fp);
    fp = fopen("Cez.bin", "w+");
    fwrite(ez, sizeof(double complex), nx1*ny1*nz, fp);
    fclose(fp);

    // Free allocated memory
    free(kx);
    free(ky);
    free(kz);
    return;
}

