/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief OCL device-side ideal gas kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Calculates the pressure and sound speed for the mesh chunk using
 *  the ideal gas equation of state, with a fixed gamma of 1.4.
 */

#include "ocl_knls.h"

__kernel void ideal_gas_ocl_kernel(
    __global const double * restrict density,
    __global const double * restrict energy,
    __global double * restrict pressure,
    __global double * restrict soundspeed)
{

    int k = get_global_id(1);
    int j = get_global_id(0);

    double sound_speed_squared,v,pressurebyenergy,pressurebyvolume;

    if ( (j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE) ) {

        v = 1.0/density[ARRAYXY(j,k,XMAXPLUSFOUR)];

        pressure[ARRAYXY(j,k,XMAXPLUSFOUR)]=(1.4-1.0)*density[ARRAYXY(j,k,XMAXPLUSFOUR)]*energy[ARRAYXY(j,k,XMAXPLUSFOUR)];

        pressurebyenergy=(1.4-1.0)*density[ARRAYXY(j,k,XMAXPLUSFOUR)];

        pressurebyvolume=-density[ARRAYXY(j,k,XMAXPLUSFOUR)]*pressure[ARRAYXY(j,k,XMAXPLUSFOUR)];

        sound_speed_squared=v*v*(pressure[ARRAYXY(j,k,XMAXPLUSFOUR)]*pressurebyenergy-pressurebyvolume);

        soundspeed[ARRAYXY(j,k,XMAXPLUSFOUR)]=sqrt(sound_speed_squared);
    }
}

__kernel void viscosity_ocl_kernel(
        __global const double * restrict celldx,
        __global const double * restrict celldy,
        __global const double * restrict density0,
        __global const double * restrict pressure,
        __global double * restrict viscosity,
        __global const double * restrict xvel0,
        __global const double * restrict yvel0)
{
    double ugrad,vgrad,grad2,pgradx,pgrady,pgradx2,pgrady2,grad,ygrad,pgrad,xgrad,div,strain2,limiter;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) {

          ugrad = (xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                  +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                 -(xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                  +xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]);

          vgrad = (yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                  +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)])
                 -(yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                  +yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]);

          div = (celldx[j]*(ugrad) 
                +celldy[k]*(vgrad));

          strain2=0.5*(xvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)]
                      +xvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                      -xvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                      -xvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)])/celldy[k]
                 +0.5*(yvel0[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                      +yvel0[ARRAYXY(j+1,k+1,XMAXPLUSFIVE)]
                      -yvel0[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                      -yvel0[ARRAYXY(j  ,k+1,XMAXPLUSFIVE)])/celldx[j];

          pgradx=(pressure[ARRAYXY(j+1,k  ,XMAXPLUSFOUR)]
                 -pressure[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)])
                /(celldx[j]+celldx[j+1]);
          pgrady=(pressure[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]
                 -pressure[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)])
                /(celldy[k]+celldy[k+1]);

          pgradx2 = pgradx*pgradx;
          pgrady2 = pgrady*pgrady;

          limiter = ((0.5*(ugrad)/celldx[j])
                      *pgradx2+(0.5*(vgrad)/celldy[k])*pgrady2+strain2*pgradx*pgrady)
                  /fmax(pgradx2+pgrady2,1.0e-16);

          pgradx = copysign(fmax(1.0e-16,fabs(pgradx)),pgradx);
          pgrady = copysign(fmax(1.0e-16,fabs(pgrady)),pgrady);
          pgrad = sqrt(pgradx*pgradx+pgrady*pgrady);
          xgrad = fabs(celldx[j]*pgrad/pgradx);
          ygrad = fabs(celldy[k]*pgrad/pgrady);
          grad  = fmin(xgrad,ygrad);
          grad2 = grad*grad;

          if(limiter > 0.0 || div >= 0.0){
              viscosity[ARRAYXY(j,k,XMAXPLUSFOUR)]=0.0;
          } else {
              viscosity[ARRAYXY(j,k,XMAXPLUSFOUR)]=2.0*density0[ARRAYXY(j,k,XMAXPLUSFOUR)]*grad2*limiter*limiter;
          }
    }
}

__kernel void update_halo_bottom_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFOUR + j] = field[ (YMINPLUSONE+k)*XMAXPLUSFOUR+j ];

    }
}

__kernel void update_halo_bottom_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSTWO+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFIVE + j] = multiplier*field[ (YMINPLUSTWO+k)*XMAXPLUSFIVE+j ];

    }
}

__kernel void update_halo_bottom_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSTWO+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFIVE + j] = field[ (YMINPLUSTWO+k)*XMAXPLUSFIVE+j ];

    }
}

__kernel void update_halo_bottom_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFOUR + j] = -1*field[ (YMINPLUSTWO+k)*XMAXPLUSFOUR+j ];

    }
}





__kernel void update_halo_top_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSONE+depth)) {

        field[ (YMAXPLUSTWO + k)*XMAXPLUSFOUR + j ] = field[ (YMAXPLUSONE - k)*XMAXPLUSFOUR + j ];
    }

}

__kernel void update_halo_top_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSTWO+depth)) {

        field[ (YMAXPLUSTHREE + k)*XMAXPLUSFIVE + j ] = multiplier*field[ (YMAXPLUSONE - k)*XMAXPLUSFIVE + j ];
    }

}

__kernel void update_halo_top_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSTWO+depth)) {

        field[ (YMAXPLUSTWO + k)*XMAXPLUSFIVE + j ] = field[ (YMAX - k)*XMAXPLUSFIVE + j ];
    }

}

__kernel void update_halo_top_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSONE+depth)) {

        field[ (YMAXPLUSTHREE + k)*XMAXPLUSFOUR + j ] = -1*field[ (YMAXPLUSONE - k)*XMAXPLUSFOUR + j ];
    }

}





__kernel void update_halo_left_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ (k*XMAXPLUSFOUR)+1-j  ] = field[ (k*XMAXPLUSFOUR)+2+j  ];
    }

}

__kernel void update_halo_left_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ (k*XMAXPLUSFIVE)+1-j  ] = multiplier*field[ (k*XMAXPLUSFIVE)+3+j  ];
    }

}

__kernel void update_halo_left_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ (k*XMAXPLUSFIVE)+1-j  ] = -1*field[ (k*XMAXPLUSFIVE)+3+j  ];
    }

}

__kernel void update_halo_left_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ (k*XMAXPLUSFOUR)+1-j  ] = field[ (k*XMAXPLUSFOUR)+3+j  ];
    }

}




__kernel void update_halo_right_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ k*XMAXPLUSFOUR+XMAXPLUSTWO+j ] = field[ k*XMAXPLUSFOUR+XMAXPLUSONE-j ];
    }
}

__kernel void update_halo_right_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ k*XMAXPLUSFIVE+XMAXPLUSTHREE+j ] = multiplier*field[ k*XMAXPLUSFIVE+XMAXPLUSONE-j ];
    }
}

__kernel void update_halo_right_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ k*XMAXPLUSFIVE+XMAXPLUSTHREE+j ] = -1*field[ k*XMAXPLUSFIVE+XMAXPLUSONE-j ];
    }
}

__kernel void update_halo_right_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ k*XMAXPLUSFOUR+XMAXPLUSTWO+j ] = field[ k*XMAXPLUSFOUR+XMAX-j ];
    }
}
