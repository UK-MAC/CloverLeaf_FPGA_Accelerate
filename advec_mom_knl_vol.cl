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
 *  @brief OCL device-side advection momentum kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
 */

#include "ocl_knls.h"

__kernel void advec_mom_vol_ocl_kernel(
    __global const double * restrict volume,
    __global const double * restrict vol_flux_x,
    __global const double * restrict vol_flux_y,
    __global double * restrict pre_vol,
    __global double * restrict post_vol,
    const int mom_sweep)
{

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j<=XMAXPLUSTHREE) && (k<=YMAXPLUSTHREE) ) { 

        if(mom_sweep==1){
              post_vol[ARRAYXY(j,k,XMAXPLUSFIVE)]=volume[ARRAYXY(j,k,XMAXPLUSFOUR)]
                                                  +vol_flux_y[ARRAYXY(j,k+1,XMAXPLUSFOUR)]
                                                  -vol_flux_y[ARRAYXY(j,k  ,XMAXPLUSFOUR)];
              pre_vol[ARRAYXY(j,k,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j,k,XMAXPLUSFIVE)]
                                                 +vol_flux_x[ARRAYXY(j+1,k,XMAXPLUSFIVE)]
                                                 -vol_flux_x[ARRAYXY(j  ,k,XMAXPLUSFIVE)];
        }
        else if(mom_sweep==2){
              post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                                      +vol_flux_x[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                                                      -vol_flux_x[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
              pre_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                                     +vol_flux_y[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]
                                                                     -vol_flux_y[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
        }
        else if(mom_sweep==3){
              post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
              pre_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                                +vol_flux_y[ARRAYXY(j  ,k+1,XMAXPLUSFOUR)]
                                                                -vol_flux_y[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
        }
        else if(mom_sweep==4){
              post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)];
              pre_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                                +vol_flux_x[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]
                                                                -vol_flux_x[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
        }
    }
}

