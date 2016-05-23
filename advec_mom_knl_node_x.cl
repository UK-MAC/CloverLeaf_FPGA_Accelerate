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

__kernel void advec_mom_node_ocl_kernel_x(
    __global const double * restrict mass_flux_x,
    __global double * restrict node_flux,
    __global const double * restrict density1,
    __global const double * restrict post_vol,
    __global double * restrict node_mass_post)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j<=XMAXPLUSTHREE) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)] = 0.25*(mass_flux_x[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                        +mass_flux_x[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        +mass_flux_x[ARRAYXY(j+1,k-1,XMAXPLUSFIVE)]
                                                        +mass_flux_x[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]);
    }

    if ( (j>=1) && (j<=XMAXPLUSTHREE) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]= 0.25*( density1[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                             +density1[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                             +density1[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j-1,k-1,XMAXPLUSFIVE)]
                                                             +density1[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]
                                                             *post_vol[ARRAYXY(j-1,k  ,XMAXPLUSFIVE)]);
    }

}

