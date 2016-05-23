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

__kernel void advec_mom_vel_ocl_kernel_y(
    __global const double * restrict node_mass_post,
    __global const double * restrict node_mass_pre,
    __global const double * restrict mom_flux,
    __global double * restrict vel1)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=(vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        *node_mass_pre[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                        +mom_flux[ARRAYXY(j  ,k-1,XMAXPLUSFIVE)]
                                                        -mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])
                                                        /node_mass_post[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];

    }
}
