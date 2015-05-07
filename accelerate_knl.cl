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
 *  @brief OCL device-side acceleration kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details The pressure and viscosity gradients are used to update the
 *  velocity field.
 */

#include "ocl_knls.h"

__kernel void accelerate_ocl_kernel(
    const double dt,
    __global const double2 * restrict xarea,
    __global const double2 * restrict yarea,
    __global const double2 * restrict volume,
    __global const double2 * restrict density0,
    __global const double2 * restrict pressure,
    __global const double2 * restrict viscosity,
    __global const double2 * restrict xvel0,
    __global const double2 * restrict yvel0,
    __global double2 * restrict xvel1,
    __global double2 * restrict yvel1,
    __global double2 * restrict stepbymass)
{
    double2 nodal_mass;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=1) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        density0_tmp_current  = density0[ARRAYXY(j  , k  ,XMAXPLUSFOUR)]; 
        density0_tmp_down     = density0[ARRAYXY(j  , k-1,XMAXPLUSFOUR)]; 
        density0_tmp_left     = density0[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)]; 
        density0_tmp_leftdown = density0[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)]; 

        volume_tmp_current  = volume[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        volume_tmp_down     = volume[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
        volume_tmp_left     = volume[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
        volume_tmp_leftdown = volume[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];

        nodal_mass.y = (density0_tmp_down.x * volume_tmp_down.x
                       +density0_tmp_down.y * volume_tmp_down.y
                       +density0_tmp_current.y * volume_tmp_current.y
                       +density0_tmp_current.x * volume_tmp_current.x)*0.25;

        nodal_mass.x = (density0_tmp_leftdown.y * volume_tmp_leftdown.y
                       +density0_tmp_down.x * volume_tmp_down.x
                       +density0_tmp_current.x * volume_tmp_current.x
                       +density0_tmp_left.y * volume_tmp_left.y) *0.25; 


        stepbymass[ARRAYXY(j,k,XMAXPLUSFIVE)]=0.5*dt/nodal_mass;
        


        //nodal_mass=(density0[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]*volume[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]
        //           +density0[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]*volume[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]
        //           +density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]*volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
        //           +density0[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]*volume[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)])
        //           *0.25;
        //stepbymass[ARRAYXY(j,k,XMAXPLUSFIVE)]=0.5*dt/nodal_mass;

        xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=xvel0[ARRAYXY(j      ,k  ,XMAXPLUSFIVE)]
                                                    -stepbymass[ARRAYXY(j,k  ,XMAXPLUSFIVE)]
                                                    *(xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)]
                                                    *(pressure[ARRAYXY(j ,k  ,XMAXPLUSFOUR)]-pressure[ARRAYXY(j-1,k,XMAXPLUSFOUR)])
                                                    +xarea[ARRAYXY(j     ,k-1,XMAXPLUSFIVE)]
                                                    *(pressure[ARRAYXY(j ,k-1,XMAXPLUSFOUR)]-pressure[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));

        yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=yvel0[ARRAYXY(j      ,k,XMAXPLUSFIVE)]
                                                   -stepbymass[ARRAYXY(j ,k,XMAXPLUSFIVE)]
                                                   *(yarea[ARRAYXY(j     ,k,XMAXPLUSFOUR)]
                                                   *(pressure[ARRAYXY(j  ,k,XMAXPLUSFOUR)]-pressure[ARRAYXY(j,k-1,XMAXPLUSFOUR)])
                                                   +yarea[ARRAYXY(j-1    ,k,XMAXPLUSFOUR)]
                                                   *(pressure[ARRAYXY(j-1,k,XMAXPLUSFOUR)]-pressure[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));

        xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=xvel1[ARRAYXY(j     ,k  ,XMAXPLUSFIVE)]
                                                   -stepbymass[ARRAYXY(j,k  ,XMAXPLUSFIVE)]
                                                   *(xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)]
                                                   *(viscosity[ARRAYXY(j,k  ,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j-1,k,XMAXPLUSFOUR)])
                                                   +xarea[ARRAYXY(j     ,k-1,XMAXPLUSFIVE)]
                                                   *(viscosity[ARRAYXY(j,k-1,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));

        yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=yvel1[ARRAYXY(j       ,k,XMAXPLUSFIVE)]
                                                   -stepbymass[ARRAYXY(j  ,k,XMAXPLUSFIVE)]
                                                   *(yarea[ARRAYXY(j      ,k,XMAXPLUSFOUR)]
                                                   *(viscosity[ARRAYXY(j  ,k,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j,k-1,XMAXPLUSFOUR)])
                                                   +yarea[ARRAYXY(j-1     ,k,XMAXPLUSFOUR)]
                                                   *(viscosity[ARRAYXY(j-1,k,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));
    }
}

