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
    __global const double4 * restrict xarea,
    __global const double4 * restrict yarea,
    __global const double4 * restrict volume,
    __global const double4 * restrict density0,
    __global const double4 * restrict pressure,
    __global const double4 * restrict viscosity,
    __global const double4 * restrict xvel0,
    __global const double4 * restrict yvel0,
    __global double4 * restrict xvel1,
    __global double4 * restrict yvel1)
    //__global double4 * restrict stepbymass
{
    double4 nodal_mass, nodal_mass_tmp, stepbymass, xvel1_tmp, yvel1_tmp, xvel1_output, yvel1_output;
    double4 density0_tmp_current, density0_tmp_down, density0_tmp_left, density0_tmp_leftdown;
    double4 volume_tmp_current, volume_tmp_down, volume_tmp_left, volume_tmp_leftdown;
    double4 pressure_tmp_current, pressure_tmp_down, pressure_tmp_left, pressure_tmp_leftdown;
    double4 xvel0_tmp_current, xarea_tmp_current, xarea_tmp_down;
    double4 yvel0_tmp_current, yarea_tmp_current, yarea_tmp_left;
    double4 viscosity_tmp_current, viscosity_tmp_down, viscosity_tmp_left, viscosity_tmp_leftdown;

    double4 den0_vol_tmp_current, den0_vol_tmp_down, den0_vol_cd_res;
    double den0_vol_tmp_left, den0_vol_tmp_leftdown, den0_vol_lld_res;
    double4 press_tmp_curr_rShift, press_tmp_down_rShift, yarea_tmp_curr_rShift; 
    double4 vis_tmp_curr_rShift, vis_tmp_down_rShift; 

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (k>=2) && (k<=YMAXPLUSTWO) ) {
        // (j>=2) && (j<=XMAXPLUSTWO) && 

        density0_tmp_current  = density0[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        density0_tmp_down     = density0[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
        volume_tmp_current  = volume[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        volume_tmp_down     = volume[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];

        if (j > 0) {
            density0_tmp_left     = density0[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            density0_tmp_leftdown = density0[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
            volume_tmp_left       = volume[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            volume_tmp_leftdown   = volume[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
        }

        den0_vol_tmp_current  = density0_tmp_current * volume_tmp_current;
        den0_vol_tmp_down     = density0_tmp_down * volume_tmp_down;

        if (j==0) {
            den0_vol_tmp_left     = 0;
            den0_vol_tmp_leftdown = 0;
        } else { 
            den0_vol_tmp_left     = density0_tmp_left.w * volume_tmp_left.w;
            den0_vol_tmp_leftdown = density0_tmp_leftdown.w * volume_tmp_leftdown.w;
        }

        den0_vol_cd_res = den0_vol_tmp_current + den0_vol_tmp_down;
        den0_vol_lld_res = den0_vol_tmp_left + den0_vol_tmp_leftdown;

        nodal_mass_tmp.x = den0_vol_lld_res + den0_vol_cd_res.x; 
        nodal_mass_tmp.y = den0_vol_cd_res.x + den0_vol_cd_res.y;
        nodal_mass_tmp.z = den0_vol_cd_res.y + den0_vol_cd_res.z;
        nodal_mass_tmp.w = den0_vol_cd_res.z + den0_vol_cd_res.w;

        nodal_mass = nodal_mass_tmp * 0.25; 

        //nodal_mass=(density0[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]*volume[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]
        //           +density0[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]*volume[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]
        //           +density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]*volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
        //           +density0[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]*volume[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)])
        //           *0.25;

        stepbymass=0.5*dt/nodal_mass;



        xvel0_tmp_current = xvel0[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)];
        xarea_tmp_current = xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)];
        xarea_tmp_down    = xarea[ARRAYXY(j    ,k-1,XMAXPLUSFIVE)];

        yvel0_tmp_current = yvel0[ARRAYXY(j     ,k,XMAXPLUSFIVE)];
        yarea_tmp_current = yarea[ARRAYXY(j     ,k,XMAXPLUSFOUR)];
        yarea_tmp_left    = yarea[ARRAYXY(j-1   ,k,XMAXPLUSFOUR)];

        pressure_tmp_current   = pressure[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        pressure_tmp_down      = pressure[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
        pressure_tmp_left      = pressure[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
        pressure_tmp_leftdown  = pressure[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];

        viscosity_tmp_current  = viscosity[ARRAYXY(j,k  ,XMAXPLUSFOUR)];
        viscosity_tmp_down     = viscosity[ARRAYXY(j,k-1,XMAXPLUSFOUR)];
        viscosity_tmp_left     = viscosity[ARRAYXY(j-1,k,XMAXPLUSFOUR)];
        viscosity_tmp_leftdown = viscosity[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)];


        press_tmp_curr_rShift.x = pressure_tmp_left.w;  
        press_tmp_curr_rShift.y = pressure_tmp_current.x;
        press_tmp_curr_rShift.z = pressure_tmp_current.y;
        press_tmp_curr_rShift.w = pressure_tmp_current.z;

        press_tmp_down_rShift.x = pressure_tmp_leftdown.w; 
        press_tmp_down_rShift.y = pressure_tmp_down.x;
        press_tmp_down_rShift.z = pressure_tmp_down.y;
        press_tmp_down_rShift.w = pressure_tmp_down.z;

        yarea_tmp_curr_rShift.x = yarea_tmp_left.w; 
        yarea_tmp_curr_rShift.y = yarea_tmp_current.x;
        yarea_tmp_curr_rShift.z = yarea_tmp_current.y; 
        yarea_tmp_curr_rShift.w = yarea_tmp_current.z;


        xvel1_tmp = xvel0_tmp_current
                    -stepbymass
                    *(xarea_tmp_current*(pressure_tmp_current - press_tmp_curr_rShift)
                     +xarea_tmp_down*(pressure_tmp_down - press_tmp_down_rShift)
                     );


        yvel1_tmp = yvel0_tmp_current
                    -stepbymass
                    *(yarea_tmp_current*(pressure_tmp_current - pressure_tmp_down)
                     +yarea_tmp_curr_rShift*(press_tmp_curr_rShift-press_tmp_down_rShift)
                     );


        vis_tmp_curr_rShift.x = viscosity_tmp_left.w; 
        vis_tmp_curr_rShift.y = viscosity_tmp_current.x;
        vis_tmp_curr_rShift.z = viscosity_tmp_current.y;
        vis_tmp_curr_rShift.w = viscosity_tmp_current.z;

        vis_tmp_down_rShift.x = viscosity_tmp_leftdown.w;
        vis_tmp_down_rShift.y = viscosity_tmp_down.x;
        vis_tmp_down_rShift.z = viscosity_tmp_down.y;
        vis_tmp_down_rShift.w = viscosity_tmp_down.z;


        xvel1_output = xvel1_tmp 
                       -stepbymass
                       *(xarea_tmp_current*(viscosity_tmp_current - vis_tmp_curr_rShift)
                        +xarea_tmp_down*(viscosity_tmp_down - vis_tmp_down_rShift)
                        );

        yvel1_output = yvel1_tmp 
                       -stepbymass
                       *(yarea_tmp_current*(viscosity_tmp_current - viscosity_tmp_down)
                        +yarea_tmp_curr_rShift*(vis_tmp_curr_rShift - vis_tmp_down_rShift)
                        );

        if (j==0) {
            // only send z and w back to memory
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].z = xvel1_output.z;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].w = xvel1_output.w;

            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].z = yvel1_output.z;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].w = yvel1_output.w;
        }
        else if (j == XLIMIT) {
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].x = xvel1_output.x;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].y = xvel1_output.y;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].z = xvel1_output.z;

            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].x = yvel1_output.x;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].y = yvel1_output.y;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].z = yvel1_output.z;
        }
        else {
            // else write the full vector back 
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)] = xvel1_output;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)] = yvel1_output;
        }
    }
}

