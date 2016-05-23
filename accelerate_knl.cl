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
    __global const double8 * restrict xarea,
    __global const double8 * restrict yarea,
    __global const double8 * restrict volume,
    __global const double8 * restrict density0,
    __global const double8 * restrict pressure,
    __global const double8 * restrict viscosity,
    __global const double8 * restrict xvel0,
    __global const double8 * restrict yvel0,
    __global double8 * restrict xvel1,
    __global double8 * restrict yvel1)
    //__global double4 * restrict stepbymass
{
    double8 nodal_mass, nodal_mass_tmp, stepbymass, xvel1_tmp, yvel1_tmp, xvel1_output, yvel1_output;
    double8 density0_tmp_current, density0_tmp_down, density0_tmp_left, density0_tmp_leftdown;
    double8 volume_tmp_current, volume_tmp_down, volume_tmp_left, volume_tmp_leftdown;
    double8 pressure_tmp_current, pressure_tmp_down, pressure_tmp_left, pressure_tmp_leftdown;
    double8 xvel0_tmp_current, xarea_tmp_current, xarea_tmp_down;
    double8 yvel0_tmp_current, yarea_tmp_current, yarea_tmp_left;
    double8 viscosity_tmp_current, viscosity_tmp_down, viscosity_tmp_left, viscosity_tmp_leftdown;

    double8 den0_vol_tmp_current, den0_vol_tmp_down, den0_vol_cd_res;
    double den0_vol_tmp_left, den0_vol_tmp_leftdown, den0_vol_lld_res;
    double8 press_tmp_curr_rShift, press_tmp_down_rShift, yarea_tmp_curr_rShift; 
    double8 vis_tmp_curr_rShift, vis_tmp_down_rShift; 

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
            den0_vol_tmp_left     = density0_tmp_left.s7 * volume_tmp_left.s7;
            den0_vol_tmp_leftdown = density0_tmp_leftdown.s7 * volume_tmp_leftdown.s7;
        }

        den0_vol_cd_res = den0_vol_tmp_current + den0_vol_tmp_down;
        den0_vol_lld_res = den0_vol_tmp_left + den0_vol_tmp_leftdown;

        nodal_mass_tmp.s0 = den0_vol_lld_res + den0_vol_cd_res.s0; 
        nodal_mass_tmp.s1 = den0_vol_cd_res.s0 + den0_vol_cd_res.s1; 
        nodal_mass_tmp.s2 = den0_vol_cd_res.s1 + den0_vol_cd_res.s2;
        nodal_mass_tmp.s3 = den0_vol_cd_res.s2 + den0_vol_cd_res.s3;
        nodal_mass_tmp.s4 = den0_vol_cd_res.s3 + den0_vol_cd_res.s4;
        nodal_mass_tmp.s5 = den0_vol_cd_res.s4 + den0_vol_cd_res.s5;
        nodal_mass_tmp.s6 = den0_vol_cd_res.s5 + den0_vol_cd_res.s6;
        nodal_mass_tmp.s7 = den0_vol_cd_res.s6 + den0_vol_cd_res.s7;

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


        press_tmp_curr_rShift.s0 = pressure_tmp_left.s7;  
        press_tmp_curr_rShift.s1 = pressure_tmp_current.s0;
        press_tmp_curr_rShift.s2 = pressure_tmp_current.s1;
        press_tmp_curr_rShift.s3 = pressure_tmp_current.s2;
        press_tmp_curr_rShift.s4 = pressure_tmp_current.s3;
        press_tmp_curr_rShift.s5 = pressure_tmp_current.s4;
        press_tmp_curr_rShift.s6 = pressure_tmp_current.s5;
        press_tmp_curr_rShift.s7 = pressure_tmp_current.s6;

        press_tmp_down_rShift.s0 = pressure_tmp_leftdown.s7; 
        press_tmp_down_rShift.s1 = pressure_tmp_down.s0;
        press_tmp_down_rShift.s2 = pressure_tmp_down.s1;
        press_tmp_down_rShift.s3 = pressure_tmp_down.s2;
        press_tmp_down_rShift.s4 = pressure_tmp_down.s3;
        press_tmp_down_rShift.s5 = pressure_tmp_down.s4;
        press_tmp_down_rShift.s6 = pressure_tmp_down.s5;
        press_tmp_down_rShift.s7 = pressure_tmp_down.s6;

        yarea_tmp_curr_rShift.s0 = yarea_tmp_left.s7; 
        yarea_tmp_curr_rShift.s1 = yarea_tmp_current.s0;
        yarea_tmp_curr_rShift.s2 = yarea_tmp_current.s1; 
        yarea_tmp_curr_rShift.s3 = yarea_tmp_current.s2;
        yarea_tmp_curr_rShift.s4 = yarea_tmp_current.s3;
        yarea_tmp_curr_rShift.s5 = yarea_tmp_current.s4;
        yarea_tmp_curr_rShift.s6 = yarea_tmp_current.s5;
        yarea_tmp_curr_rShift.s7 = yarea_tmp_current.s6;


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


        vis_tmp_curr_rShift.s0 = viscosity_tmp_left.s7; 
        vis_tmp_curr_rShift.s1 = viscosity_tmp_current.s0;
        vis_tmp_curr_rShift.s2 = viscosity_tmp_current.s1;
        vis_tmp_curr_rShift.s3 = viscosity_tmp_current.s2;
        vis_tmp_curr_rShift.s4 = viscosity_tmp_current.s3;
        vis_tmp_curr_rShift.s5 = viscosity_tmp_current.s4;
        vis_tmp_curr_rShift.s6 = viscosity_tmp_current.s5;
        vis_tmp_curr_rShift.s7 = viscosity_tmp_current.s6;

        vis_tmp_down_rShift.s0 = viscosity_tmp_leftdown.s7; 
        vis_tmp_down_rShift.s1 = viscosity_tmp_down.s0;
        vis_tmp_down_rShift.s2 = viscosity_tmp_down.s1;
        vis_tmp_down_rShift.s3 = viscosity_tmp_down.s2;
        vis_tmp_down_rShift.s4 = viscosity_tmp_down.s3;
        vis_tmp_down_rShift.s5 = viscosity_tmp_down.s4;
        vis_tmp_down_rShift.s6 = viscosity_tmp_down.s5;
        vis_tmp_down_rShift.s7 = viscosity_tmp_down.s6;


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
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s2 = xvel1_output.s2;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s3 = xvel1_output.s3;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s4 = xvel1_output.s4;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s5 = xvel1_output.s5;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s6 = xvel1_output.s6;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s7 = xvel1_output.s7;

            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s2 = yvel1_output.s2;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s3 = yvel1_output.s3;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s4 = yvel1_output.s4;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s5 = yvel1_output.s5;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s6 = yvel1_output.s6;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s7 = yvel1_output.s7;
        }
        else if (j == XLIMIT) {
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s0 = xvel1_output.s0;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s1 = xvel1_output.s1;
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s2 = xvel1_output.s2;

            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s0 = yvel1_output.s0;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s1 = yvel1_output.s1;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].s2 = yvel1_output.s2;
        }
        else {
            // else write the full vector back 
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)] = xvel1_output;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)] = yvel1_output;
        }
    }
}

