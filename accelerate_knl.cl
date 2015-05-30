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

__attribute((reqd_work_group_size(8,16,1)))
__attribute((max_work_group_size(128)))
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
    __global double2 * restrict yvel1)
    //__global double2 * restrict stepbymass
{
    double2 nodal_mass, stepbymass, xvel1_tmp, yvel1_tmp, xvel1_output, yvel1_output;
    double2 xvel0_tmp_current, yvel0_tmp_current; 
    double2 den0_vol_tmp_down_res, den0_vol_tmp_current_res;

    //double2 density0_tmp_current, density0_tmp_down, density0_tmp_left, density0_tmp_leftdown;
    //double2 volume_tmp_current, volume_tmp_down, volume_tmp_left, volume_tmp_leftdown;
    //double2 pressure_tmp_current, pressure_tmp_down, pressure_tmp_left, pressure_tmp_leftdown;
    //double2 yarea_tmp_current, yarea_tmp_left, xarea_tmp_current, xarea_tmp_down;
    //double2 viscosity_tmp_current, viscosity_tmp_down, viscosity_tmp_left, viscosity_tmp_leftdown; 


    __local double2 density0_cache[17][9];
    __local double2 volume_cache[17][9];
    __local double2 pressure_cache[17][9];
    __local double2 viscosity_cache[17][9];
    __local double2 xarea_cache[17][8];
    __local double2 yarea_cache[16][9];

    int k = get_global_id(1);
    int j = get_global_id(0);

    int lk = get_local_id(1);
    int lj = get_local_id(0);

    if ( (j<=XLIMIT) && (k>=1) && (k<=YMAXPLUSTWO) ) {

        if ( (lj == 0) && (j != 0)) {
            density0_cache[lk+1][0]  =  density0[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            volume_cache[lk+1][0]    =    volume[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            pressure_cache[lk+1][0]  =  pressure[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            viscosity_cache[lk+1][0] = viscosity[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];

            yarea_cache[lk][lj] = yarea[ARRAYXY(j-1   ,k,XMAXPLUSFOUR)];
        }
        if ( (lk == 0) && (k != 0)) {
            density0_cache[lk][lj+1]  =  density0[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
            volume_cache[lk][lj+1]    =    volume[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
            pressure_cache[lk][lj+1]  =  pressure[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
            viscosity_cache[lk][lj+1] = viscosity[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];

            xarea_cache[lk][lj] = xarea[ARRAYXY(j    ,k-1,XMAXPLUSFIVE)];
        }
        if ((lj==0) && (j!=0) && (lk==0) && (k!=0)) {
            density0_cache[0][0]  =  density0[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
            volume_cache[0][0]    =    volume[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
            pressure_cache[0][0]  =  pressure[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
            viscosity_cache[0][0] = viscosity[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
        }

        density0_cache[lk+1][lj+1]  =  density0[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        volume_cache[lk+1][lj+1]    =    volume[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        pressure_cache[lk+1][lj+1]  =  pressure[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        viscosity_cache[lk+1][lj+1] = viscosity[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];

        xarea_cache[lk+1][lj] = xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)];

        yarea_cache[lk][lj+1] = yarea[ARRAYXY(j     ,k,XMAXPLUSFOUR)];
    }


    //barrier(CLK_LOCAL_MEM_FENCE);


    if ( (j>=1) && (j<=XLIMIT) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        //density0_tmp_current  = density0[ARRAYXY(j  , k  ,XMAXPLUSFOUR)]; 
        //density0_tmp_down     = density0[ARRAYXY(j  , k-1,XMAXPLUSFOUR)]; 
        //density0_tmp_left     = density0[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)]; 
        //density0_tmp_leftdown = density0[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)]; 

        //volume_tmp_current  = volume[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        //volume_tmp_down     = volume[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
        //volume_tmp_left     = volume[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
        //volume_tmp_leftdown = volume[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];

        //pressure_tmp_current  = pressure[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        //pressure_tmp_down     = pressure[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
        //pressure_tmp_left     = pressure[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
        //pressure_tmp_leftdown = pressure[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)]; 

        //viscosity_tmp_current  = viscosity[ARRAYXY(j,k  ,XMAXPLUSFOUR)]; 
        //viscosity_tmp_down     = viscosity[ARRAYXY(j,k-1,XMAXPLUSFOUR)]; 
        //viscosity_tmp_left     = viscosity[ARRAYXY(j-1,k,XMAXPLUSFOUR)]; 
        //viscosity_tmp_leftdown = viscosity[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]; 

        xvel0_tmp_current = xvel0[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)];
        yvel0_tmp_current = yvel0[ARRAYXY(j     ,k,XMAXPLUSFIVE)]; 

        //xarea_tmp_current = xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)];
        //xarea_tmp_down    = xarea[ARRAYXY(j    ,k-1,XMAXPLUSFIVE)];

        //yarea_tmp_current = yarea[ARRAYXY(j     ,k,XMAXPLUSFOUR)];
        //yarea_tmp_left    = yarea[ARRAYXY(j-1   ,k,XMAXPLUSFOUR)];


        //den0_vol_tmp_down_res    = density0_tmp_down * volume_tmp_down; 
        //den0_vol_tmp_current_res = density0_tmp_current * volume_tmp_current;
        den0_vol_tmp_down_res    = density0_cache[lk][lj+1] * volume_cache[lk][lj+1];  
        den0_vol_tmp_current_res = density0_cache[lk+1][lj+1] * volume_cache[lk+1][lj+1]; 


        nodal_mass.y = (den0_vol_tmp_down_res.x + den0_vol_tmp_down_res.y + den0_vol_tmp_current_res.y + den0_vol_tmp_current_res.x)*0.25;

        //nodal_mass.x = (density0_tmp_leftdown.y * volume_tmp_leftdown.y + den0_vol_tmp_down_res.x + den0_vol_tmp_current_res.x + density0_tmp_left.y * volume_tmp_left.y)*0.25; 
        nodal_mass.x = (density0_cache[lk][lj].y * volume_cache[lk][lj].y + den0_vol_tmp_down_res.x + den0_vol_tmp_current_res.x + density0_cache[lk+1][lj].y * volume_cache[lk+1][lj].y)*0.25; 

        stepbymass=0.5*dt/nodal_mass;
        





        //xvel1_tmp.y = xvel0_tmp_current.y - stepbymass.y 
        //              *(xarea_tmp_current.y * (pressure_tmp_current.y - pressure_tmp_current.x) + xarea_tmp_down.y * (pressure_tmp_down.y - pressure_tmp_down.x) );
        xvel1_tmp.y = xvel0_tmp_current.y - stepbymass.y 
                      *(xarea_cache[lk+1][lj].y * (pressure_cache[lk+1][lj+1].y - pressure_cache[lk+1][lj+1].x) + xarea_cache[lk][lj].y * (pressure_cache[lk][lj+1].y - pressure_cache[lk][lj+1].x) );



        //xvel1_tmp.x = xvel0_tmp_current.x - stepbymass.x
        //              *(xarea_tmp_current.x * (pressure_tmp_current.x - pressure_tmp_left.y) + xarea_tmp_down.x *(pressure_tmp_down.x - pressure_tmp_leftdown.y) );
        xvel1_tmp.x = xvel0_tmp_current.x - stepbymass.x
                      *(xarea_cache[lk+1][lj].x * (pressure_cache[lk+1][lj+1].x - pressure_cache[lk+1][lj].y) + xarea_cache[lk][lj].x *(pressure_cache[lk][lj+1].x - pressure_cache[lk][lj].y) );


        //double ya_press_tmp_x = yarea_tmp_current.x * (pressure_tmp_current.x - pressure_tmp_down.x); 
        double ya_press_tmp_x = yarea_cache[lk][lj+1].x * (pressure_cache[lk+1][lj+1].x - pressure_cache[lk][lj+1].x); 

        //yvel1_tmp.y = yvel0_tmp_current.y - stepbymass.y * (yarea_tmp_current.y * (pressure_tmp_current.y - pressure_tmp_down.y) + ya_press_tmp_x);
        yvel1_tmp.y = yvel0_tmp_current.y - stepbymass.y * (yarea_cache[lk][lj+1].y * (pressure_cache[lk+1][lj+1].y - pressure_cache[lk][lj+1].y) + ya_press_tmp_x);

        //yvel1_tmp.x = yvel0_tmp_current.x - stepbymass.x * (ya_press_tmp_x + yarea_tmp_left.y * (pressure_tmp_left.y - pressure_tmp_leftdown.y));
        yvel1_tmp.x = yvel0_tmp_current.x - stepbymass.x * (ya_press_tmp_x + yarea_cache[lk][lj].y * (pressure_cache[lk+1][lj].y - pressure_cache[lk][lj].y));






 


        //xvel1_output.y = xvel1_tmp.y - stepbymass.y
        //                 *(xarea_tmp_current.y *(viscosity_tmp_current.y - viscosity_tmp_current.x) + xarea_tmp_down.y *(viscosity_tmp_down.y - viscosity_tmp_down.x));
        xvel1_output.y = xvel1_tmp.y - stepbymass.y
                         *(xarea_cache[lk+1][lj].y *(viscosity_cache[lk+1][lj+1].y - viscosity_cache[lk+1][lj+1].x) + xarea_cache[lk][lj].y *(viscosity_cache[lk][lj+1].y - viscosity_cache[lk][lj+1].x));

        //xvel1_output.x = xvel1_tmp.x - stepbymass.x
        //                 *(xarea_tmp_current.x *(viscosity_tmp_current.x - viscosity_tmp_left.y) + xarea_tmp_down.x *(viscosity_tmp_down.x - viscosity_tmp_leftdown.y));
        xvel1_output.x = xvel1_tmp.x - stepbymass.x
                         *(xarea_cache[lk+1][lj].x *(viscosity_cache[lk+1][lj+1].x - viscosity_cache[lk+1][lj].y) + xarea_cache[lk][lj].x *(viscosity_cache[lk][lj+1].x - viscosity_cache[lk][lj].y));



        //double ya_vis_tmp_x = yarea_tmp_current.x * (viscosity_tmp_current.x - viscosity_tmp_down.x);
        double ya_vis_tmp_x = yarea_cache[lk][lj+1].x * (viscosity_cache[lk+1][lj+1].x - viscosity_cache[lk][lj+1].x);



        //yvel1_output.y = yvel1_tmp.y - stepbymass.y *(yarea_tmp_current.y *(viscosity_tmp_current.y - viscosity_tmp_down.y) + ya_vis_tmp_x);
        yvel1_output.y = yvel1_tmp.y - stepbymass.y *(yarea_cache[lk][lj+1].y *(viscosity_cache[lk+1][lj+1].y - viscosity_cache[lk][lj+1].y) + ya_vis_tmp_x);



        //yvel1_output.x = yvel1_tmp.x - stepbymass.x *(ya_vis_tmp_x + yarea_tmp_left.y *(viscosity_tmp_left.y - viscosity_tmp_leftdown.y) );
        yvel1_output.x = yvel1_tmp.x - stepbymass.x *(ya_vis_tmp_x + yarea_cache[lk][lj].y *(viscosity_cache[lk+1][lj].y - viscosity_cache[lk][lj].y) );



        //write results to memory, masked for final element 
        if (j == XLIMIT) {
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].x = xvel1_output.x;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)].x = yvel1_output.x; 
        } else {
            xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)] = xvel1_output;
            yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)] = yvel1_output; 
        }

    }
}

