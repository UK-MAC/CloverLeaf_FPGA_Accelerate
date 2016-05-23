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
    double4 nodal_mass, stepbymass, xvel1_tmp, yvel1_tmp, xvel1_output, yvel1_output;
    double4 density0_tmp_current, density0_tmp_down, density0_tmp_left, density0_tmp_leftdown;
    double4 volume_tmp_current, volume_tmp_down, volume_tmp_left, volume_tmp_leftdown;
    double4 pressure_tmp_current, pressure_tmp_down, pressure_tmp_left, pressure_tmp_leftdown;
    double4 xvel0_tmp_current, xarea_tmp_current, xarea_tmp_down;
    double4 yvel0_tmp_current, yarea_tmp_current, yarea_tmp_left;
    double4 viscosity_tmp_current, viscosity_tmp_down, viscosity_tmp_left, viscosity_tmp_leftdown;

    double4 den0_vol_tmp_current, den0_vol_tmp_down, den0_vol_cd_res;
    double den0_vol_tmp_left, den0_vol_tmp_leftdown, den0_vol_lld_res;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (k>=2) && (k<=YMAXPLUSTWO) ) {

        density0_tmp_current  = density0[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        density0_tmp_down     = density0[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];
        volume_tmp_current  = volume[ARRAYXY(j  , k  ,XMAXPLUSFOUR)];
        volume_tmp_down     = volume[ARRAYXY(j  , k-1,XMAXPLUSFOUR)];

        den0_vol_tmp_current  = density0_tmp_current * volume_tmp_current;
        den0_vol_tmp_down     = density0_tmp_down * volume_tmp_down;

        if (j==0) {
            den0_vol_tmp_left     = 0;
            den0_vol_tmp_leftdown = 0;
        } else { 

            density0_tmp_left     = density0[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            density0_tmp_leftdown = density0[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];
            volume_tmp_left       = volume[ARRAYXY(j-1, k  ,XMAXPLUSFOUR)];
            volume_tmp_leftdown   = volume[ARRAYXY(j-1, k-1,XMAXPLUSFOUR)];

            den0_vol_tmp_left     = density0_tmp_left.w * volume_tmp_left.w;
            den0_vol_tmp_leftdown = density0_tmp_leftdown.w * volume_tmp_leftdown.w;
        }

        den0_vol_cd_res = den0_vol_tmp_current + den0_vol_tmp_down;
        den0_vol_lld_res = den0_vol_tmp_left + den0_vol_tmp_leftdown;

        nodal_mass.x = (den0_vol_lld_res + den0_vol_cd_res.x)*0.25; 
        nodal_mass.y = (den0_vol_cd_res.x + den0_vol_cd_res.y)*0.25;
        nodal_mass.z = (den0_vol_cd_res.y + den0_vol_cd_res.z)*0.25;
        nodal_mass.w = (den0_vol_cd_res.z + den0_vol_cd_res.w)*0.25;


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




        xvel1_tmp.w = xvel0_tmp_current.w
                      -stepbymass.w
                      *(xarea_tmp_current.w*(pressure_tmp_current.w - pressure_tmp_current.z)
                       +xarea_tmp_down.w*(pressure_tmp_down.w - pressure_tmp_down.z)
                       );

        xvel1_tmp.z = xvel0_tmp_current.z
                      -stepbymass.z
                      *(xarea_tmp_current.z*(pressure_tmp_current.z - pressure_tmp_current.y)
                       +xarea_tmp_down.z*(pressure_tmp_down.z - pressure_tmp_down.y)
                       );

        xvel1_tmp.y = xvel0_tmp_current.y
                      -stepbymass.y
                      *(xarea_tmp_current.y*(pressure_tmp_current.y - pressure_tmp_current.x)
                       +xarea_tmp_down.y*(pressure_tmp_down.y - pressure_tmp_down.x)
                       );

        xvel1_tmp.x = xvel0_tmp_current.x
                      -stepbymass.x
                      *(xarea_tmp_current.x *(pressure_tmp_current.x - pressure_tmp_left.w)
                       +xarea_tmp_down.x *(pressure_tmp_down.x - pressure_tmp_leftdown.w)
                       );



        double ya_press_tmp_x = yarea_tmp_current.x * (pressure_tmp_current.x - pressure_tmp_down.x);
        double ya_press_tmp_y = yarea_tmp_current.y * (pressure_tmp_current.y - pressure_tmp_down.y);
        double ya_press_tmp_z = yarea_tmp_current.z * (pressure_tmp_current.z - pressure_tmp_down.z);

        yvel1_tmp.w = yvel0_tmp_current.w
                      -stepbymass.w
                      *(yarea_tmp_current.w * (pressure_tmp_current.w - pressure_tmp_down.w) + ya_press_tmp_z);

        yvel1_tmp.z = yvel0_tmp_current.z - stepbymass.z * (ya_press_tmp_z + ya_press_tmp_y);

        yvel1_tmp.y = yvel0_tmp_current.y - stepbymass.y * (ya_press_tmp_y + ya_press_tmp_x);

        yvel1_tmp.x = yvel0_tmp_current.x
                      -stepbymass.x
                      *(ya_press_tmp_x + yarea_tmp_left.w*(pressure_tmp_left.w - pressure_tmp_leftdown.w));





        xvel1_output.w = xvel1_tmp.w
                         -stepbymass.w
                         *(xarea_tmp_current.w*(viscosity_tmp_current.w - viscosity_tmp_current.z)
                          +xarea_tmp_down.w*(viscosity_tmp_down.w - viscosity_tmp_down.z)
                          );
        xvel1_output.z = xvel1_tmp.z
                         -stepbymass.z
                         *(xarea_tmp_current.z*(viscosity_tmp_current.z - viscosity_tmp_current.y)
                          +xarea_tmp_down.z*(viscosity_tmp_down.z - viscosity_tmp_down.y)
                          );
        xvel1_output.y = xvel1_tmp.y
                         -stepbymass.y
                         *(xarea_tmp_current.y*(viscosity_tmp_current.y - viscosity_tmp_current.x)
                          +xarea_tmp_down.y*(viscosity_tmp_down.y - viscosity_tmp_down.x)
                          );
        xvel1_output.x = xvel1_tmp.x
                         - stepbymass.x
                         *(xarea_tmp_current.x*(viscosity_tmp_current.x - viscosity_tmp_left.w)
                          +xarea_tmp_down.x*(viscosity_tmp_down.x - viscosity_tmp_leftdown.w)
                          );

        double ya_vis_tmp_x = yarea_tmp_current.x*(viscosity_tmp_current.x - viscosity_tmp_down.x);
        double ya_vis_tmp_y = yarea_tmp_current.y*(viscosity_tmp_current.y - viscosity_tmp_down.y);
        double ya_vis_tmp_z = yarea_tmp_current.z*(viscosity_tmp_current.z - viscosity_tmp_down.z);

        yvel1_output.w = yvel1_tmp.w
                         -stepbymass.w
                         *(yarea_tmp_current.w*(viscosity_tmp_current.w - viscosity_tmp_down.w) + ya_vis_tmp_z);

        yvel1_output.z = yvel1_tmp.z - stepbymass.z *(ya_vis_tmp_z + ya_vis_tmp_y);

        yvel1_output.y = yvel1_tmp.y - stepbymass.y *(ya_vis_tmp_y + ya_vis_tmp_x);

        yvel1_output.x = yvel1_tmp.x
                         -stepbymass.x
                         *(ya_vis_tmp_x + yarea_tmp_left.w*(viscosity_tmp_left.w - viscosity_tmp_leftdown.w));





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

