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
    __global const double2 * restrict g_xarea,
    __global const double2 * restrict g_yarea,
    __global const double2 * restrict g_volume,
    __global const double2 * restrict g_density0,
    __global const double2 * restrict g_pressure,
    __global const double2 * restrict g_viscosity,
    __global const double2 * restrict g_xvel0,
    __global const double2 * restrict g_yvel0,
    __global double2 * restrict g_xvel1,
    __global double2 * restrict g_yvel1)
    //__global double2 * restrict stepbymass
{
    __global double2 (*xarea)[XMAXPLUSFIVE] = (__global double2 (*)[XMAXPLUSFIVE])g_xarea;
    __global double2 (*yarea)[XMAXPLUSFOUR] = (__global double2 (*)[XMAXPLUSFOUR])g_yarea;
    __global double2 (*volume)[XMAXPLUSFOUR] = (__global double2 (*)[XMAXPLUSFOUR])g_volume;
    __global double2 (*density0)[XMAXPLUSFOUR] = (__global double2 (*)[XMAXPLUSFOUR])g_density0;
    __global double2 (*pressure)[XMAXPLUSFOUR] = (__global double2 (*)[XMAXPLUSFOUR])g_pressure;
    __global double2 (*viscosity)[XMAXPLUSFOUR] = (__global double2 (*)[XMAXPLUSFOUR])g_viscosity;

    __global double2 (*xvel0)[XMAXPLUSFIVE] = (__global double2 (*)[XMAXPLUSFIVE])g_xvel0;
    __global double2 (*yvel0)[XMAXPLUSFIVE] = (__global double2 (*)[XMAXPLUSFIVE])g_yvel0;
    __global double2 (*xvel1)[XMAXPLUSFIVE] = (__global double2 (*)[XMAXPLUSFIVE])g_xvel1;
    __global double2 (*yvel1)[XMAXPLUSFIVE] = (__global double2 (*)[XMAXPLUSFIVE])g_yvel1;


    double2 nodal_mass, stepbymass, xvel1_tmp, yvel1_tmp, xvel1_output, yvel1_output;
    double2 density0_tmp_current, density0_tmp_down, density0_tmp_left, density0_tmp_leftdown;
    double2 volume_tmp_current, volume_tmp_down, volume_tmp_left, volume_tmp_leftdown;
    double2 pressure_tmp_current, pressure_tmp_down, pressure_tmp_left, pressure_tmp_leftdown;
    double2 xvel0_tmp_current, xarea_tmp_current, xarea_tmp_down;
    double2 yvel0_tmp_current, yarea_tmp_current, yarea_tmp_left;
    double2 viscosity_tmp_current, viscosity_tmp_down, viscosity_tmp_left, viscosity_tmp_leftdown; 
    double2 den0_vol_tmp_down_res, den0_vol_tmp_current_res;
    double2 xvel1_last, yvel1_last; 


    for (unsigned k=2; k<=YMAXPLUSTWO; k++) {
        for (unsigned j=1; j<=XLIMIT; j++) {

            density0_tmp_current  = density0[k  ][j  ]; 
            density0_tmp_down     = density0[k-1][j  ]; 
            density0_tmp_left     = density0[k  ][j-1]; 
            density0_tmp_leftdown = density0[k-1][j-1]; 

            volume_tmp_current  = volume[k  ][j  ];
            volume_tmp_down     = volume[k-1][j  ];
            volume_tmp_left     = volume[k  ][j-1];
            volume_tmp_leftdown = volume[k-1][j-1];

            pressure_tmp_current  = pressure[k  ][j  ];
            pressure_tmp_down     = pressure[k-1][j  ];
            pressure_tmp_left     = pressure[k  ][j-1];
            pressure_tmp_leftdown = pressure[k-1][j-1]; 

            xvel0_tmp_current    = xvel0[k][j];
            xarea_tmp_current    = xarea[k][j];
            xarea_tmp_down       = xarea[k-1][j];

            yvel0_tmp_current = yvel0[k][j]; 
            yarea_tmp_current = yarea[k][j];
            yarea_tmp_left    = yarea[k][j-1];

            viscosity_tmp_current  = viscosity[k  ][j  ]; 
            viscosity_tmp_down     = viscosity[k-1][j  ]; 
            viscosity_tmp_left     = viscosity[k  ][j-1]; 
            viscosity_tmp_leftdown = viscosity[k-1][j-1]; 

            if (j == XLIMIT) {
                xvel1_last = xvel1[k][j];
                yvel1_last = yvel1[k][j];
            } 

            den0_vol_tmp_down_res    = density0_tmp_down * volume_tmp_down; 
            den0_vol_tmp_current_res = density0_tmp_current * volume_tmp_current;


            nodal_mass.y = (den0_vol_tmp_down_res.x
                           +den0_vol_tmp_down_res.y
                           +den0_vol_tmp_current_res.y
                           +den0_vol_tmp_current_res.x)*0.25;

            nodal_mass.x = (density0_tmp_leftdown.y * volume_tmp_leftdown.y
                           +den0_vol_tmp_down_res.x 
                           +den0_vol_tmp_current_res.x
                           +density0_tmp_left.y * volume_tmp_left.y)*0.25; 

            stepbymass=0.5*dt/nodal_mass;
            





            xvel1_tmp.y = xvel0_tmp_current.y
                          -stepbymass.y
                          *(xarea_tmp_current.y
                           *(pressure_tmp_current.y - pressure_tmp_current.x)
                            +xarea_tmp_down.y
                            *(pressure_tmp_down.y - pressure_tmp_down.x)
                           );

            xvel1_tmp.x = xvel0_tmp_current.x
                          -stepbymass.x
                          *(xarea_tmp_current.x
                           *(pressure_tmp_current.x - pressure_tmp_left.y)
                            +xarea_tmp_down.x
                            *(pressure_tmp_down.x - pressure_tmp_leftdown.y)
                           );

            double ya_press_tmp_x = yarea_tmp_current.x * (pressure_tmp_current.x - pressure_tmp_down.x); 

            yvel1_tmp.y = yvel0_tmp_current.y 
                          - stepbymass.y * (yarea_tmp_current.y * (pressure_tmp_current.y - pressure_tmp_down.y) + ya_press_tmp_x);

            yvel1_tmp.x = yvel0_tmp_current.x 
                          - stepbymass.x * (ya_press_tmp_x + yarea_tmp_left.y * (pressure_tmp_left.y - pressure_tmp_leftdown.y));






 


            xvel1_output.y = xvel1_tmp.y
                             - stepbymass.y
                               *(xarea_tmp_current.y
                                 *(viscosity_tmp_current.y - viscosity_tmp_current.x)
                                   + xarea_tmp_down.y 
                                     *(viscosity_tmp_down.y - viscosity_tmp_down.x)
                                );

            xvel1_output.x = xvel1_tmp.x
                             - stepbymass.x
                               *(xarea_tmp_current.x 
                                 *(viscosity_tmp_current.x - viscosity_tmp_left.y)
                                   + xarea_tmp_down.x 
                                     *(viscosity_tmp_down.x - viscosity_tmp_leftdown.y)
                                );

            double ya_vis_tmp_x = yarea_tmp_current.x * (viscosity_tmp_current.x - viscosity_tmp_down.x);


            yvel1_output.y = yvel1_tmp.y
                             - stepbymass.y *(yarea_tmp_current.y *(viscosity_tmp_current.y - viscosity_tmp_down.y) + ya_vis_tmp_x);

            yvel1_output.x = yvel1_tmp.x
                             - stepbymass.x *(ya_vis_tmp_x + yarea_tmp_left.y *(viscosity_tmp_left.y - viscosity_tmp_leftdown.y) );


            //write results to memory, masked for final element 
            if (j == XLIMIT) {
                xvel1_output.y = xvel1_last.y;
                yvel1_output.y = yvel1_last.y; 
            }

            xvel1[k][j] = xvel1_output;
            yvel1[k][j] = yvel1_output; 

        }
    }
}

