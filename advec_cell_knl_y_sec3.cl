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
 *  @brief OCL device-side advection cell kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
 */

#include "ocl_knls.h"

__kernel void advec_cell_ydir_section3_kernel(
    __global double * restrict density1,    
    __global double * restrict energy1,     
    __global const double * restrict mass_flux_y, 
    __global const double * restrict vol_flux_y,  
    __global const double * restrict pre_vol,     
    __global double * restrict pre_mass,    
    __global double * restrict post_mass,   
    __global double * restrict advec_vol,   
    __global double * restrict post_ener,   
    __global const double * restrict ener_flux)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE)) {
    
    
        pre_mass[ARRAYXY(j,k,XMAXPLUSFIVE)] = density1[ARRAYXY(j,k,XMAXPLUSFOUR)] * pre_vol[ARRAYXY(j,k,XMAXPLUSFIVE)];
        
        post_mass[ARRAYXY(j,k,XMAXPLUSFIVE)] = pre_mass[ARRAYXY(j,k,XMAXPLUSFIVE)] 
                                               + mass_flux_y[ARRAYXY(j, k  , XMAXPLUSFOUR)] 
                                               - mass_flux_y[ARRAYXY(j, k+1, XMAXPLUSFOUR)];
        
        post_ener[ARRAYXY(j,k,XMAXPLUSFIVE)] = ( energy1[ARRAYXY(j,k,XMAXPLUSFOUR)] * 
                                                 pre_mass[ARRAYXY(j,k,XMAXPLUSFIVE)] + 
        					                     ener_flux[ARRAYXY(j, k  , XMAXPLUSFIVE)] - 
        					                     ener_flux[ARRAYXY(j, k+1, XMAXPLUSFIVE)] 
                                               ) / post_mass[ARRAYXY(j,k,XMAXPLUSFIVE)];
        
        advec_vol[ARRAYXY(j,k,XMAXPLUSFIVE)] = pre_vol[ARRAYXY(j,k,XMAXPLUSFIVE)] + 
                                               vol_flux_y[ARRAYXY(j,k  , XMAXPLUSFOUR)] - 
        					                   vol_flux_y[ARRAYXY(j,k+1, XMAXPLUSFOUR)];
        
        density1[ARRAYXY(j,k,XMAXPLUSFOUR)] = post_mass[ARRAYXY(j,k,XMAXPLUSFIVE)] / advec_vol[ARRAYXY(j,k,XMAXPLUSFIVE)];
        
        energy1[ARRAYXY(j,k,XMAXPLUSFOUR)] = post_ener[ARRAYXY(j,k,XMAXPLUSFIVE)];
    
    }

}


