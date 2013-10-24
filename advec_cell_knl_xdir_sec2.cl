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

__kernel void advec_cell_xdir_section2_kernel(
    __global const double * restrict vertexdx,    
    __global const double * restrict density1,    
    __global const double * restrict energy1,     
    __global double * restrict mass_flux_x, 
    __global const double * restrict vol_flux_x,  
    __global const double * restrict pre_vol,     
    __global double * restrict ener_flux)
{
    int upwind, donor, downwind, dif;
    double sigma, sigmat, sigmav, sigmam, sigma3, sigma4; 
    double diffuw, diffdw, limiter;
    __const double one_by_six=1.0/6.0;


    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSTHREE) && (k>=2) && (k<=YMAXPLUSONE)) {
    
        if ( vol_flux_x[ARRAYXY(j, k, XMAXPLUSFIVE)] > 0.0 ) {
            upwind   = j-2;
            donor    = j-1;
            downwind = j;
            dif      = donor;
        } else { 
            upwind   = min(j+1,XMAXPLUSTWO);
            donor    = j;
            downwind = j-1;
            dif      = upwind;
        } 
        
        sigmat = fabs( vol_flux_x[ARRAYXY(j,k,XMAXPLUSFIVE)] ) / pre_vol[ARRAYXY(donor,k,XMAXPLUSFIVE)];
        sigma3 = (1.0 + sigmat)*( vertexdx[j] / vertexdx[dif] );
        sigma4 = 2.0 - sigmat;
        
        sigma = sigmat;
        sigmav = sigmat;
        
        diffuw = density1[ARRAYXY(donor,k,XMAXPLUSFOUR)] - density1[ARRAYXY(upwind,k,XMAXPLUSFOUR)];
        diffdw = density1[ARRAYXY(downwind,k,XMAXPLUSFOUR)] - density1[ARRAYXY(donor,k,XMAXPLUSFOUR)];
        
        if (diffuw*diffdw > 0.0) { 
            limiter = (1.0 - sigmav) * copysign(1.0,diffdw) * fmin(fabs(diffuw), fmin( fabs(diffdw), one_by_six*(sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw) ) ) );
        } else { 
            limiter = 0.0;
        } 
        mass_flux_x[ARRAYXY(j,k,XMAXPLUSFIVE)] = vol_flux_x[ARRAYXY(j,k,XMAXPLUSFIVE)] * 
                                                 ( density1[ARRAYXY(donor,k,XMAXPLUSFOUR)] + limiter );
        
        sigmam = fabs( mass_flux_x[ARRAYXY(j,k,XMAXPLUSFIVE)] ) / ( density1[ARRAYXY(donor,k,XMAXPLUSFOUR)] * 
                                                                               pre_vol[ARRAYXY(donor,k,XMAXPLUSFIVE)] );
        diffuw = energy1[ARRAYXY(donor,k,XMAXPLUSFOUR)] - energy1[ARRAYXY(upwind,k,XMAXPLUSFOUR)];
        diffdw = energy1[ARRAYXY(downwind,k,XMAXPLUSFOUR)] - energy1[ARRAYXY(donor,k,XMAXPLUSFOUR)];
        
        if (diffuw*diffdw > 0.0) { 
            limiter = (1.0 - sigmam) * copysign(1.0,diffdw) * fmin(fabs(diffuw), fmin( fabs(diffdw), one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw) ) ) );
        } else {
            limiter = 0.0;
        } 
        
        ener_flux[ARRAYXY(j,k,XMAXPLUSFIVE)] = mass_flux_x[ARRAYXY(j,k,XMAXPLUSFIVE)] * 
                                               ( energy1[ARRAYXY(donor,k,XMAXPLUSFOUR)] + limiter );
    } 
}

