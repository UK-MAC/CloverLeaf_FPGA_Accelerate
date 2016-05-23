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

__kernel void advec_mom_flux_ocl_kernel_y_notvec1(
    __global const double * restrict node_flux,
    __global const double * restrict node_mass_pre,
    __global const double * restrict vel1,
    __global double * restrict advec_vel,
    __global double * restrict mom_flux,
    __global const double * restrict celldy)
{
    int upwind, donor, downwind, dif;
    double sigma, width, wind;
    double vdiffuw, vdiffdw, auw, adw, limiter;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=1) &&  (k<=YMAXPLUSTWO) ) {

        if(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]<0.0){
          upwind=k+2;
          donor=k+1;
          downwind=k;
          dif=donor;
        }
        else{
          upwind=k-1;
          donor=k;
          downwind=k+1;
          dif=upwind;
        }
        sigma=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j  ,donor,XMAXPLUSFIVE)]);
        width=celldy[k];
        vdiffuw=vel1[ARRAYXY(j  ,donor,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,upwind,XMAXPLUSFIVE)];
        vdiffdw=vel1[ARRAYXY(j  ,downwind ,XMAXPLUSFIVE)]-vel1[ARRAYXY(j  ,donor,XMAXPLUSFIVE)];
        limiter=0.0;
        if(vdiffuw*vdiffdw>0.0){
          auw=fabs(vdiffuw);
          adw=fabs(vdiffdw);
          wind=1.0;
          if(vdiffdw<=0.0) wind=-1.0;
          limiter=wind*fmin(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldy[dif])/6.0,fmin(auw,adw));
        }
        advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j  ,donor,XMAXPLUSFIVE)]+(1.0-sigma)*limiter;
        mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                           *node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

