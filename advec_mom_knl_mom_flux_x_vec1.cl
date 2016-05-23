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

__kernel void advec_mom_flux_ocl_kernel_x_vec1(
    __global const double * restrict node_flux,
    __global const double * restrict node_mass_pre,
    __global const double * restrict vel1,
    __global double * restrict advec_vel,
    __global double * restrict mom_flux,
    __global const double * restrict celldx)
{
    double sigma, sigma2, wind, wind2, width;
    double vdiffuw, vdiffdw, vdiffuw2, vdiffdw2, auw, adw, auw2, limiter, limiter2;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=1) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

          sigma=fabs(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j+1,k, XMAXPLUSFIVE)]);
          sigma2=fabs(node_flux[ARRAYXY(j ,k  ,XMAXPLUSFIVE)])/(node_mass_pre[ARRAYXY(j  ,k, XMAXPLUSFIVE)]);
          width=celldx[j];
          vdiffuw=vel1[ARRAYXY(j+1,k, XMAXPLUSFIVE)]-vel1[ARRAYXY(j+2, k, XMAXPLUSFIVE)];
          vdiffdw=vel1[ARRAYXY(j  ,k, XMAXPLUSFIVE)]-vel1[ARRAYXY(j+1, k, XMAXPLUSFIVE)];
          vdiffuw2=vel1[ARRAYXY(j ,k, XMAXPLUSFIVE)]-vel1[ARRAYXY(j-1, k, XMAXPLUSFIVE)];
          vdiffdw2=-1*vdiffdw;
          auw=fabs(vdiffuw);
          adw=fabs(vdiffdw);
          auw2=fabs(vdiffuw2);
          wind=1.0;
          wind2=1.0;

          if(vdiffdw<=0.0) wind=-1.0;
          if(vdiffdw2<=0.0) wind2=-1.0;
          limiter=wind*fmin(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldx[j+1])/6.0,fmin(auw,adw));
          limiter2=wind2*fmin(width*((2.0-sigma2)*adw/width+(1.0+sigma2)*auw2/celldx[j-1])/6.0,fmin(auw2,adw));
          if(vdiffuw*vdiffdw<=0.0) limiter=0.0;
          if(vdiffuw2*vdiffdw2<=0.0) limiter2=0.0;
          if(node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]<0.0){
            advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j+1,k  ,XMAXPLUSFIVE)]+(1.0-sigma)*limiter;
          }
          else{
            advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=vel1[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]+(1.0-sigma2)*limiter2;
          }
          mom_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]=advec_vel[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)]
                                                             *node_flux[ARRAYXY(j  ,k  ,XMAXPLUSFIVE)];
    }
}

