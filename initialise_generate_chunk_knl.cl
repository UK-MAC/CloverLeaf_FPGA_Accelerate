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
 *  @brief OCL device-side initialise chunk kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Invokes the user specified chunk initialisation kernels.
 */

#include "ocl_knls.h"

__kernel void initialise_chunk_cell_x_ocl_kernel(
    const double dx,
    __global const double * restrict vertexx,
    __global double * restrict cellx,
    __global double * restrict celldx)
{

    int j = get_global_id(0)-2;

    if (j<=XMAX+2) {

        cellx[ARRAY1D(j, XMIN-2)] = 0.5 * ( vertexx[ARRAY1D(j, XMIN-2)] + vertexx[ARRAY1D(j+1, XMIN-2)] );
        
        celldx[ARRAY1D(j, XMIN-2)] = dx;

    }
}


__kernel void initialise_chunk_cell_y_ocl_kernel(
    const double dy,
    __global const double * restrict vertexy,
    __global double * restrict celly,
    __global double * restrict celldy)
{

    int k = get_global_id(0)-2;
    
    if (k<=YMAX+2) {

        celly[ARRAY1D(k, YMIN-2)] = 0.5 * ( vertexy[ARRAY1D(k, YMIN-2)] + vertexy[ARRAY1D(k+1, YMIN-2)] );
        
        celldy[ARRAY1D(k, YMIN-2)] = dy;
    }
}


__kernel void initialise_chunk_vertex_x_ocl_kernel(
    const double xmin,
    const double dx,
    __global double * restrict vertexx,
    __global double * restrict vertexdx)
{

    int j = get_global_id(0)-2;

    if (j<=XMAX+3) {

        vertexx[ARRAY1D(j,XMIN-2)] = xmin + dx * (j-XMIN);
        vertexdx[ARRAY1D(j,XMIN-2)] = dx;

    }
}


__kernel void initialise_chunk_vertex_y_ocl_kernel(
    const double ymin,
    const double dy,
    __global double * restrict vertexy,
    __global double * restrict vertexdy)
{  

    int k = get_global_id(0)-2;

    if (k<=YMAX+3) {

        vertexy[ARRAY1D(k, YMIN-2)] = ymin + dy * (k-YMIN);

        vertexdy[ARRAY1D(k, YMIN-2)] = dy;

    }

}

__kernel void initialise_chunk_volume_area_ocl_kernel(
    const double dx,
    const double dy,
    __global double * restrict volume,
    __global const double * restrict celldx,
    __global const double * restrict celldy,
    __global double * restrict xarea,
    __global double * restrict yarea)
{   
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j<=XMAXPLUSTHREE) && (k<=YMAXPLUSTHREE) ) {
    
        volume[ARRAYXY(j,k, XMAXPLUSFOUR)] = dx * dy;

        xarea[ARRAYXY(j,k, XMAXPLUSFIVE)] = celldy[k];

        yarea[ARRAYXY(j,k, XMAXPLUSFOUR)] = celldx[j];
    }
}


__kernel void generate_chunk_ocl_kernel(
        __global const double * restrict vertexx,    
        __global const double * restrict vertexy,    
        __global const double * restrict cellx,
        __global const double * restrict celly,
        __global double * restrict density0,
        __global double * restrict energy0,
        __global double * restrict xvel0,
        __global double * restrict yvel0,
        const int number_of_states,
        __global const double * restrict state_density,
        __global const double * restrict state_energy,
        __global const double * restrict state_xvel,
        __global const double * restrict state_yvel,
        __global const double * restrict state_xmin,
        __global const double * restrict state_xmax,
        __global const double * restrict state_ymin,
        __global const double * restrict state_ymax,
        __global const double * restrict state_radius,
        __global const int * restrict state_geometry,
        const int g_rect,
        const int g_circ)
{
    double radius;

    int k = get_global_id(1)-2;
    int j = get_global_id(0)-2;

    if ( (j<=XMAX+2) && (k<=YMAX+2) ) {

        int highest_state = 1;

        for (int state = 1; state <= number_of_states; state++) {
            if(state_geometry[ARRAY1D(state,1)] == g_rect) {
              if(vertexx[ARRAY1D(j,-1)] >= state_xmin[ARRAY1D(state,1)] && vertexx[ARRAY1D(j,-1)] < state_xmax[ARRAY1D(state,1)]) {
                if(vertexy[ARRAY1D(k,-1)] >= state_ymin[ARRAY1D(state,1)] && vertexy[ARRAY1D(k,-1)] < state_ymax[ARRAY1D(state,1)]) {
                    highest_state = state;
                }
              }
            } else if(state_geometry[ARRAY1D(state,1)] == g_circ ) {
              radius=sqrt(cellx[ARRAY1D(j,-1)]*cellx[ARRAY1D(j,-1)]+celly[ARRAY1D(k,-1)]*celly[ARRAY1D(k,-1)]);
              if(radius <= state_radius[ARRAY1D(state,1)]){
                    highest_state = state;
              }
            }
        }

        energy0[ARRAY2D(j,k,XMAX+4,XMIN-2,YMIN-2)]=state_energy[ARRAY1D(highest_state,1)];
        density0[ARRAY2D(j,k,XMAX+4,XMIN-2,YMIN-2)]=state_density[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j,k,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j+1,k,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j+1,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j,k,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j+1,k,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j+1,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
  }

}
