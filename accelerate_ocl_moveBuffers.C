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
 *  @brief OCL host-side accelerate kernel.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Launches the OCL device-side accelerate kernel 
*/


#include "CloverCL.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <cmath>

#include <sys/time.h>
#define AOCL_ALIGNMENT 64


extern "C" void accelerate_ocl_writebuffers_(double * density0, double * pressure, double * viscosity, 
                                             double * xvel0, double * xvel1, double * yvel0, double * yvel1, 
                                             double * volume , double * xarea, double * yarea);


extern "C" void accelerate_ocl_readbuffers_(double * xvel1, double * yvel1);

extern "C" void allocate_aligned_array_(void** pointer, int* size);

extern "C" void free_aligned_array_(void* pointer);

void accelerate_ocl_writebuffers_(double * density0, double * pressure, double * viscosity, 
                                  double * xvel0, double * xvel1, double * yvel0, double * yvel1, 
                                  double * volume , double * xarea, double * yarea)
{

    CloverCL::write_accelerate_buffers_tocard(density0, pressure, viscosity, xvel0, xvel1, yvel0, yvel1, volume, xarea, yarea);

}

void accelerate_ocl_readbuffers_(double * xvel1, double * yvel1)
{

    CloverCL::read_accelerate_buffers_backfromcard(xvel1, yvel1);

}

void allocate_aligned_array_(void** pointer, int* size)
{
    std::cout << "Allocating: " << *size << " elements" << std::endl;
    posix_memalign(pointer, AOCL_ALIGNMENT, *size*sizeof(double));
}

void free_aligned_array_(void* pointer)
{
    free(pointer);
}

