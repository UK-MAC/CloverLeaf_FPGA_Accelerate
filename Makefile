#Crown Copyright 2012 AWE.
#
# This file is part of CloverLeaf.
#
# CloverLeaf is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
# CloverLeaf is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# CloverLeaf. If not, see http://www.gnu.org/licenses/.

#  @brief Makefile for OCL version of CloverLeaf
#  @author Wayne Gaudin, Andrew Mallinson, David Beckingsale 
#  @details Agnostic, platform independent makefile for the Clover Leaf benchmark code.

# It is not meant to be clever in anyway, just a simple build out of the box script.
# Machine/compiler specific options can be set in the FLAGS variable, if required for debugging/optimisation etc
# Just make sure mpif90 is in your path. It uses mpif90 even for all builds because this abstracts the base
#  name of the compiler. If you are on a system that doesn't use mpif90, just replace mpif90 with the compiler name
#  of choice. The only mpi dependencies in this non-MPI version are mpi_wtime in timer.f90.

# There is no single way of turning OpenMP compilation on with all compilers.
# The known compilers have been added as a variable. By default the make
#  will use no options, which will work on Cray for example, but not on other
#  compilers.
# To select a OpenMP compiler option, do this in the shell before typing make:-
#
#  export COMPILER=INTEL       # to select the Intel flags
#  export COMPILER=SUN         # to select the Sun flags
#  export COMPILER=GNU         # to select the Gnu flags
#  export COMPILER=CRAY        # to select the Cray flags
#  export COMPILER=PGI         # to select the PGI flags
#  export COMPILER=PATHSCALE   # to select the Pathscale flags
#  export COMPILER=XL          # to select the IBM Xlf flags

# or this works as well:-
#
# make COMPILER=INTEL
# make COMPILER=SUN
# make COMPILER=GNU
# make COMPILER=CRAY
# make COMPILER=PGI
# make COMPILER=PATHSCALE
# make COMPILER=XL
#

# Don't forget to set the number of threads you want to use, like so
# export OMP_NUM_THREADS=4

# usage: make                     # Will make the binary
#        make clean               # Will clean up the directory
#        make DEBUG=1             # Will select debug options. If a compiler is selected, it will use compiler specific debug options
#        make IEEE=1              # Will select debug options as long as a compiler is selected as well
# e.g. make COMPILER=INTEL MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc DEBUG=1 IEEE=1 # will compile with the intel compiler with intel debug and ieee flags included

ifndef COMPILER
  MESSAGE=select a compiler to compile in OpenMP, e.g. make COMPILER=INTEL
endif

ifndef OCL_VENDOR
	OCLMESSAGE=If you want to use OpenCL kernels, please specify the OCL_VENDOR variable
endif

OCL_LOCAL_WG_SIZE_XDIM=256 #this value must be a power of 2, less than the devices maximum size and a multiple of its preferred vector width 

OMP_INTEL     = 
OMP_SUN       = -xopenmp=parallel -vpara
OMP_GNU       = -fopenmp
OMP_CRAY      =
OMP_PGI       = -mp=nonuma
OMP_PATHSCALE = -mp
OMP_XL        = -qsmp=omp -qthreaded
OMP=$(OMP_$(COMPILER))

FLAGS_INTEL     = -O3 -no-prec-div -fpp
FLAGS_SUN       = -fast -xipo=2 -Xlistv4
FLAGS_GNU       = -O3 -march=native -funroll-loops -cpp
FLAGS_CRAY      = -em -ra -h acc_model=fast_addr:no_deep_copy:auto_async_all -eZ
FLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
FLAGS_PATHSCALE = -O3
FLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036
FLAGS_          = -O3
CFLAGS_INTEL     = -O3 -no-prec-div -restrict -fno-alias
CFLAGS_SUN       = -fast -xipo=2
CFLAGS_GNU       = -O3 -march=native -funroll-loops
CFLAGS_CRAY      = -em -h list=a
CFLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
CFLAGS_PATHSCALE = -O3
CFLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg
CFLAGS_          = -O3

OCL_INTEL_INC=-I/opt/opencl/intel-2013_xe_sdk_3.0.67279/include
OCL_INTEL_LIB=-L/opt/opencl/intel-2013_xe_sdk_3.0.67279/lib64 -lOpenCL
OCL_WILLOW_INC=-I/opt/amd-app-sdk/2/5-RC2/include
OCL_WILLOW_LIB=-L/opt/amd-app-sdk/2/5-RC2/lib64 -lOpenCL -lstdc++
OCL_AMD_INC=-I/opt/opencl/amd-app-2.7/include
OCL_AMD_LIB=-L/opt/opencl/amd-app-2.7/lib/x86_64 -lOpenCL -lstdc++
OCL_CRAY_INC=-I/opt/nvidia/cudatoolkit/default/include -I/home/users/p01379/lustre/OpenCL_headers
OCL_CRAY_LIB=-L/opt/cray/nvidia/default/lib64 -lOpenCL
OCL_LIB=$(OCL_$(OCL_VENDOR)_LIB)
OCL_INC=$(OCL_$(OCL_VENDOR)_INC)

ifdef DEBUG
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created -fpp
  FLAGS_SUN       = -g -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check -cpp
  FLAGS_CRAY      = -O0 -g -em -eD -eZ
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr
  FLAGS_PATHSCALE = -O0 -g
  FLAGS_XL       = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qinit=f90ptr -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c
  FLAGS_          = -O0 -g
  CFLAGS_INTEL    = -O0 -g -debug all -traceback
  CFLAGS_SUN      = -g -O0 -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  CFLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  CFLAGS_CRAY     = -O0 -g -em -eD
  CFLAGS_PGI      = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk
  CFLAGS_PATHSCALE= -O0 -g
  CFLAGS_XL      = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qsrcmsg
endif

ifdef IEEE
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_SUN       = -fsimple=0 -fns=no
  I3E_GNU       = -ffloat-store
  I3E_CRAY      = -hflex_mp=intolerant
  I3E_PGI       = -Kieee
  I3E_PATHSCALE = -mieee-fp
  I3E_XL       = -qfloat=nomaf
  I3E=$(I3E_$(COMPILER))
endif

FLAGS=$(FLAGS_$(COMPILER)) $(I3E) $(OPTIONS) $(OCL_LIB) -DUSE_EXPLICIT_COMMS_BUFF_PACK 
CFLAGS=$(CFLAGS_$(COMPILER)) $(I3E) $(COPTIONS) -c -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DWG_SIZE_X=$(OCL_LOCAL_WG_SIZE_XDIM) #-DOCL_VERBOSE=1 #-DPROFILE_OCL_KERNELS=1 #-DDUMP_BINARY  
MPI_COMPILER=mpif90
C_MPI_COMPILER=mpicc
CXX_MPI_COMPILER=mpiCC

clover_leaf: c_lover clover_ocl *.f90 Makefile
	$(MPI_COMPILER) $(FLAGS)	\
	data.f90			\
	definitions.f90			\
	clover.f90			\
	report.f90			\
	timer.f90			\
	parse.f90			\
	read_input.f90			\
	initialise_chunk.f90		\
	build_field.f90			\
	update_halo.f90			\
	ideal_gas.f90			\
	start.f90			\
	generate_chunk.f90		\
	initialise.f90			\
	field_summary.f90		\
	viscosity.f90			\
	calc_dt.f90			\
	timestep.f90			\
	accelerate.f90			\
	revert.f90			\
	PdV.f90				\
	flux_calc.f90			\
	advec_cell_driver.f90		\
	advec_mom_kernel.f90        \
	advec_mom_driver.f90		\
	advection.f90			\
	reset_field.f90			\
	hydro.f90			\
	visit.f90			\
	clover_leaf.f90			\
	ideal_gas_kernel_ocl.o          \
	viscosity_kernel_ocl.o          \
	flux_calc_kernel_ocl.o          \
	accelerate_kernel_ocl.o         \
	advec_mom_kernel_ocl.o          \
	advec_cell_kernel_ocl.o         \
	pdv_kernel_ocl.o                \
	calc_dt_kernel_ocl.o            \
	reset_field_kernel_ocl.o        \
	generate_chunk_kernel_ocl.o     \
	initialise_chunk_kernel_ocl.o   \
	revert_kernel_ocl.o             \
	field_summary_kernel_ocl.o      \
	update_halo_kernel_ocl.o        \
	setup_ocl.o                     \
	comms_buffers_kernel_ocl.o      \
	ocl_read_buffers.o              \
	timer_c.o                       \
	CloverCL.o                      \
	-o clover_leaf; echo $(MESSAGE)

c_lover: *.c Makefile
	$(C_MPI_COMPILER) $(CFLAGS)     \
	timer_c.c

clover_ocl: Makefile
	$(CXX_MPI_COMPILER) $(CFLAGS) \
	$(OCL_INC)                    \
	ideal_gas_kernel_ocl.C        \
	viscosity_kernel_ocl.C        \
	flux_calc_kernel_ocl.C        \
	accelerate_kernel_ocl.C       \
	advec_mom_kernel_ocl.C        \
	advec_cell_kernel_ocl.C       \
	calc_dt_kernel_ocl.C          \
	update_halo_kernel_ocl.C      \
	pdv_kernel_ocl.C              \
	reset_field_kernel_ocl.C      \
	revert_kernel_ocl.C           \
	generate_chunk_kernel_ocl.C   \
	initialise_chunk_kernel_ocl.C \
	field_summary_kernel_ocl.C    \
	setup_ocl.C                   \
	comms_buffers_kernel_ocl.C    \
	ocl_read_buffers.C            \
	CloverCL.C; echo $(OCLMESSAGE)

clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx clover_leaf cloverleaf_ocl_binary
