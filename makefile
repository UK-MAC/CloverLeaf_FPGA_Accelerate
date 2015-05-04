# Builds individual kernels drivers

ifndef COMPILER
  MESSAGE=select a compiler to compile in OpenMP, e.g. make COMPILER=INTEL
endif

ifndef OCL_VENDOR
    OCLMESSAGE=If you want to use OpenCL kernels, please specify the OCL_VENDOR variable
endif

OCL_LOCAL_WG_SIZE_XDIM=64 #this value must be a power of 2, less than the devices maximum size and a multiple of its preferred vector width 
OCL_LOCAL_WG_SIZE_YDIM=4
OCL_UH_LOCALWG_SMALLDIM_DEPTHTWO=1 #this value controls the small dimension of the local workgroup size in the update halo kernel when the depth is 2
                                   #it should be set to 1 or 2. 
OCL_COMMS_LOCALWG_SMALLDIM_DEPTHTWO=1 #this value controls the small dimemsion of the local wg size in the comms buffer pack / unpack kernel
                                      #when the depth is 2, it should be set to 1 or 2 
OCL_REDUCTION_WG_SIZE=128 #this variable sets the local workgroup size for the reduction kernels, must be a power of two and less than device max


OMP_INTEL     = -openmp
OMP_SUN       = -xopenmp=parallel -vpara
OMP_GNU       = -fopenmp
OMP_CRAY      =
OMP_PGI       = -mp=nonuma
OMP_PATHSCALE = -mp
OMP_XL        = -qsmp=omp -qthreaded
OMP=$(OMP_$(COMPILER))

FLAGS_INTEL     = -O3 -ipo -no-prec-div
FLAGS_SUN       = -fast -xipo=2 -Xlistv4
FLAGS_GNU       = -O3 -march=native -funroll-loops
FLAGS_CRAY      = -em -ra -h acc_model=fast_addr:no_deep_copy:auto_async_all
FLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
FLAGS_PATHSCALE = -O3
FLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036
FLAGS_          = -O3
CFLAGS_INTEL     = -O3 -ipo -no-prec-div -restrict -fno-alias
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
#OCL_ALTERA_INC=-I/opt/altera/13.0/AOCL/host/include
#OCL_ALTERA_LIB=-L/opt/altera/13.0/AOCL/linux64/lib -L/opt/altera/13.0/AOCL/host/linux64/lib -lalterahalmmd -lalterammdpcie -lpkg_editor -lalteracl -lelf -lrt -lstdc++
#OCL_ALTERA_INC=-I/opt/altera/13.1/hld/host/include
#OCL_ALTERA_LIB=-L/usr/lib64 -L/opt/altera/13.1/hld/linux64/lib -L/opt/altera/13.1/hld/host/linux64/lib -lalterahalmmd -lalterammdpcie -lalteracl -lelf -lrt -lstdc++
OCL_ALTERA_INC=-I/opt/altera/14.1/hld/host/include -I/opt/altera/14.1/hld/board/nalla_pcie/include
OCL_ALTERA_LIB=-L/opt/altera/14.1/hld/board/nalla_pcie/linux64/lib -L/opt/altera/14.1/hld/host/linux64/lib -lalteracl -lacl_emulator_kernel_rt  -lalterahalmmd -lnalla_pcie_mmd -lelf -lrt -ldl -lstdc++

OCL_LIB=$(OCL_$(OCL_VENDOR)_LIB)
OCL_INC=$(OCL_$(OCL_VENDOR)_INC)



ifdef DEBUG
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  FLAGS_SUN       = -g -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  FLAGS_CRAY      = -O0 -g -em -eD
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr
  FLAGS_PATHSCALE = -O0 -g
  FLAGS_XL       = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qinit=f90ptr -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c
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

FLAGS=$(FLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS) -cpp
CFLAGS=$(CFLAGS_$(COMPILER)) $(OMP) $(I3E) $(C_OPTIONS) -c -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DWG_SIZE_X=$(OCL_LOCAL_WG_SIZE_XDIM) -DWG_SIZE_Y=$(OCL_LOCAL_WG_SIZE_YDIM) -DUH_SMALL_DIM_DEPTHTWO=$(OCL_UH_LOCALWG_SMALLDIM_DEPTHTWO) -DCOMMS_SMALL_DIM_DEPTHTWO=$(OCL_COMMS_LOCALWG_SMALLDIM_DEPTHTWO) -DREDUCTION_WG_SIZE=$(OCL_REDUCTION_WG_SIZE) #-DOCL_VERBOSE=1 -DPROFILE_OCL_KERNELS=1 #-DDUMP_BINARY

MPI_COMPILER=mpif90
C_MPI_COMPILER=mpicc
CXX_MPI_COMPILER=mpiCC


accelerate_driver:  accelerate_driver.f90
	$(CXX_MPI_COMPILER) $(CFLAGS) $(OCL_INC) accelerate_kernel_ocl.C setup_ocl.C CloverCL.C accelerate_ocl_moveBuffers.C 
	$(C_MPI_COMPILER) $(CFLAGS) timer_c.c 
	$(MPI_COMPILER) -c $(FLAGS) set_data.f90 timer.f90 accelerate_driver.f90 
	$(MPI_COMPILER) $(FLAGS) $(OCL_LIB) timer_c.o set_data.o timer.o CloverCL.o accelerate_ocl_moveBuffers.o  setup_ocl.o accelerate_kernel_ocl.o  accelerate_driver.o -o accelerate_driver ; echo $(MESSAGE)


clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx accelerate_driver
