!Crown Copyright 2012 AWE.
!
! This file is part of CloverLeaf.
!
! CloverLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! CloverLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! CloverLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief standalone driver for the acceleration kernels
!>  @author Wayne Gaudin
!>  @details Calls user requested kernel in standalone mode


PROGRAM accelerate_driver

  USE set_data_module
  USE iso_c_binding
  !USE accelerate_kernel_module

  IMPLICIT NONE

  INTEGER :: numargs,iargc,i
  CHARACTER (LEN=20)  :: command_line,temp
  CHARACTER(LEN=12) :: OpenCL_vendor
  CHARACTER(LEN=12) :: OpenCL_type


  INTEGER :: x_size,y_size

  REAL(KIND=8) :: kernel_time,timer,acceleration_time

  LOGICAL :: use_fortran_kernels,use_C_kernels
  INTEGER :: x_min,x_max,y_min,y_max,its,iteration
  REAL(KIND=8) :: dt
  !REAL(KIND=8),ALLOCATABLE :: xarea(:,:),yarea(:,:),volume(:,:)
  !REAL(KIND=8),ALLOCATABLE :: density0(:,:),pressure(:,:),viscosity(:,:)
  !REAL(KIND=8),ALLOCATABLE :: xvel0(:,:),yvel0(:,:),xvel1(:,:),yvel1(:,:),work_array1(:,:)

    REAL(C_DOUBLE), POINTER :: density0(:,:)
    REAL(C_DOUBLE), POINTER :: pressure(:,:)
    REAL(C_DOUBLE), POINTER :: viscosity(:,:)
    REAL(C_DOUBLE), POINTER :: xarea(:,:)
    REAL(C_DOUBLE), POINTER :: yarea(:,:)
    REAL(C_DOUBLE), POINTER :: volume(:,:)
    REAL(C_DOUBLE), POINTER :: xvel0(:,:)
    REAL(C_DOUBLE), POINTER :: yvel0(:,:)
    REAL(C_DOUBLE), POINTER :: xvel1(:,:)
    REAL(C_DOUBLE), POINTER :: yvel1(:,:)
    REAL(C_DOUBLE), POINTER :: work_array1(:,:)

    TYPE(C_PTR) :: density0_cptr, pressure_cptr, viscosity_cptr, xarea_cptr, yarea_cptr, volume_cptr
    TYPE(C_PTR) :: xvel0_cptr, yvel0_cptr, xvel1_cptr, yvel1_cptr, work_array_cptr


  x_size=100
  y_size=100
  its=1
  OpenCL_vendor = "Nvidia"
  OpenCL_type = "GPU"

  numargs = iargc()

  DO i=1,numargs,2
    CALL GETARG(i,command_line)
    SELECT CASE (command_line)
        CASE("-help")
          WRITE(*,*) "Usage -nx 100 -ny 100 -its 10 -kernel fortran|c"
          stop
        CASE("-nx")
            CALL GETARG(i+1,temp)
            READ(UNIT=temp,FMT="(I20)") x_size
        CASE("-ny")
            CALL GETARG(i+1,temp)
            READ(UNIT=temp,FMT="(I20)") y_size
        CASE("-its")
            CALL GETARG(i+1,temp)
            READ(UNIT=temp,FMT="(I20)") its
        CASE("-kernel")
            CALL GETARG(i+1,temp)
            IF(temp.EQ."fortran") THEN
              use_fortran_kernels=.TRUE.
              use_C_kernels=.FALSE.
            ENDIF
            IF(temp.EQ."c") THEN
              use_fortran_kernels=.FALSE.
              use_C_kernels=.TRUE.
            ENDIF

        CASE("-ocltype")
            CALL GETARG(i+1,temp)
            OpenCL_type = temp

        CASE("-oclvendor")
            CALL GETARG(i+1,temp)
            OpenCL_vendor = temp

    END SELECT
  ENDDO

  x_min=1
  y_min=1
  x_max=x_size
  y_max=y_size

  WRITE(*,*) "Accelerate Kernel"
  WRITE(*,*) "Mesh size ",x_size,y_size
  WRITE(*,*) "OpenCL Type: ", OpenCL_type, " OpenCL Vendor: ", OpenCL_vendor
  WRITE(*,*) "Iterations ",its

  CALL allocate_aligned_array(density0_cptr  , (x_max+4)*(y_max+4))
  CALL allocate_aligned_array(pressure_cptr  , (x_max+4)*(y_max+4))
  CALL allocate_aligned_array(viscosity_cptr , (x_max+4)*(y_max+4))
  CALL allocate_aligned_array(xarea_cptr     , (x_max+5)*(y_max+4))
  CALL allocate_aligned_array(yarea_cptr     , (x_max+4)*(y_max+5))
  CALL allocate_aligned_array(volume_cptr    , (x_max+4)*(y_max+4))
  CALL allocate_aligned_array(xvel0_cptr     , (x_max+5)*(y_max+5))
  CALL allocate_aligned_array(yvel0_cptr     , (x_max+5)*(y_max+5))
  CALL allocate_aligned_array(xvel1_cptr     , (x_max+5)*(y_max+5))
  CALL allocate_aligned_array(yvel1_cptr     , (x_max+5)*(y_max+5))
  CALL allocate_aligned_array(work_array_cptr, (x_max+5)*(y_max+5))

    CALL C_F_POINTER(density0_cptr, density0     , [(x_max+4),(y_max+4)])
    CALL C_F_POINTER(pressure_cptr, pressure     , [(x_max+4),(y_max+4)])
    CALL C_F_POINTER(viscosity_cptr, viscosity   , [(x_max+4),(y_max+4)] )
    CALL C_F_POINTER(xarea_cptr, xarea           , [(x_max+5),(y_max+4)])
    CALL C_F_POINTER(yarea_cptr, yarea           , [(x_max+4),(y_max+5)]            )
    CALL C_F_POINTER(volume_cptr, volume         , [(x_max+4),(y_max+4)])
    CALL C_F_POINTER(xvel0_cptr, xvel0           , [(x_max+5),(y_max+5)])
    CALL C_F_POINTER(yvel0_cptr, yvel0           , [(x_max+5),(y_max+5)])
    CALL C_F_POINTER(xvel1_cptr, xvel1           , [(x_max+5),(y_max+5)])
    CALL C_F_POINTER(yvel1_cptr, yvel1           , [(x_max+5),(y_max+5)])
    CALL C_F_POINTER(work_array_cptr, work_array1, [(x_max+5),(y_max+5)])

    WRITE(*,*) "After c to f pointers "

    !density0(1,1)=2.0_8
    density0((x_max/2)+2:,:)=2.0_8
    density0(:(x_max/2)+2,:)=1.0_8

    pressure((x_max/2)+2:,:)=2.0_8
    pressure(:(x_max/2)+2,:)=1.0_8

    viscosity=0.0_8
    viscosity((x_max/2)+2,:)=0.1_8

    xarea=1.0_8
    yarea=1.0_8
    volume=1.0_8
    xvel0=1.0
    yvel0=1.0
    xvel1=1.0
    yvel1=1.0
    dt=0.0001_8 ! Needs to be mesh specific
    WRITE(*,*) "Data set"



  !caLL set_data(x_min,x_max,y_min,y_max, &
  !              xarea=xarea,             &
  !              yarea=yarea,             &
  !              volume=volume,           &
  !              density0=density0,       &
  !              pressure=pressure,       &
  !              viscosity=viscosity,     &
  !              xvel0=xvel0,             &
  !              xvel1=xvel1,             &
  !              yvel0=yvel0,             &
  !              yvel1=yvel1,             &
  !              work_array1=work_array1, &
  !              dt=dt                    )



  CALL setup_opencl(TRIM(OpenCL_vendor)//char(0), TRIM(OpenCL_type)//char(0),&
                    x_min, x_max, y_min, y_max, &
                    !number_of_states, &
                    2, &
                    1, 1, 1, 1, 1, 1, 1)
                    !g_small, g_big, dtmin, dtc_safe, dtu_safe, dtv_safe, dtdiv_safe)



  CALL accelerate_ocl_writebuffers(density0, pressure, viscosity, xvel0, xvel1, yvel0, yvel1, volume, xarea, yarea)



  WRITE(*,*) "Running OpenCL kernel"


  acceleration_time=0.0_8
  kernel_time=timer()

  DO iteration=1,its

    CALL accelerate_kernel_ocl(x_min, x_max, y_min, y_max, dt )

  ENDDO


  acceleration_time=acceleration_time+(timer()-kernel_time)


  CALL accelerate_ocl_readbuffers(xvel1, yvel1);



  WRITE(*,*) "Accelerate time ",acceleration_time 
  WRITE(*,*) "X vel ",SUM(xvel1)
  WRITE(*,*) "Y vel ",SUM(yvel1)

  ! Answers need checking

  !CALL free_aligned_array(density0_cptr)
  !CALL free_aligned_array(pressure_cptr)
  !CALL free_aligned_array(viscosity_cptr)
  !CALL free_aligned_array(xarea_cptr)
  !CALL free_aligned_array(yarea_cptr)
  !CALL free_aligned_array(volume_cptr)
  !CALL free_aligned_array(xvel0_cptr)
  !CALL free_aligned_array(yvel0_cptr)
  !CALL free_aligned_array(xvel1_cptr)
  !CALL free_aligned_array(yvel1_cptr)
  !CALL free_aligned_array(work_array_cptr)

  !DEALLOCATE(xarea)
  !DEALLOCATE(yarea)
  !DEALLOCATE(volume)
  !DEALLOCATE(density0)
  !DEALLOCATE(pressure)
  !DEALLOCATE(viscosity)
  !DEALLOCATE(xvel0)
  !DEALLOCATE(yvel0)
  !DEALLOCATE(xvel1)
  !DEALLOCATE(yvel1)
  !DEALLOCATE(work_array1)

END PROGRAM accelerate_driver

