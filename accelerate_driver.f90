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
  !USE accelerate_kernel_module

  IMPLICIT NONE

!$ INTEGER :: OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM

  INTEGER :: numargs,iargc,i
  CHARACTER (LEN=20)  :: command_line,temp
  CHARACTER(LEN=12) :: OpenCL_vendor
  CHARACTER(LEN=12) :: OpenCL_type

  INTEGER :: x_size,y_size

  REAL(KIND=8) :: kernel_time,timer,acceleration_time,timer_tod

  LOGICAL :: use_fortran_kernels,use_C_kernels,reset_data
  INTEGER :: x_min,x_max,y_min,y_max,its,iteration
  REAL(KIND=8) :: dt
  REAL(KIND=8),ALLOCATABLE :: xarea(:,:),yarea(:,:),volume(:,:)
  REAL(KIND=8),ALLOCATABLE :: celldx(:),celldy(:)
  REAL(KIND=8),ALLOCATABLE :: density0(:,:),energy0(:,:),pressure(:,:),soundspeed(:,:),viscosity(:,:)
  REAL(KIND=8),ALLOCATABLE :: xvel0(:,:),yvel0(:,:),xvel1(:,:),yvel1(:,:),work_array1(:,:)
  REAL(KIND=8),ALLOCATABLE :: xvel_orig(:,:),yvel_orig(:,:)
  REAL(KIND=8),ALLOCATABLE :: iter_timings(:)

    REAL(KIND=8) :: accelerate_iter1_after, accelerate_iter1_before, accelerate_main_after, accelerate_main_before, first_iteration 

!$OMP PARALLEL
!$  IF(OMP_GET_THREAD_NUM().EQ.0) THEN
!$    WRITE(*,'(a15,i5)') 'Thread Count: ',OMP_GET_NUM_THREADS()
!$  ENDIF
!$OMP END PARALLEL

  x_size=100
  y_size=100
  its=1
  use_fortran_kernels=.TRUE.
  use_C_kernels=.FALSE.
  reset_data=.FALSE.
  OpenCL_vendor = "Nvidia"
  OpenCL_type = "GPU"

  numargs = iargc()

  DO i=1,numargs,2
    CALL GETARG(i,command_line)
    SELECT CASE (command_line)
      CASE("-help")
        WRITE(*,*) "Usage -nx 100 -ny 100 -its 10 -kernel fortran|c -reset off|on"
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
      CASE("-reset")
        CALL GETARG(i+1,temp)
        IF(temp.EQ."on") THEN
          reset_data=.TRUE.
        ENDIF
        IF(temp.EQ."off") THEN
          reset_data=.FALSE.
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

    ALLOCATE(iter_timings(its))

  WRITE(*,*) "Accelerate Kernel"
  WRITE(*,*) "Mesh size ",x_size,y_size
  WRITE(*,*) "OpenCL Type: ", OpenCL_type, " OpenCL Vendor: ", OpenCL_vendor
  WRITE(*,*) "Iterations ",its

  kernel_time=timer_tod()

  CALL set_data(x_min,x_max,y_min,y_max, &
                xarea=xarea,             &
                yarea=yarea,             &
                celldx=celldx,           &
                celldy=celldy,           &
                volume=volume,           &
                density0=density0,       &
                energy0=energy0,         &
                pressure=pressure,       &
                soundspeed=soundspeed,   &
                viscosity=viscosity,     &
                xvel0=xvel0,             &
                xvel1=xvel1,             &
                yvel0=yvel0,             &
                yvel1=yvel1,             &
                work_array1=work_array1, &
                dt=dt                    )

  WRITE(*,*) "Data set"
  WRITE(*,*) "X vel before:",SUM(xvel1)
  WRITE(*,*) "Y vel before:",SUM(yvel1)

  CALL setup_opencl(TRIM(OpenCL_vendor)//char(0), TRIM(OpenCL_type)//char(0),&
                    x_min, x_max, y_min, y_max, &
                    !number_of_states, &
                    2, &
                    1, 1, 1, 1, 1, 1, 1)
                    !g_small, g_big, dtmin, dtc_safe, dtu_safe, dtv_safe, dtdiv_safe)



  CALL accelerate_ocl_writebuffers(density0, pressure, viscosity, xvel0, xvel1, yvel0, yvel1, volume, xarea, yarea)

  CALL accelerate_ocl_call_clfinish()

  IF(reset_data) THEN
    !ALLOCATE(xvel_orig(x_min-2:x_max+3,y_min-2:y_max+3))
    !ALLOCATE(xvel_orig(x_min-2:x_max+3,y_min-2:y_max+3))
    ALLOCATE(yvel_orig(x_min-2:x_max+6,y_min-2:y_max+3))
    ALLOCATE(yvel_orig(x_min-2:x_max+6,y_min-2:y_max+3))
    xvel_orig=xvel1
    yvel_orig=yvel1
  ENDIF
  
  WRITE(*,*) "Setup time ",timer_tod()-kernel_time

  WRITE(*,*) "Data initialised"

  IF(use_fortran_kernels) THEN
    WRITE(*,*) "Running Fortran kernel"
  ENDIF

  IF(use_C_kernels) THEN
    WRITE(*,*) "Running C kernel"
  ENDIF

  IF(reset_data) THEN
    WRITE(*,*) "Resetting data for each iteration"
  ELSE
    WRITE(*,*) "Not resetting data for each iteration"
  ENDIF

  acceleration_time=0.0


    accelerate_iter1_before = timer_tod()
    CALL accelerate_kernel_ocl(x_min, x_max, y_min, y_max, dt, first_iteration )

    CALL accelerate_ocl_call_clfinish()
    accelerate_iter1_after = timer_tod()







  kernel_time=timer_tod()

  DO iteration=1,its


#ifdef PROFILE_OCL_KERNELS
    CALL accelerate_kernel_ocl(x_min, x_max, y_min, y_max, dt, iter_timings(iteration) )
#else
    CALL accelerate_kernel_ocl(x_min, x_max, y_min, y_max, dt, first_iteration )
#endif

    !doesnt work on card so moving the timing outside the loop, also not required for accelerate
    IF(reset_data) THEN
      xvel1=xvel_orig
      yvel1=yvel_orig
    ENDIF

  ENDDO


  CALL accelerate_ocl_call_clfinish()
  acceleration_time=acceleration_time+(timer_tod()-kernel_time)

  CALL accelerate_ocl_readbuffers(xvel1, yvel1);

  CALL accelerate_ocl_call_clfinish()


  !IF(use_fortran_kernels) THEN
  !  DO iteration=1,its
  !    kernel_time=timer()
  !    CALL accelerate_kernel(x_min,                  &
  !                           x_max,                  &
  !                           y_min,                  &
  !                           y_max,                  &
  !                           dt,                     &
  !                           xarea,                  &
  !                           yarea,                  &
  !                           volume,                 &
  !                           density0,               &
  !                           pressure,               &
  !                           viscosity,              &
  !                           xvel0,                  &
  !                           yvel0,                  &
  !                           xvel1,                  &
  !                           yvel1,                  &
  !                           work_array1             )
  !    acceleration_time=acceleration_time+(timer()-kernel_time)
  !    IF(reset_data) THEN
  !      xvel1=xvel_orig
  !      yvel1=yvel_orig
  !    ENDIF
  !  ENDDO
  !ELSEIF(use_C_kernels)THEN
  !  DO iteration=1,its
  !    kernel_time=timer()
  !    CALL accelerate_kernel_c(x_min,                &
  !                           x_max,                  &
  !                           y_min,                  &
  !                           y_max,                  &
  !                           dt,                     &
  !                           xarea,                  &
  !                           yarea,                  &
  !                           volume,                 &
  !                           density0,               &
  !                           pressure,               &
  !                           viscosity,              &
  !                           xvel0,                  &
  !                           yvel0,                  &
  !                           xvel1,                  &
  !                           yvel1,                  &
  !                           work_array1             )
  !    acceleration_time=acceleration_time+(timer()-kernel_time)
  !    IF(reset_data) THEN
  !      xvel1=xvel_orig
  !      yvel1=yvel_orig
  !    ENDIF
  !  ENDDO
  !ENDIF


  WRITE(*,*) "Accelerate time ",acceleration_time 
  WRITE(*,*) "X vel ",SUM(xvel1)
  WRITE(*,*) "Y vel ",SUM(yvel1)

#ifdef PROFILE_OCL_KERNELS
    WRITE(*,*) "First kernel time: ", first_iteration
    WRITE(*,*) "Average of next ", SIZE(iter_timings(1:)), " iterations: ", SUM(iter_timings(1:))/(MAX(1, SIZE(iter_timings(1:))))
#endif

    WRITE(*,*) ""
    WRITE(*,*) "Tod before first kernel : ", accelerate_iter1_before
    WRITE(*,*) "Tod after first kernel  : ", accelerate_iter1_after
    WRITE(*,*) "First kernel launch took: ", accelerate_iter1_after-accelerate_iter1_before, " usec  ", &
                                             (accelerate_iter1_after-accelerate_iter1_before)/10**6, " secs"
    


    !WRITE(*,*) "Tod before main loop:    ", accelerate_main_before
    !WRITE(*,*) "Tod after main loop :    ", accelerate_main_after
    WRITE(*,*) "Main loop took   : ", acceleration_time, " usec ", (acceleration_time)/10**6, " secs"
    WRITE(*,*) "Average iteration: ", (acceleration_time)/its, " usec ", (acceleration_time)/its/10**6, " secs"

    !WRITE(*,*) "Main loop took   : ", accelerate_main_after-accelerate_main_before, " usec ", & 
    !                                 (accelerate_main_after-accelerate_main_before)/10**6, " secs"
    !WRITE(*,*) "Average iteration: ", (accelerate_main_after-accelerate_main_before)/its, " usec ", &
    !                                  (accelerate_main_after-accelerate_main_before)/its/10**6, " secs"




  ! Answers need checking
  DEALLOCATE(xarea)
  DEALLOCATE(yarea)
  DEALLOCATE(celldx)
  DEALLOCATE(celldy)
  DEALLOCATE(volume)
  DEALLOCATE(density0)
  DEALLOCATE(energy0)
  DEALLOCATE(pressure)
  DEALLOCATE(soundspeed)
  DEALLOCATE(viscosity)
  DEALLOCATE(xvel0)
  DEALLOCATE(yvel0)
  DEALLOCATE(xvel1)
  DEALLOCATE(yvel1)
  DEALLOCATE(work_array1)
  IF(reset_data) THEN
    DEALLOCATE(xvel_orig)
    DEALLOCATE(yvel_orig)
  ENDIF

END PROGRAM accelerate_driver

