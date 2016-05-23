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

!>  @brief Controls the main hydro cycle.
!>  @author Wayne Gaudin
!>  @details Controls the top level cycle, invoking all the drivers and checks
!>  for outputs and completion.

SUBROUTINE hydro

  USE clover_module
  USE timestep_module
  USE viscosity_module
  USE PdV_module
  USE accelerate_module
  USE flux_calc_module
  USE advection_module
  USE reset_field_module

  IMPLICIT NONE

  REAL(KIND=8)    :: timer,timerstart,wall_clock,step_clock
  
  REAL(KIND=8)    :: grind_time,cells,rstep
  REAL(KIND=8)    :: step_time,step_grind
  REAL(KIND=8)    :: first_step,second_step

  timerstart = timer()

  DO

    step_time = timer()

    step = step + 1

    CALL timestep()
    
    CALL PdV(.TRUE.)

    CALL accelerate()

    CALL PdV(.FALSE.)

    CALL flux_calc()

    CALL advection()

    CALL reset_field()

    advect_x = .NOT. advect_x
  
    time = time + dt

    IF(summary_frequency.NE.0) THEN
      IF(MOD(step, summary_frequency).EQ.0) CALL field_summary()
    ENDIF
    IF(visit_frequency.NE.0) THEN
      IF(MOD(step, visit_frequency).EQ.0) CALL visit()
    ENDIF

    ! Sometimes there can be a significant start up cost that appears in the first step.
    ! Sometimes it is due to the number of MPI tasks, or OpenCL kernel compilation.
    ! On the short test runs, this can skew the results, so should be taken into account
    !  in recorded run times.
    IF(step.EQ.1) first_step=(timer() - step_time)
    IF(step.EQ.2) second_step=(timer() - step_time)

    IF(time+g_small.GT.end_time.OR.step.GE.end_step) THEN

      complete=.TRUE.
      CALL field_summary()
      IF(visit_frequency.NE.0) CALL visit()

      wall_clock=timer() - timerstart
      IF ( parallel%boss ) THEN
        WRITE(g_out,*)
        WRITE(g_out,*) 'Calculation complete'
        WRITE(g_out,*) 'Clover is finishing'
        WRITE(g_out,*) 'Wall clock ', wall_clock
        WRITE(g_out,*) 'First step overhead', first_step-second_step
        WRITE(    0,*) 'Wall clock ', wall_clock
        WRITE(    0,*) 'First step overhead', first_step-second_step
      ENDIF

      CALL clover_finalize

      EXIT

    END IF

    IF (parallel%boss) THEN
      wall_clock=timer()-timerstart
      step_clock=timer()-step_time
      WRITE(g_out,*)"Wall clock ",wall_clock
      WRITE(0    ,*)"Wall clock ",wall_clock
      cells = grid%x_cells * grid%y_cells
      rstep = step
      grind_time   = wall_clock/(rstep * cells)
      step_grind   = step_clock/cells
      WRITE(0    ,*)"Average time per cell ",grind_time
      WRITE(g_out,*)"Average time per cell ",grind_time
      WRITE(0    ,*)"Step time per cell    ",step_grind
      WRITE(g_out,*)"Step time per cell    ",step_grind

     END IF

  END DO

END SUBROUTINE hydro
