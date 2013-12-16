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

!>  @brief allocate and sets data for standalone mode
!>  @author Wayne Gaudin
!>  @details Calls user requested kernel in standalone mode

MODULE set_data_module

CONTAINS

SUBROUTINE set_data(x_min,x_max,y_min,y_max,     &
                    xarea,                       &
                    yarea,                       &
                    volume,                      &
                    density0,                    &
                    density1,                    &
                    energy0,                     &
                    energy1,                     &
                    viscosity,                   &
                    pressure,                    &
                    xvel0,                       &
                    xvel1,                       &
                    yvel0,                       &
                    yvel1,                       &
                    work_array1,                 &
                    work_array2,                 &
                    work_array3,                 &
                    work_array4,                 &
                    work_array5,                 &
                    work_array6,                 &
                    work_array7,                 &
                    work_array8,                 &
                    dt                           )

  IMPLICIT NONE

  INTEGER :: x_min,x_max,y_min,y_max
  REAL(KIND=8),OPTIONAL :: dt
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: xarea(:,:),yarea(:,:),volume(:,:)
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: density0(:,:),density1(:,:)
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: energy0(:,:),energy1(:,:)
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: pressure(:,:),viscosity(:,:)
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: xvel0(:,:),yvel0(:,:),xvel1(:,:),yvel1(:,:)
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: work_array1(:,:),work_array2(:,:),work_array3(:,:),work_array4(:,:)
  REAL(KIND=8),ALLOCATABLE,OPTIONAL :: work_array5(:,:),work_array6(:,:),work_array7(:,:),work_array8(:,:)

  ! Set the initial data, needs OMP pragmas for first touch


  IF(PRESENT(dt)) THEN
    dt=0.0001_8 ! Needs to be mesh specific
  ENDIF

  IF(PRESENT(xarea)) THEN
    ALLOCATE(xarea(x_min-2:x_max+3 ,y_min-2:y_max+2))
    xarea=1.0_8
  ENDIF
  IF(PRESENT(yarea)) THEN
    ALLOCATE(yarea(x_min-2:x_max+2 ,y_min-2:y_max+3))
    yarea=1.0_8
  ENDIF
  IF(PRESENT(volume)) THEN
    ALLOCATE(volume(x_min-2:x_max+2,y_min-2:y_max+2))
    volume=1.0_8
  ENDIF
  IF(PRESENT(density0)) THEN
    ALLOCATE(density0(x_min-2:x_max+2,y_min-2:y_max+2))
    density0(x_max/2:,:)=2.0_8
    density0(:x_max/2,:)=1.0_8
  ENDIF
  IF(PRESENT(pressure)) THEN
    ALLOCATE(pressure(x_min-2:x_max+2,y_min-2:y_max+2))
    pressure(x_max/2:,:)=2.0_8
    pressure(:x_max/2,:)=1.0_8
  ENDIF
  IF(PRESENT(viscosity)) THEN
    ALLOCATE(viscosity(x_min-2:x_max+2,y_min-2:y_max+2))
    viscosity=0.0_8
    viscosity(x_max/2,:)=0.1_8
  ENDIF
  IF(PRESENT(xvel0)) THEN
    ALLOCATE(xvel0(x_min-2:x_max+3,y_min-2:y_max+3))
    xvel0=1.0
  ENDIF
  IF(PRESENT(yvel0)) THEN
    ALLOCATE(yvel0(x_min-2:x_max+3,y_min-2:y_max+3))
    yvel0=1.0
  ENDIF
  IF(PRESENT(xvel1)) THEN
    ALLOCATE(xvel1(x_min-2:x_max+3,y_min-2:y_max+3))
    xvel1=1.0
  ENDIF
  IF(PRESENT(yvel1)) THEN
    ALLOCATE(yvel1(x_min-2:x_max+3,y_min-2:y_max+3))
    yvel1=1.0
  ENDIF
  IF(PRESENT(work_array1)) THEN
    ALLOCATE(work_array1(x_min-2:x_max+3,y_min-2:y_max+3))
  ENDIF

END SUBROUTINE set_data

END MODULE set_data_module
    
