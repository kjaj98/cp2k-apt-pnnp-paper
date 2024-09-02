!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2022 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief  Methods dealing with atomic polar tensor calculations
!> \author Kit Joll
!> \date   2023-08-22
! **************************************************************************************************
MODULE apt_force
   USE atomic_kind_types,               ONLY: atomic_kind_type, get_atomic_kind
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_get_default_unit_nr,&
                                              cp_logger_type, cp_to_string
   USE cp_output_handling,              ONLY: cp_p_file,&
                                              cp_print_key_finished_output,&
                                              cp_print_key_should_output,&
                                              cp_print_key_unit_nr
   USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_type
   USE cp_units,                        ONLY: cp_unit_from_cp2k
   USE distribution_1d_types,           ONLY: distribution_1d_type
   USE input_section_types,             ONLY: section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: default_path_length,&
                                              default_string_length,&
                                              dp
   USE message_passing,                 ONLY: mp_sync, mp_bcast
                                              
   USE apt_environment_types,           ONLY: apt_env_get,&
                                              apt_type
   USE particle_types,                  ONLY: particle_type
   USE periodic_table,                  ONLY: get_ptable_info
   USE physcon,                         ONLY: angstrom
   USE virial_types,                    ONLY: virial_type

   USE apt_utils,                       ONLY: driver_connect, driver_subroutine
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   LOGICAL, PARAMETER, PRIVATE :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'apt_force'

   PUBLIC :: apt_calc_energy_force

CONTAINS

! **************************************************************************************************
!> \brief Calculate the energy and force for a given configuration with the APT
!> \param apt ...
!> \param calc_forces ...
!> \date   2023-08-22
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE apt_calc_energy_force(apt_env,calc_forces)
      TYPE(apt_type), INTENT(INOUT), POINTER             :: apt_env
      CHARACTER(len=*), PARAMETER :: routineN = 'apt_calc_energy_force'
      INTEGER                                            :: handle,  &
                                                             unit_nr, &
                                                             num_atoms,i,j,k
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(cp_subsys_type), POINTER                      :: subsys
      TYPE(distribution_1d_type), POINTER                :: local_particles
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(section_vals_type), POINTER                   :: apt_input
      LOGICAL, INTENT(IN)                                :: calc_forces
      TYPE(virial_type), POINTER                         :: virial
      LOGICAL                                            :: calc_stress
      TYPE(section_vals_type), POINTER                   :: print_section


      ! socket variables
      integer, parameter :: msglen=12   ! length of the headers of the driver/wrapper communication protocol

      if (calc_forces) then
         CALL timeset(routineN, handle)

         NULLIFY (particle_set, logger, local_particles, subsys, &
                  atomic_kind_set)
         logger => cp_get_default_logger()
         unit_nr = cp_logger_get_default_unit_nr(logger)

         IF (logger%para_env%ionode) THEN
            WRITE(unit_nr,*) 'entered apt_calc_energy_force'
            FLUSH(unit_nr)
         END IF



         CPASSERT(ASSOCIATED(apt_env))
         CALL apt_env_get(apt_env=apt_env, apt_input=apt_input,particle_set=particle_set, &
                        subsys=subsys, local_particles=local_particles, &
                        atomic_kind_set=atomic_kind_set)

         IF (logger%para_env%ionode) THEN
            WRITE(unit_nr,*) 'got the apt_env'
            FLUSH(unit_nr)
         END IF

         CALL cp_subsys_get(subsys,virial=virial)
         calc_stress = virial%pv_availability .AND. (.NOT. virial%pv_numer)
         IF (calc_stress .AND. .NOT. calc_forces) CPABORT('Stress cannot be calculated without forces')
         ! get the coords
         particle_set => subsys%particles%els
         num_atoms = SIZE(particle_set)
         ! send the coords to the server

         call mp_sync(logger%para_env%group)
         IF (logger%para_env%ionode) THEN
            WRITE(unit_nr,*) 'about to start sending the coords to the server'
            FLUSH(unit_nr)
            CALL driver_subroutine(apt_env,socket=apt_env%socket)
         END IF
         call mp_sync(logger%para_env%group)

         call mp_sync(logger%para_env%group)
         if (logger%para_env%ionode) then
            apt_env%apt_potential_energy=0.0_dp
            if (.not. allocated (apt_env%apt_forces)) then
               allocate(apt_env%apt_forces(apt_env%num_atoms,3))
            end if
            WRITE(unit_nr,*) ''
            WRITE(unit_nr,*) 'E FIELD =', apt_env%electric_field
            WRITE(unit_nr,*) ''
            ! Compute forces
            FLUSH(unit_nr)
            DO i = 1, num_atoms
               apt_env%apt_forces(i, :) = 0.0_dp
               DO j = 1, 3
                  DO k = 1, 3
                  !apt_env%apt_forces(i, j) =apt_env%apt_forces(i,j) +apt_env%apt_tensors(i, j, k) * apt_env%electric_field(k)
                  apt_env%apt_forces(i, j) =apt_env%apt_forces(i,j) +apt_env%apt_tensors(i, k, j) * apt_env%electric_field(k)
                  END DO
               END DO
            END DO


         end if
         call mp_sync(logger%para_env%group)
         ! now need to broadcast the forces and energy too all processes 
         DO i =1, num_atoms
         !loop through the forces and broadcast them
            CALL mp_bcast(apt_env%apt_forces(i,:), logger%para_env%source, logger%para_env%group)
         END DO
         CALL mp_bcast(apt_env%apt_potential_energy, logger%para_env%source, logger%para_env%group)
         DO i = 1, num_atoms
         particle_set(i)%f(:) = apt_env%apt_forces(i, :)
         END DO

         IF (logger%para_env%ionode) THEN
            WRITE(unit_nr,*) 'leaving apt_force_prediction'
            FLUSH(unit_nr)
         END IF

         ! print the forces 
         print_section => section_vals_get_subs_vals(apt_env%apt_input, "PRINT")
         CALL apt_print(apt_env, print_section)
         CALL timestop(handle)
      end if


   END SUBROUTINE apt_calc_energy_force


! **************************************************************************************************
!> \brief Print properties according to the requests in input file
!> \param apt_env ...
!> \param print_section ...
!> \date   2023-08-22
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE apt_print(apt_env, print_section)
      TYPE(apt_type), INTENT(INOUT)                      :: apt_env 
      TYPE(section_vals_type), INTENT(IN), POINTER       :: print_section

      INTEGER                                            :: unit_nr
      LOGICAL                                            ::  file_is_new
      TYPE(cp_logger_type), POINTER                      :: logger
      TYPE(section_vals_type), POINTER                   :: print_key

      NULLIFY (logger, print_key)
      logger => cp_get_default_logger()

      print_key => section_vals_get_subs_vals(print_section, "TENSORS")
      IF (BTEST(cp_print_key_should_output(logger%iter_info, print_key), cp_p_file)) THEN
         unit_nr = cp_print_key_unit_nr(logger, print_key, extension=".xyz", &
                                        middle_name="apt_tensors", is_new_file=file_is_new)
         IF (unit_nr > 0) CALL apt_print_tensors(apt_env, unit_nr)
         CALL cp_print_key_finished_output(unit_nr, logger, print_key)
      END IF

      print_key => section_vals_get_subs_vals(print_section, "FORCES")
      IF (BTEST(cp_print_key_should_output(logger%iter_info, print_key), cp_p_file)) THEN
         unit_nr = cp_print_key_unit_nr(logger, print_key, extension=".xyz", &
                                        middle_name="apt_forces", is_new_file=file_is_new)
         IF (unit_nr > 0) CALL apt_print_forces(apt_env, unit_nr)
         CALL cp_print_key_finished_output(unit_nr, logger, print_key)
      END IF

      print_key => section_vals_get_subs_vals(print_section, "STDS")
      IF (BTEST(cp_print_key_should_output(logger%iter_info, print_key), cp_p_file)) THEN
         unit_nr = cp_print_key_unit_nr(logger, print_key, extension=".xyz", &
                                        middle_name="apt_stds", is_new_file=file_is_new)
         IF (unit_nr > 0) CALL apt_print_std(apt_env, unit_nr)
         CALL cp_print_key_finished_output(unit_nr, logger, print_key)
      END IF
   END SUBROUTINE apt_print
! **************************************************************************************************
!> \brief Print APT tensors  
!> \param apt_env...
!> \param unit_nr ...
!> \param file_is_new ...
!> \date   2023-08-22
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE apt_print_tensors(apt_env, unit_nr)
      TYPE(apt_type), INTENT(INOUT)                      :: apt_env 
      INTEGER, INTENT(IN)                                :: unit_nr
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      REAL(KIND=dp)                                      :: unit_conv   
      INTEGER                                            :: ii,jj,kk
      NULLIFY (particle_set)
      CALL apt_env_get(apt_env=apt_env, particle_set=particle_set)
      unit_conv= cp_unit_from_cp2k(1.0_dp, "angstrom")

      WRITE(unit_nr,*) apt_env%num_atoms
      WRITE(unit_nr,*) 'fmt="pa"'
      do ii=1,apt_env%num_atoms
         write(unit_nr,'(A)',advance='no') particle_set(ii)%atomic_kind%element_symbol
         write(unit_nr, '(3F20.10,2(1X))', advance='no') particle_set(ii)%r(:)*unit_conv
         do jj=1,3
            do kk=1,3
               write(unit_nr,'(F20.10,1X)',advance='no') apt_env%apt_tensors(ii,jj,kk)
            end do
         end do
         write(unit_nr,*)
      end do
      
   END SUBROUTINE apt_print_tensors

! **************************************************************************************************
!> \brief Print APT std tensors  
!> \param apt_env...
!> \param unit_nr ...
!> \param file_is_new ...
!> \date   2023-08-22
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE apt_print_std(apt_env, unit_nr)
      TYPE(apt_type), INTENT(INOUT)                      :: apt_env 
      INTEGER, INTENT(IN)                                :: unit_nr
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set

      INTEGER                                            :: ii,jj,kk
      NULLIFY (particle_set)
      CALL apt_env_get(apt_env=apt_env, particle_set=particle_set)

      
      WRITE(unit_nr,*) apt_env%num_atoms
      WRITE(unit_nr,*) 'fmt="s"'
      do ii=1,apt_env%num_atoms
         write(unit_nr,'(A)',advance='no') particle_set(ii)%atomic_kind%element_symbol
         do jj=1,3
            do kk=1,3
               write(unit_nr,'(F20.10,1X)',advance='no') apt_env%apt_tensor_variance(ii,jj,kk)
            end do
         end do
         write(unit_nr,*)
      end do
   END SUBROUTINE apt_print_std
! **************************************************************************************************
!> \brief Print apt forces
!> \param apt ...
!> \param print_key ...
!> \date   2023-08-22
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE apt_print_forces(apt_env, unit_nr)
      TYPE(apt_type), INTENT(INOUT)                      :: apt_env
      INTEGER, INTENT(IN)                                :: unit_nr  
      INTEGER                                            :: i
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set

      NULLIFY (particle_set)
      CALL apt_env_get(apt_env=apt_env, particle_set=particle_set)

      WRITE(unit_nr,*) apt_env%num_atoms
      WRITE(unit_nr,*) '# APT derived forces'
      ! Output forces
      DO i = 1, apt_env%num_atoms
         WRITE(unit_nr,*) particle_set(i)%atomic_kind%element_symbol,apt_env%apt_forces(i, :)
      END DO
   END SUBROUTINE apt_print_forces
END MODULE apt_force
