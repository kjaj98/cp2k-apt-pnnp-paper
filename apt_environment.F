!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2022 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief  Methods dealing with Atomic Polar Tensor (APT) calculations
!> \author Kit Joll
!> \date   2023-08-21
! **************************************************************************************************
MODULE apt_environment

   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cell_methods,                    ONLY: read_cell,&
                                              write_cell
   USE cell_types,                      ONLY: cell_release,&
                                              cell_type,&
                                              get_cell
   USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_get_default_unit_nr,&
                                              cp_logger_type, cp_to_string
   USE cp_para_types,                   ONLY: cp_para_env_type
   USE cp_parser_methods,               ONLY: parser_read_line,&
                                              parser_search_string
   USE cp_parser_types,                 ONLY: cp_parser_type,&
                                              parser_create,&
                                              parser_release,&
                                              parser_reset
   USE cp_subsys_methods,               ONLY: cp_subsys_create
   USE cp_subsys_types,                 ONLY: cp_subsys_set,&
                                              cp_subsys_type
   USE distribution_1d_types,           ONLY: distribution_1d_release,&
                                              distribution_1d_type
   USE distribution_methods,            ONLY: distribute_molecules_1d
   USE input_section_types,             ONLY: section_vals_get,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: default_path_length,&
                                              dp
   USE molecule_kind_types,             ONLY: molecule_kind_type,&
                                              write_molecule_kind_set
   USE molecule_types,                  ONLY: molecule_type
   USE message_passing,                 ONLY: mp_sync
   USE apt_environment_types,           ONLY: apt_env_set,apt_env_get, &
                                              apt_type
   USE particle_methods,                ONLY: write_fist_particle_coordinates,&
                                              write_particle_distances,&
                                              write_structure_data
   USE particle_types,                  ONLY: particle_type
   USE periodic_table,                  ONLY: get_ptable_info
   USE apt_utils,                       ONLY: driver_connect
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   LOGICAL, PARAMETER, PRIVATE :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'apt_environment'

   PUBLIC :: apt_init

CONTAINS

! **************************************************************************************************
!> \brief Read and initialize all the information for Atomic Polar Tensor (APT) calculations
!> \param apt_env ...
!> \param root_section ...
!> \param para_env ...
!> \param force_env_section ...
!> \param subsys_section ...
!> \param use_motion_section ...
!> \date   2023-08-21
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE apt_init(apt_env, root_section, para_env, force_env_section, subsys_section, &
                       use_motion_section)
      TYPE(apt_type), INTENT(INOUT), POINTER             :: apt_env
      TYPE(section_vals_type), INTENT(IN), POINTER       :: root_section
      TYPE(cp_para_env_type), INTENT(IN), POINTER        :: para_env
      TYPE(section_vals_type), INTENT(INOUT), POINTER    :: force_env_section, subsys_section
      LOGICAL, INTENT(IN)                                :: use_motion_section

      CHARACTER(len=*), PARAMETER                        :: routineN = 'apt_init'

      INTEGER                                            :: handle
      LOGICAL                                            :: explicit, use_ref_cell
      REAL(KIND=dp), DIMENSION(3)                        :: abc
      TYPE(cell_type), POINTER                           :: cell, cell_ref
      TYPE(cp_subsys_type), POINTER                      :: subsys
      TYPE(section_vals_type), POINTER                   :: cell_section, apt_section
      TYPE(cp_logger_type), POINTER                      :: logger
      INTEGER                                            ::  unit_nr

      CALL timeset(routineN, handle)

      CPASSERT(ASSOCIATED(apt_env))

      NULLIFY (cell_section, apt_section, cell, cell_ref, subsys,logger)

      logger => cp_get_default_logger()

      IF (logger%para_env%ionode) THEN
         unit_nr = cp_logger_get_default_unit_nr(logger)
      END IF

      IF (.NOT. ASSOCIATED(subsys_section)) THEN
         subsys_section => section_vals_get_subs_vals(force_env_section, "SUBSYS")
      END IF
      cell_section => section_vals_get_subs_vals(subsys_section, "CELL")
      apt_section => section_vals_get_subs_vals(force_env_section, "APT")
      CALL section_vals_get(apt_section, explicit=explicit)
      IF (.NOT. explicit) THEN
         CPWARN("APT section not explicitly stated. Using default file names.")
      END IF

      CALL apt_env_set(apt_env=apt_env, apt_input=apt_section, &
                       force_env_input=force_env_section)

      CALL read_cell(cell=cell, cell_ref=cell_ref, use_ref_cell=use_ref_cell, cell_section=cell_section, &
                     para_env=para_env)
      CALL get_cell(cell=cell, abc=abc)
      CALL write_cell(cell=cell, subsys_section=subsys_section)

      CALL cp_subsys_create(subsys, para_env, root_section, &
                            force_env_section=force_env_section, subsys_section=subsys_section, &
                            use_motion_section=use_motion_section)

      CALL apt_init_subsys(apt_env=apt_env, subsys=subsys, cell=cell, &
                           cell_ref=cell_ref, use_ref_cell=use_ref_cell, &
                           subsys_section=subsys_section)

      ! I FEEL LIKE I SHOULD CONNECT TO THE SERVER HERE
      ! I THINK I NEED TO SPAWN THE SERVER INDEPENDANT OF THE CP2K CALCULATION
      ! that's coolthough that's how it currently works
      

      CALL cell_release(cell)
      CALL cell_release(cell_ref)



      CALL timestop(handle)

   END SUBROUTINE apt_init
   SUBROUTINE apt_init_subsys(apt_env, subsys, cell, cell_ref, use_ref_cell, subsys_section)
      TYPE(apt_type), INTENT(INOUT), POINTER             :: apt_env
      TYPE(cp_subsys_type), INTENT(IN), POINTER          :: subsys
      TYPE(cell_type), INTENT(INOUT), POINTER            :: cell, cell_ref
      LOGICAL, INTENT(IN)                                :: use_ref_cell
      TYPE(section_vals_type), INTENT(IN), POINTER       :: subsys_section
      CHARACTER(len=*), PARAMETER                        :: routineN = 'apt_init_subsys'
      INTEGER                                            :: handle, natom
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(distribution_1d_type), POINTER                :: local_molecules, local_particles
      TYPE(molecule_kind_type), DIMENSION(:), POINTER    :: molecule_kind_set
      TYPE(molecule_type), DIMENSION(:), POINTER         :: molecule_set
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      CALL timeset(routineN, handle)

      NULLIFY (atomic_kind_set, molecule_kind_set, particle_set, molecule_set, &
               local_molecules, local_particles)

      particle_set => subsys%particles%els
      atomic_kind_set => subsys%atomic_kinds%els
      molecule_kind_set => subsys%molecule_kinds%els
      molecule_set => subsys%molecules%els

      !Print the molecule kind set
      CALL write_molecule_kind_set(molecule_kind_set, subsys_section)

      !Print the atomic coordinates
      CALL write_fist_particle_coordinates(particle_set, subsys_section)
      CALL write_particle_distances(particle_set, cell=cell, &
                                    subsys_section=subsys_section)
      CALL write_structure_data(particle_set, cell=cell, &
                                input_section=subsys_section)

      !Distribute molecules and atoms using the new data structures
      CALL distribute_molecules_1d(atomic_kind_set=atomic_kind_set, &
                                   particle_set=particle_set, &
                                   local_particles=local_particles, &
                                   molecule_kind_set=molecule_kind_set, &
                                   molecule_set=molecule_set, &
                                   local_molecules=local_molecules, &
                                   force_env_section=apt_env%force_env_input)

      natom = SIZE(particle_set)

      ALLOCATE (apt_env%apt_forces(natom, 3))

      apt_env%apt_forces(:, :) = 0.0_dp

      apt_env%apt_potential_energy = 0.0_dp

      ! Set up arrays for calculation:
      apt_env%num_atoms = natom
      ALLOCATE (apt_env%ele_ind(natom))
      ALLOCATE (apt_env%nuc_atoms(natom))
      ALLOCATE (apt_env%coord(3, natom))
      ALLOCATE (apt_env%atoms(natom))
      ALLOCATE (apt_env%sort(natom))
      ALLOCATE (apt_env%sort_inv(natom))

      CALL cp_subsys_set(subsys, cell=cell)

      CALL apt_env_set(apt_env=apt_env, subsys=subsys, &
                       cell_ref=cell_ref, use_ref_cell=use_ref_cell, &
                       local_molecules=local_molecules, &
                       local_particles=local_particles)

      CALL distribution_1d_release(local_particles)
      CALL distribution_1d_release(local_molecules)

      CALL apt_init_model(apt_env)

      CALL timestop(handle)
   END SUBROUTINE apt_init_subsys

   SUBROUTINE apt_init_model(apt_env)
      TYPE(apt_type), INTENT(INOUT), POINTER             :: apt_env
      CHARACTER(len=*), PARAMETER                        :: routineN = 'apt_init_model'
      TYPE(cp_logger_type), POINTER                      :: logger
      INTEGER                                            :: socket, unit_nr,handle
      REAL(KIND=dp), DIMENSION(:), POINTER               :: r_vals_ptr
      TYPE(section_vals_type), POINTER                   :: apt_input
      CALL timeset(routineN, handle)
      ! i only want to call this connection from one of the cp2k processes
      ! so i need to check if i'm the first process
      ! if i am, i need to call driver_connect(socket)
      ! if i'm not, i need to wait
      logger => cp_get_default_logger()
      unit_nr = cp_logger_get_default_unit_nr(logger)

      IF (logger%para_env%ionode) THEN
         WRITE (unit_nr, *) "APT| About to try to connect to the server"
      END IF


      call mp_sync(logger%para_env%group)
      IF (logger%para_env%ionode) THEN
         call driver_connect(socket)
      END IF
      call mp_sync(logger%para_env%group)
      ! Handle potential MPI error
      ! need to implement a comm barrier below here
      apt_env%socket = socket

      ! get the value for the electric field vector now too

      !!get the electric field vector 
      !allocate(r_vals_ptr(3))
      !CALL section_vals_val_get(apt_env%apt_input, "E_FIELD_VECTOR", r_vals=r_vals_ptr)
      !apt_env%electric_field=r_vals_ptr
      !deallocate(r_vals_ptr)
      !write(unit_nr,*) 'this is the electric field',apt_env%electric_field
      !FLUSH(unit_nr)

      CPASSERT(ASSOCIATED(apt_env))
      CALL apt_env_get(apt_env=apt_env, apt_input=apt_input)                       
      ALLOCATE (r_vals_ptr(3))
      CALL section_vals_val_get(apt_env%apt_input, "E_FIELD_VECTOR", r_vals=r_vals_ptr)
      apt_env%electric_field=r_vals_ptr
      DEALLOCATE(r_vals_ptr)

      IF (logger%para_env%ionode) THEN
         WRITE(unit_nr,*) ''
         WRITE(unit_nr,*) 'this is the electric field',apt_env%electric_field
         WRITE (unit_nr, *) "APT| Exiting apt_init_model"
         FLUSH(unit_nr)
      END IF
      call mp_sync(logger%para_env%group) 

      
      CALL timestop(handle)

   END SUBROUTINE apt_init_model

END MODULE apt_environment