!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2022 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief  Data types for apt environments
!> \author Kit Joll
!> \date   2023-08-23
! **************************************************************************************************
MODULE apt_environment_types
   USE atomic_kind_list_types,          ONLY: atomic_kind_list_create,&
                                              atomic_kind_list_release,&
                                              atomic_kind_list_type
   USE atomic_kind_types,               ONLY: atomic_kind_type
   USE cell_types,                      ONLY: cell_release,&
                                              cell_retain,&
                                              cell_type
   USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_release,&
                                              cp_subsys_set,&
                                              cp_subsys_type
   USE distribution_1d_types,           ONLY: distribution_1d_type
   USE input_section_types,             ONLY: section_vals_type
   USE kinds,                           ONLY: default_string_length,&
                                              dp
   USE molecule_kind_list_types,        ONLY: molecule_kind_list_create,&
                                              molecule_kind_list_release,&
                                              molecule_kind_list_type
   USE molecule_kind_types,             ONLY: molecule_kind_type
   USE molecule_list_types,             ONLY: molecule_list_create,&
                                              molecule_list_release,&
                                              molecule_list_type
   USE molecule_types,                  ONLY: molecule_type
   USE particle_list_types,             ONLY: particle_list_create,&
                                              particle_list_release,&
                                              particle_list_type
   USE particle_types,                  ONLY: particle_type
   USE virial_types,                    ONLY: virial_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'apt_environment_types'

   !> derived data types
   PUBLIC :: apt_type

   ! Public subroutines ***
   PUBLIC :: apt_env_release, &
             apt_env_set, &
             apt_env_get, &
             apt_env_create

! **************************************************************************************************
!> \brief Main data type collecting all relevant data for apt
!> \author Kit Joll
!> \date   2023-08-23
! **************************************************************************************************
   TYPE apt_type
      INTEGER                                             :: n_ele ! # elements
      CHARACTER(len=2), ALLOCATABLE, DIMENSION(:)         :: ele ! elements(n_ele)
      INTEGER, ALLOCATABLE, DIMENSION(:)                  :: nuc_ele ! elements(n_ele)
      INTEGER                                             :: num_atoms
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)         :: coord
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)      :: myforce
      CHARACTER(len=default_string_length), &
         ALLOCATABLE, DIMENSION(:)                        :: atoms
      REAL(KIND=dp), DIMENSION(:, :), ALLOCATABLE         :: apt_forces
      REAL(KIND=dp)                                       :: apt_potential_energy
      TYPE(cp_subsys_type), POINTER                       :: subsys
      TYPE(section_vals_type), POINTER                    :: apt_input, &
                                                             force_env_input
      TYPE(cell_type), POINTER                            :: cell_ref
      INTEGER, ALLOCATABLE, DIMENSION(:)                  :: ele_ind, nuc_atoms, sort, sort_inv
      LOGICAL                                             :: use_ref_cell
      INTEGER                                             :: socket
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:,:,:) :: apt_tensors
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:,:,:) :: apt_tensor_variance
      REAL(KIND=dp), DIMENSION(3) :: electric_field
   END TYPE apt_type
CONTAINS
! **************************************************************************************************
!> \brief Create data structure that holds all the information for apt calculations
!> \param apt_env ...
!> \date   2023-08-23
!> \author Kit Joll 
! **************************************************************************************************
   SUBROUTINE apt_env_create(apt_env)
      TYPE(apt_type), INTENT(OUT)                        :: apt_env
      NULLIFY (apt_env%subsys)
      NULLIFY (apt_env%apt_input)
      NULLIFY (apt_env%force_env_input)
      NULLIFY (apt_env%cell_ref)

      apt_env%use_ref_cell = .FALSE.


   END SUBROUTINE apt_env_create

! **************************************************************************************************
!> \brief Release data structure that holds all the information for apt calculations
!> \param apt_env ...
!> \date   2023-08-23
!> \author Kit Joll 
! **************************************************************************************************
   SUBROUTINE apt_env_release(apt_env)
      TYPE(apt_type), INTENT(INOUT)                      :: apt_env
      IF (ALLOCATED(apt_env%ele)) DEALLOCATE (apt_env%ele)
      IF (ALLOCATED(apt_env%nuc_ele)) DEALLOCATE (apt_env%nuc_ele)
      IF (ALLOCATED(apt_env%apt_forces)) DEALLOCATE (apt_env%apt_forces)
      IF (ALLOCATED(apt_env%ele_ind)) DEALLOCATE (apt_env%ele_ind)
      IF (ALLOCATED(apt_env%nuc_atoms)) DEALLOCATE (apt_env%nuc_atoms)
      IF (ALLOCATED(apt_env%sort)) DEALLOCATE (apt_env%sort)
      IF (ALLOCATED(apt_env%sort_inv)) DEALLOCATE (apt_env%sort_inv)
      IF (ALLOCATED(apt_env%coord)) DEALLOCATE (apt_env%coord)
      IF (ALLOCATED(apt_env%myforce)) DEALLOCATE (apt_env%myforce)
      IF (ALLOCATED(apt_env%atoms)) DEALLOCATE (apt_env%atoms)

      IF (ASSOCIATED(apt_env%subsys)) THEN
         CALL cp_subsys_release(apt_env%subsys)
      END IF
      IF (ASSOCIATED(apt_env%subsys)) THEN
         CALL cp_subsys_release(apt_env%subsys)
      END IF
      IF (ASSOCIATED(apt_env%cell_ref)) THEN
         CALL cell_release(apt_env%cell_ref)
      END IF

   END SUBROUTINE apt_env_release

! **************************************************************************************************
!> \brief Returns various attributes of the apt environment
!> \param apt_env ...
!> \param apt_forces ...
!> \param subsys the particles, molecules,... of this environment
!> \param atomic_kind_set The set of all atomic kinds involved
!> \param particle_set The set of all particles
!> \param local_particles All particles on this particular node
!> \param molecule_kind_set The set of all different molecule kinds involved
!> \param molecule_set The set of all molecules
!> \param local_molecules All molecules on this particular node
!> \param apt_input ...
!> \param force_env_input Pointer to the force_env input section
!> \param cell The simulation cell
!> \param cell_ref The reference simulation cell
!> \param use_ref_cell Logical which indicates if reference
!>                      simulation cell is used
!> \note
!>      For possible missing arguments see the attributes of
!>      apt_type
! **************************************************************************************************
   SUBROUTINE apt_env_get(apt_env, apt_forces, subsys, &
                          atomic_kind_set, particle_set, local_particles, &
                          molecule_kind_set, molecule_set, local_molecules, &
                          apt_input, force_env_input, cell, cell_ref, &
                          use_ref_cell,apt_potential_energy,virial)

      TYPE(apt_type), INTENT(IN)                         :: apt_env
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: apt_forces
      TYPE(cp_subsys_type), OPTIONAL, POINTER            :: subsys
      TYPE(atomic_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: atomic_kind_set
      TYPE(particle_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: particle_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_particles
      TYPE(molecule_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_kind_set
      TYPE(molecule_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_molecules
      TYPE(section_vals_type), OPTIONAL, POINTER         :: apt_input, force_env_input
      TYPE(cell_type), OPTIONAL, POINTER                 :: cell, cell_ref
      LOGICAL, INTENT(OUT), OPTIONAL                     :: use_ref_cell
      TYPE(atomic_kind_list_type), POINTER               :: atomic_kinds
      TYPE(molecule_kind_list_type), POINTER             :: molecule_kinds
      TYPE(molecule_list_type), POINTER                  :: molecules
      TYPE(particle_list_type), POINTER                  :: particles
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: apt_potential_energy
      TYPE(virial_type), OPTIONAL, POINTER               :: virial
      NULLIFY (atomic_kinds, particles, molecules, molecule_kinds)

      IF (PRESENT(apt_potential_energy)) THEN
        apt_potential_energy = apt_env%apt_potential_energy
      END IF
      IF (PRESENT(apt_forces)) apt_forces = apt_env%apt_forces

      IF (PRESENT(subsys)) subsys => apt_env%subsys
      IF (ASSOCIATED(apt_env%subsys)) THEN
         CALL cp_subsys_get(apt_env%subsys, &
                            atomic_kinds=atomic_kinds, &
                            particles=particles, &
                            molecule_kinds=molecule_kinds, &
                            molecules=molecules, &
                            local_molecules=local_molecules, &
                            local_particles=local_particles, &
                            virial=virial, &
                            cell=cell)
      END IF
      IF (PRESENT(atomic_kind_set)) atomic_kind_set => atomic_kinds%els
      IF (PRESENT(particle_set)) particle_set => particles%els
      IF (PRESENT(molecule_kind_set)) molecule_kind_set => molecule_kinds%els
      IF (PRESENT(molecule_set)) molecule_set => molecules%els
      IF (PRESENT(apt_input)) apt_input => apt_env%apt_input
      IF (PRESENT(force_env_input)) force_env_input => apt_env%force_env_input
      IF (PRESENT(cell_ref)) cell_ref => apt_env%cell_ref
      IF (PRESENT(use_ref_cell)) use_ref_cell = apt_env%use_ref_cell

   END SUBROUTINE apt_env_get

! **************************************************************************************************
!> \brief Sets various attributes of the apt environment
!> \param apt_env ...
!> \param apt_forces ...
!> \param subsys the particles, molecules,... of this environment
!> \param atomic_kind_set The set of all atomic kinds involved
!> \param particle_set The set of all particles
!> \param molecule_kind_set The set of all different molecule kinds involved
!> \param molecule_set The set of all molecules
!> \param apt_input ...
!> \param force_env_input Pointer to the force_env input section
!> \param cell_ref The reference simulation cell
!> \param use_ref_cell Logical which indicates if reference
!>                      simulation cell is used
!> \note
!>   For possible missing arguments see the attributes of apt_type
! **************************************************************************************************
   SUBROUTINE apt_env_set(apt_env, apt_forces, subsys, &
                          atomic_kind_set, particle_set, local_particles, &
                          molecule_kind_set, molecule_set, local_molecules, &
                          apt_input, force_env_input, cell_ref, &
                          use_ref_cell,apt_potential_energy)

      TYPE(apt_type), INTENT(INOUT)                      :: apt_env
      REAL(KIND=dp), DIMENSION(:, :), OPTIONAL, POINTER  :: apt_forces

      TYPE(cp_subsys_type), OPTIONAL, POINTER            :: subsys
      TYPE(atomic_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: atomic_kind_set
      TYPE(particle_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: particle_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_particles
      TYPE(molecule_kind_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_kind_set
      TYPE(molecule_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: molecule_set
      TYPE(distribution_1d_type), OPTIONAL, POINTER      :: local_molecules
      TYPE(section_vals_type), OPTIONAL, POINTER         :: apt_input, force_env_input
      TYPE(cell_type), OPTIONAL, POINTER                 :: cell_ref
      LOGICAL, INTENT(IN), OPTIONAL                      :: use_ref_cell

      TYPE(atomic_kind_list_type), POINTER               :: atomic_kinds
      TYPE(molecule_kind_list_type), POINTER             :: molecule_kinds
      TYPE(molecule_list_type), POINTER                  :: molecules
      TYPE(particle_list_type), POINTER                  :: particles
      REAL(KIND=dp), INTENT(OUT), OPTIONAL               :: apt_potential_energy

      IF (PRESENT(apt_potential_energy)) THEN
        apt_potential_energy = apt_env%apt_potential_energy
      END IF

      IF (PRESENT(apt_forces)) apt_env%apt_forces(:, :) = apt_forces


      IF (PRESENT(subsys)) THEN
         IF (ASSOCIATED(apt_env%subsys)) THEN
         IF (.NOT. ASSOCIATED(apt_env%subsys, subsys)) THEN
            CALL cp_subsys_release(apt_env%subsys)
         END IF
         END IF
         apt_env%subsys => subsys
      END IF
      IF (PRESENT(atomic_kind_set)) THEN
         CALL atomic_kind_list_create(atomic_kinds, els_ptr=atomic_kind_set)
         CALL cp_subsys_set(apt_env%subsys, atomic_kinds=atomic_kinds)
         CALL atomic_kind_list_release(atomic_kinds)
      END IF
      IF (PRESENT(particle_set)) THEN
         CALL particle_list_create(particles, els_ptr=particle_set)
         CALL cp_subsys_set(apt_env%subsys, particles=particles)
         CALL particle_list_release(particles)
      END IF
      IF (PRESENT(molecule_kind_set)) THEN
         CALL molecule_kind_list_create(molecule_kinds, els_ptr=molecule_kind_set)
         CALL cp_subsys_set(apt_env%subsys, molecule_kinds=molecule_kinds)
         CALL molecule_kind_list_release(molecule_kinds)
      END IF
      IF (PRESENT(molecule_set)) THEN
         CALL molecule_list_create(molecules, els_ptr=molecule_set)
         CALL cp_subsys_set(apt_env%subsys, molecules=molecules)
         CALL molecule_list_release(molecules)
      END IF
      IF (PRESENT(local_particles)) THEN
         CALL cp_subsys_set(apt_env%subsys, local_particles=local_particles)
      END IF
      IF (PRESENT(local_molecules)) THEN
         CALL cp_subsys_set(apt_env%subsys, local_molecules=local_molecules)
      END IF

      IF (PRESENT(apt_input)) apt_env%apt_input => apt_input

      IF (PRESENT(force_env_input)) THEN
         apt_env%force_env_input => force_env_input
      END IF
      IF (PRESENT(cell_ref)) THEN
         CALL cell_retain(cell_ref)
         CALL cell_release(apt_env%cell_ref)
         apt_env%cell_ref => cell_ref
      END IF
      IF (PRESENT(use_ref_cell)) apt_env%use_ref_cell = use_ref_cell
   END SUBROUTINE apt_env_set

END MODULE apt_environment_types
