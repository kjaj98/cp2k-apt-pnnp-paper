!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2022 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief  Creates the APT section of the input
!> \author Kit Joll
!> \date   2023-08-23
! **************************************************************************************************
MODULE input_cp2k_apt
   USE cp_output_handling,              ONLY: cp_print_key_section_create,&
                                              medium_print_level
   USE cp_units,                        ONLY: cp_unit_to_cp2k
   USE input_keyword_types,             ONLY: keyword_create,&
                                              keyword_release,&
                                              keyword_type
   USE input_section_types,             ONLY: section_add_keyword,&
                                              section_add_subsection,&
                                              section_create,&
                                              section_release,&
                                              section_type
   USE input_val_types,                 ONLY: char_t,&
                                              real_t
   USE kinds,                           ONLY: dp
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'input_cp2k_apt'

   PUBLIC :: create_apt_section

CONTAINS

! **************************************************************************************************
!> \brief Create the input section for APT
!> \param section the section to create the APT section
!> \date   2023-08-23
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE create_apt_section(section)
      TYPE(section_type), POINTER                        :: section
      TYPE(section_type), POINTER                        :: subsection
      TYPE(keyword_type), POINTER                        :: keyword

      CPASSERT(.NOT. ASSOCIATED(section))

      write(*,*) "Creating APT section"


      CALL section_create(section, __LOCATION__, name="APT", &
                          description="This section contains all information to run a "// &
                          "Atomic Polar Tensor (APT) calculation.", &
                          n_keywords=2, n_subsections=2, repeats=.FALSE.)

      WRITE(*,*) "Creating subsections"
      ! print subsection
      NULLIFY(subsection, keyword)
      CALL create_apt_print_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

      WRITE(*,*) "Creating efield subsection"

      ! efield vec subsection
      CALL create_apt_e_field_vec_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)


      ! create torch file keyword
      CALL keyword_create(keyword, __LOCATION__, name="TORCH_FILE", &
                          description="File containing the torch model.", &
                          repeats=.FALSE., type_of_var=char_t,n_var=1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="APT_SERVER", &
                          description="Address to append to /tmp/qiskit_ unix socket prefix.", &
                          repeats=.FALSE., type_of_var=char_t,n_var=1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

   END SUBROUTINE create_apt_section



! **************************************************************************************************
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE create_apt_print_section(section)
      TYPE(section_type), POINTER                        :: section
      TYPE(section_type), POINTER                        :: print_key

      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="PRINT", &
                          description="Section of possible print options in APT code.", &
                          n_keywords=0, n_subsections=3, repeats=.FALSE.)

      NULLIFY (print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "TENSORS", &
                                       description="Controls the printing of the atomic polar tensors.", &
                                       print_level=medium_print_level, common_iter_levels=1)
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "FORCES", &
                                       description="Controls the printing of the apt forces.", &
                                       print_level=medium_print_level, common_iter_levels=1)
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

      CALL cp_print_key_section_create(print_key, __LOCATION__, "STDS", &
                                       description="Controls the printing of the apt standard deviations.", &
                                       print_level=medium_print_level, common_iter_levels=1)
      CALL section_add_subsection(section, print_key)
      CALL section_release(print_key)

   END SUBROUTINE create_apt_print_section
! **************************************************************************************************
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE create_apt_e_field_vec_section(section)
      TYPE(section_type), POINTER                        :: section,subsection
      TYPE(keyword_type), POINTER                        :: keyword

      WRITE(*,*) "calling section_create efield subsection"

      NULLIFY(keyword)
      CALL section_create(section, __LOCATION__, name="ELECTRIC_FIELD_VECTOR", &
                          description="Section to define E_FIELD_VECTOR in APT code.", &
                          n_keywords=1, n_subsections=3, repeats=.FALSE.)

      CALL keyword_create(keyword, __LOCATION__, name="MD_TIMESTEP", &
                          description="Timestep of md to evaluate field", &
                          repeats=.FALSE., type_of_var=real_t,n_var=1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)


      write(*,*) "Creating efield component subsections"
      NULLIFY(subsection)

      CALL create_efield_component_subsection(subsection,"X_COMPONENT")
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)
      NULLIFY(subsection)
      
      write(*,*) "after x component subsection"
      CALL create_efield_component_subsection(subsection,"Y_COMPONENT")
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)
      NULLIFY(subsection)
      
      write(*,*) "after y component subsection"

      CALL create_efield_component_subsection(subsection,"Z_COMPONENT")
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)
      NULLIFY(subsection)
      write(*,*) "after z component subsection"

   END SUBROUTINE create_apt_e_field_vec_section
! **************************************************************************************************
!> \brief Create a subsection for the electric field vector component
!> \param section The section to create the component subsection in
!> \param name The name of the component (X_component, Y_component, Z_component)
!> \author Kit Joll
! **************************************************************************************************
SUBROUTINE create_efield_component_subsection(section, component_name)
    TYPE(section_type), POINTER                        :: section
    TYPE(keyword_type), POINTER                        :: keyword
    CHARACTER(len=*), INTENT(IN)                       :: component_name


    CPASSERT(.NOT. ASSOCIATED(section))
    WRITE(*,*) "inside efield component subsection"

    ! Create the component section
    CALL section_create(section, __LOCATION__, name=component_name, &
                        description="Defines " // component_name // " of the electric field vector.", &
                        n_keywords=5, n_subsections=0, repeats=.FALSE.)

    WRITE(*,*) "Creating keywords"
    NULLIFY(keyword)
    CALL keyword_create(keyword, __LOCATION__, name="Function", &
                        description="Function to compute the " // component_name // " component.", &
                        repeats=.FALSE., type_of_var=char_t,n_var=1)
    CALL section_add_keyword(section, keyword)
    CALL keyword_release(keyword)


    CALL keyword_create(keyword, __LOCATION__, name="PARAMETERS", &
                        description="Parameters of function to compute the " // component_name // " component.", &
                        repeats=.FALSE., type_of_var=char_t,n_var=-1)
    CALL section_add_keyword(section, keyword)
    CALL keyword_release(keyword)

    CALL keyword_create(keyword, __LOCATION__, name="VALUES", &
                        description="Values of parameters of function to compute the " // component_name // " component.", &
                        repeats=.FALSE., type_of_var=real_t,n_var=-1)
    CALL section_add_keyword(section, keyword)
    CALL keyword_release(keyword)

    CALL keyword_create(keyword, __LOCATION__, name="UNITS", &
                        description="Units of values of parameters of function to compute the " // component_name // " component.", &
                        repeats=.FALSE., type_of_var=char_t,n_var=-1)
    CALL section_add_keyword(section, keyword)
    CALL keyword_release(keyword)

    CALL keyword_create(keyword, __LOCATION__, name="VARIABLES", &
                        description="Variables of function to compute the " // component_name // " component.", &
                        repeats=.FALSE., type_of_var=char_t,n_var=-1)
    CALL section_add_keyword(section, keyword)
    CALL keyword_release(keyword)
    



END SUBROUTINE create_efield_component_subsection
END MODULE input_cp2k_apt
