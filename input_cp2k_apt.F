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

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: subsection, subsubsection

      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="APT", &
                          description="This section contains all information to run a "// &
                          "Atomic Polar Tensor (APT) calculation.", &
                          n_keywords=3, n_subsections=1, repeats=.FALSE.)

      NULLIFY (subsection, subsubsection, keyword)
      ! create keywords - only the apt torch file needed as keyword
      CALL keyword_create(keyword, __LOCATION__, name="APT_TORCH_FILE", &
                          description="File containing the input information for "// &
                          "the setup of the APT neural network (pytorch).", &
                          repeats=.FALSE., default_lc_val="aptnn.torch")
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)


      CALL keyword_create(keyword, __LOCATION__, name="E_FIELD_VECTOR", &
                          description="Electric field vector to use with apt "// &
                          "NN to generate forces.", &
                          repeats=.FALSE., &
                          n_var=3, type_of_var=real_t)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)


      ! print subsection
      CALL create_apt_print_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

   END SUBROUTINE create_apt_section

! **************************************************************************************************
!> \brief Creates the print section for the apt subsection
!> \param create_nnp_print_section the section to create the print apt information/forces
!> \date   2023-08-23
!> \author Kit Joll
! **************************************************************************************************
   SUBROUTINE create_apt_print_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: print_key

      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="PRINT", &
                          description="Section of possible print options in APT code.", &
                          n_keywords=0, n_subsections=2, repeats=.FALSE.)

      NULLIFY (print_key, keyword)

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

      CALL keyword_release(keyword)
   END SUBROUTINE create_apt_print_section
END MODULE input_cp2k_apt
