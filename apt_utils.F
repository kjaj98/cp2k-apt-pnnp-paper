! The main program which runs our driver test case potentials
!
! Copyright (C) 2019, Bernat Font Garcia

! Permission is hereby granted, free of charge, to any person obtaining
! a copy of this software and associated documentation files (the
! "Software"), to deal in the Software without restriction, including
! without limitation the rights to use, copy, modify, merge, publish,
! distribute, sublicense, and/or sell copies of the Software, and to
! permit persons to whom the Software is furnished to do so, subject to
! the following conditions:
!
! The above copyright notice and this permission notice shall be included
! in all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
! EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
! MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
! IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
! CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
! TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
! SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
module apt_utils
    use sockets_interface,  ONLY: open_connect_socket, close_socket,  writebuffer, readbuffer,readbuffer_csocket
    use, intrinsic :: iso_c_binding
    USE cell_types,                      ONLY: cell_type
    USE atomic_kind_types,               ONLY: atomic_kind_type, get_atomic_kind
    USE input_section_types,             ONLY: section_vals_get,&
                                               section_vals_get_subs_vals,&
                                               section_vals_type,&
                                               section_vals_val_get     
    USE cp_units,                        ONLY: cp_unit_from_cp2k     

    USE cp_log_handling,                 ONLY: cp_get_default_logger,&
                                              cp_logger_get_default_unit_nr,&
                                              cp_logger_type, cp_to_string

    USE cp_subsys_types,                 ONLY: cp_subsys_get,&
                                              cp_subsys_type
    
    USE apt_environment_types,           ONLY: apt_env_get,&
                                              apt_type

    USE particle_types,                  ONLY: particle_type
    
    USE distribution_1d_types,           ONLY: distribution_1d_type

    USE kinds,                           ONLY: default_path_length,&
                                              default_string_length,&
                                              dp
#include "./base/base_uses.f90"
   
    IMPLICIT NONE
    PRIVATE 
    LOGICAL, PARAMETER, PRIVATE :: debug_this_module = .TRUE.
    CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'apt_utils'
    PUBLIC :: driver_subroutine, driver_connect, helpmessage, socket_close

    interface
        subroutine c_sleep(seconds) bind(c, name="sleep")
            import :: c_int
            integer(c_int), value :: seconds
        end subroutine c_sleep
    end interface

    CONTAINS
  SUBROUTINE driver_connect(apt_env, socket)
        implicit none
        REAL(kind=dp)                                      :: unit_conv
        !input variables
        TYPE(apt_type), INTENT(INOUT), POINTER             :: apt_env
        ! output variables 
        integer, intent(out) :: socket
        ! socket variables
        integer, parameter :: msglen=12   ! length of the headers of the driver/wrapper communication protocol
        integer            :: inet, port,unit_nr,retcode,tensor_fname_length    ! socket id & address of the server
        character(len=1024) :: tensor_fname
        TYPE(cp_logger_type), POINTER                      :: logger
        ! cell matrix
        type(cell_type), pointer :: cell
 


        NULLIFY (logger,cell)
        logger => cp_get_default_logger()

        call cp_subsys_get(apt_env%subsys, cell=cell)
        
        !only the root process will execute this anyway
        unit_nr = cp_logger_get_default_unit_nr(logger)
        unit_conv= cp_unit_from_cp2k(1.0_dp, "angstrom")



        WRITE(unit_nr,*) "APT| driver_connect: starting to look for file"
        ! need to start the server here - first set up checks for when server is ready to proceed
        ! intialize the socket variables
        ! read the file for the server machine ip here and then store it in host
        ! that way can use many nodes
        inet = 0
        port = 31415
        ! open port
        call open_connect_socket(socket, inet, port, TRIM(apt_env%host_address)//CHAR(0))
        socket = socket
        WRITE(unit_nr,*) "APT| driver_connect: socket opened with value "//cp_to_string(socket)
        FLUSH(unit_nr)

        ! now set up socket settings
        ! send the torch file  
        write(unit_nr,*) "APT| driver_connect: sending the torch file"
        FLUSH(unit_nr)
        call writebuffer(socket, 1) ! 1 for SET, 4 for the length of the message (integer)
        call writebuffer(socket, 0) ! 0 for the torch file, 4 for the length of the message (integer)
        call writebuffer(socket, len(TRIM(apt_env%torch_file))) ! send the length of the torch file string, 4 for integer 
        call writebuffer(socket, TRIM(apt_env%torch_file)//CHAR(0), len(TRIM(apt_env%torch_file))) ! send the torch file string
        call readbuffer(socket, retcode) ! read the response from the server
        write(unit_nr, *) "APT| driver_connect: received response from server: ", retcode
        flush(unit_nr)


        write(unit_nr,*) "APT| driver_connect: sending the cell matrix"
        FLUSH(unit_nr)
        call writebuffer(socket, 1) ! 1 for SET, 4 for the length of the message (integer)
        call writebuffer(socket, 1) ! 1 for cell, 4 for the length of the message (integer)
        call writebuffer(socket, 9*8) ! send the cell matrix 9 elements each 8 bytes (real dp)
        ! send in 3 calls because only takeas a 1d array
        call writebuffer(socket, cell%hmat(1,:)*unit_conv,3)
        call writebuffer(socket, cell%hmat(2,:)*unit_conv,3)
        call writebuffer(socket, cell%hmat(3,:)*unit_conv,3)
        call readbuffer(socket, retcode) ! read the response from the server
        write(unit_nr, *) "APT| driver_connect: received response from server: ", retcode
        flush(unit_nr)



        ! send value to the server that makes it print the apts at its end
        tensor_fname="server_apt_tensors.xyz"//CHAR(0)
        tensor_fname_length=22
        write(unit_nr,*) "APT| driver_connect: sending the print apts value"
        call writebuffer(socket,1) ! send a 1 to tell sever sending a setting
        call writebuffer(socket,3) ! 3 makes the server print the apts at the end
        call writebuffer(socket, tensor_fname_length)  ! send the legnth of the message
        call writebuffer(socket, tensor_fname, tensor_fname_length) ! send the message
        call readbuffer(socket, retcode) ! read the response from the serverk
        write(unit_nr, *) "APT| driver_connect: received response from server: ", retcode
        FLUSH(unit_nr)  


        




    end subroutine driver_connect  

    subroutine socket_close(socket)
        implicit none
        integer, intent(in) :: socket
        ! socket variables

        call close_socket(socket)

    end subroutine socket_close


    subroutine driver_subroutine(apt_env,socket)

        implicit none

        TYPE(cp_subsys_type), POINTER                      :: subsys
        TYPE(apt_type), INTENT(INOUT), POINTER             :: apt_env 
        ! socket variables
        integer, parameter :: msglen=12   ! length of the headers of the driver/wrapper communication protocol
        integer, intent(in) :: socket
        ! command line parsing

        ! socket communication buffers
        !character(len=12)                                  :: header

        ! data to send 
        integer                                            :: i, unit_nr,num_atoms                    ! LOOPING VARAIBLE
        ! data to receive
        !real(kind=dp)                                      :: tensor(3,3)
        integer                                            :: dim1, dim2, dim3 ,string_length, retval,tensorsizebytes
        TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
        TYPE(cp_logger_type), POINTER                      :: logger
        TYPE(distribution_1d_type), POINTER                :: local_particles
        TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
        REAL(kind=dp)                                      :: unit_conv
        character(len=:), allocatable                      :: tempval
        real(kind=dp),allocatable                          :: temptensors(:)         
        real(kind=dp), allocatable                         :: coords(:,:)
        character(len=:), allocatable                 :: atomic_kinds_string
        !real(kind=dp), allocatable                         :: flattened_coords(:)


        unit_conv= cp_unit_from_cp2k(1.0_dp, "angstrom")

        logger => cp_get_default_logger()
        unit_nr = cp_logger_get_default_unit_nr(logger)
    
        NULLIFY (particle_set, logger, local_particles, subsys, &
               atomic_kind_set)

        
        CPASSERT(ASSOCIATED(apt_env))
        CALL apt_env_get(apt_env=apt_env, particle_set=particle_set, &
                       subsys=subsys, local_particles=local_particles, &
                       atomic_kind_set=atomic_kind_set)

        CALL cp_subsys_get(subsys)
        write(unit_nr,*) "APT| driver_subroutine: got the subsys"
        flush(unit_nr)
        ! main loop
        particle_set => subsys%particles%els
        num_atoms = SIZE(particle_set)
        dim1=num_atoms
        dim2=3
        dim3=3
        !allocate memory and initialize for the arrays
        write(unit_nr,*) 'allocating memory for the arrays'
        flush(unit_nr)
        if (.not. allocated(apt_env%apt_tensors)) allocate(apt_env%apt_tensors(dim1,dim2,dim3))
        if (.not. allocated(apt_env%apt_tensor_variance)) allocate(apt_env%apt_tensor_variance(dim1,dim2,dim3))
        if (.not. allocated(coords)) allocate(coords(num_atoms,3))
        coords(:,:)=0.0_dp
        apt_env%apt_tensors(:,:,:)=0.0_dp
        apt_env%apt_tensor_variance(:,:,:)=0.0_dp
        ! loop over atomic kinds to get the element list
        atomic_kinds_string = ""
        do i=1,num_atoms
            !WRITE(unit_nr,*) "APT| driver_subroutine: particle_set(i)%atomic_kind%name: ", particle_set(i)%atomic_kind%name
            tempval=TRIM(particle_set(i)%atomic_kind%name)
            if (i == num_atoms) then
                atomic_kinds_string = trim(atomic_kinds_string) // tempval
            else
                atomic_kinds_string = trim(atomic_kinds_string) // tempval // ','
            endif
        end do
        string_length = len(trim(atomic_kinds_string))

        write(unit_nr,*) "APT| driver_subroutine: atomic_kinds_string: ", atomic_kinds_string
        write(unit_nr,*) "APT| driver_subroutine: string_length: ", string_length

        call writebuffer(socket, 0) ! send a 0 to tell server to predict
        call writebuffer(socket, num_atoms) ! send the number of atoms
        call writebuffer(socket, string_length) ! send the length of the atomic kinds string
        call writebuffer(socket,atomic_kinds_string,string_length) ! send the atomic kinds string

        !send the position data to the server
        write(unit_nr,*) "APT| driver_subroutine: sending the position data"
        call writebuffer(socket,num_atoms*3*8) ! send the size of the message
        do i = 1 , num_atoms
            call writebuffer(socket, particle_set(i)%r(1:3)*unit_conv, 3)
        end do

        write(unit_nr,*) "APT| driver_subroutine: sent the position data"
        flush(unit_nr)
        ! now receive the tensors from the server

        call readbuffer(socket, retval) ! read the response from the server
        write(unit_nr, *) "APT| driver_subroutine: received response from server: ", retval
        flush(unit_nr)
        
        
        write(unit_nr,*) "APT| driver_subroutine: receiving the tensors"
        flush(unit_nr)


        ! when tensors are sent, first we receive an integer with the size of the message
        ! then we receive the message
        allocate(temptensors(dim1*dim2*dim3))
        call readbuffer(socket, tensorsizebytes) ! read size of data in bytes 
        call readbuffer_csocket(socket, c_loc(apt_env%apt_tensors), tensorsizebytes) ! read the data (tensorsizebytes/8 is the number of elements in the tensor array
        !call readbuffer(socket, temptensors, tensorsizebytes/8)! no need for the *8 because it will do a double vector readbuffer anyway

        !write(unit_nr,*) "APT| driver_subroutine: received the tensors"
        !flush(unit_nr)
        !write(unit_nr,*) ""
        !write(unit_nr,*) temptensors
        !write(unit_nr,*) ""
        !flush(unit_nr)

        !call readbuffer(socket, reinterpret_cast<char*>(&apt_env%apt_tensors), dim1*dim2*dim3*8)
        !do ii=1,dim1 ! inside this loop receive the arrays 
        !    do jj=1,dim2
        !        do kk=1, dim3
        !            apt_env%apt_tensors(ii,jj,kk)= temptensors((kk-1)*dim2*dim1 + (jj-1)*dim1 + ii)
        !        end do
        !    end do
        !end do 
        write(unit_nr,*) "APT| driver_subroutine: allocated the tensors to env"
        flush(unit_nr)

        if (allocated(coords)) deallocate(coords)
        if (allocated(atomic_kinds_string)) deallocate(atomic_kinds_string)
        if (allocated(temptensors)) deallocate(temptensors)
        
        write(unit_nr,*) "APT| driver_subroutine: finished"
        flush(unit_nr)

    end subroutine driver_subroutine
    subroutine helpmessage
        write(*,*) " couldn't connect broski "
    end subroutine helpmessage
end module 