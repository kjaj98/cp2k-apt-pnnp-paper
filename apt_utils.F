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
    use f90sockets,  ONLY: open_socket, close_socket,  writebuffer, readbuffer
    use, intrinsic :: iso_c_binding

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
  SUBROUTINE driver_connect(socket)
        implicit none
        ! output variables 
        integer, intent(out) :: socket
        ! socket variables
        integer, parameter :: msglen=12   ! length of the headers of the driver/wrapper communication protocol
        integer inet, port,unit_nr        ! socket id & address of the server
        character(len=1024) :: host, filename
        logical :: file_exists
        integer :: iwait
        TYPE(cp_logger_type), POINTER                      :: logger
        CHARACTER(LEN=100) :: hostname
        !CHARACTER(LEN=500) :: sruncmd
        INTEGER :: ierr

        

        NULLIFY (logger)
        logger => cp_get_default_logger()
        !only the root process will execute this anyway
        unit_nr = cp_logger_get_default_unit_nr(logger)
        WRITE(unit_nr,*) "APT| driver_connect: starting to look for file"
        ! need to start the server here - first set up checks for when server is ready to proceed
        WRITE(unit_nr,*) "APT| getting the hostname" 
        ! only one ip will be printed because only zeroth rank process calls driver_connect
        filename="ready.txt"
        file_exists = .false.
       ! now launch the server from root zero 
        CALL EXECUTE_COMMAND_LINE("hostname > hostname.cp2k")
        OPEN(UNIT=10, FILE='hostname.cp2k', STATUS='OLD', ACTION='READ', IOSTAT=ierr)
        IF (ierr == 0) THEN
        READ(10, '(A)') hostname
        CLOSE(UNIT=10)
        ELSE
        CPABORT('Error opening file and reading 0th rank machine ip address')
        STOP
        END IF
       ! WRITE(sruncmd, '(A, A, A)') "OMP_NUM_THREADS=128 srun --overlap --oversubscribe --nodes=8 --ntasks=8 --cpus-per-task=128 --nodelist=", TRIM(hostname), " python3 main.py > my_output.txt 2>&1 &"
       ! WRITE(unit_nr,*) "The srun command about to be run is ", sruncmd
       ! CALL EXECUTE_COMMAND_LINE(TRIM(sruncmd),wait=.false.)
        WRITE(unit_nr,*) "APT| driver_connect: server launched"
        ! now the root process should wait until the file exists
        iwait = 0
        WRITE(unit_nr,*) "APT| driver_connect: entering loop checking file existance"
        FLUSH(unit_nr)
        do while (.NOT. file_exists .AND. iwait < 500)   ! example max wait attempts
            INQUIRE(FILE=filename, EXIST=file_exists)
            IF (.NOT. file_exists) THEN
                CALL c_sleep(2)  ! Wait for 2 seconds
                WRITE(unit_nr,*) "APT| driver_connect: waiting for file to exist", iwait
                FLUSH(unit_nr)
                iwait = iwait + 1
            END IF
        END DO
        IF (file_exists) THEN
            WRITE(unit_nr,*) "APT| driver_connect: file exists"
        ELSE
            CPABORT("APT| driver_connect: file does not exist")
        END IF
        FLUSH(unit_nr)
        ! intialize the socket variables
        ! read the file for the server machine ip here and then store it in host
        ! that way can use many nodes
        inet = 1
        host = "localhost"//char(0)
        port = 31415
        ! open port
        call open_socket(socket, inet, port, host)
        socket = socket
        WRITE(unit_nr,*) "APT| driver_connect: socket opened with value "//cp_to_string(socket)
        FLUSH(unit_nr)
        !CPABORT("haven't implemented anything after attempting to connect to the server")
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
        character(len=12)                                  :: header

        ! data to send 
        integer                                            :: i, unit_nr,num_atoms                    ! LOOPING VARAIBLE
        ! data to receive
        real(kind=dp)                                      :: tensor(3,3)
        real(kind=dp), allocatable                         :: msgbuffer(:)
        integer                                            :: dim1, dim2, dim3, ii, jj,kk,opcount
        TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
        TYPE(cp_logger_type), POINTER                      :: logger
        TYPE(distribution_1d_type), POINTER                :: local_particles
        TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
        REAL(kind=dp)                                      :: unit_conv
        character(len=:), allocatable                      :: ele_list
        character(len=10)                                  :: num_atoms_str    
        real(kind=dp), allocatable                         :: coords(:,:)
        real(kind=dp), allocatable                         :: flattened_coords(:)
        



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
        ! main loop
        particle_set => subsys%particles%els
        num_atoms = SIZE(particle_set)

        ! initialize the operation counter and dimensions of the array
        opcount=0
        
        dim1=num_atoms
        dim2=3
        dim3=3
        write(num_atoms_str,'(I0)') num_atoms

        !allocate memory for the array
        write(unit_nr,*) 'allocating memory for the arrays'
        if (.not. allocated(apt_env%apt_tensors)) allocate(apt_env%apt_tensors(dim1,dim2,dim3))
        if (.not. allocated(apt_env%apt_tensor_variance)) allocate(apt_env%apt_tensor_variance(dim1,dim2,dim3))
        if (.not. allocated(coords)) allocate(coords(num_atoms,3))
        ele_list=""
        coords(:,:)=0.0_dp
        apt_env%apt_tensors(:,:,:)=0.0_dp
        apt_env%apt_tensor_variance(:,:,:)=0.0_dp


        write(unit_nr,*) 'entering the send and receive loop'
        flush(unit_nr)
        do while (opcount<2) ! loops forever (or until the wrapper ends!)
            ! reads from the socket one message header
            write(unit_nr,*) 'checking if this is a blocking function or not'
            flush(unit_nr)
            call readbuffer(socket, header, msglen)
            if (trim(header) == "SENDDATA") then  ! Server wants to send data to the driver
                if (.not. allocated(msgbuffer)) allocate(msgbuffer(size(tensor)))
                do ii=1,dim1 ! inside this loop receive the arrays 
                    call readbuffer(socket, msgbuffer, size(tensor))
                    tensor = reshape(msgbuffer,shape(tensor))
                    do jj=1,dim2
                        do kk=1, dim3
                            apt_env%apt_tensors(ii,jj,kk)=tensor(jj,kk)
                        end do
                    end do
                end do
                do ii=1,dim1 ! inside this loop receive the arrays 
                    call readbuffer(socket, msgbuffer, size(tensor))
                    tensor = reshape(msgbuffer,shape(tensor))
                    do jj=1,dim2
                        do kk=1, dim3
                            apt_env%apt_tensor_variance(ii,jj,kk)=tensor(jj,kk)
                        end do
                    end do
                end do
                if (allocated(msgbuffer)) deallocate(msgbuffer)
                opcount=opcount+1
            elseif (trim(header) == "GETDATA") then  ! Server signaling driver to send data
                ! send the number of atoms
                ! convert to string first
                call writebuffer(socket,num_atoms_str,len(num_atoms_str))
                ! prepare the coordinates array
                do i=1,num_atoms
                    coords(i,:)=particle_set(i)%r(1:3)*unit_conv
                end do

                allocate(flattened_coords(num_atoms*3))
                flattened_coords=reshape(coords, [num_atoms*3])
                ! send the coordinates array

                call writebuffer(socket,flattened_coords,size(flattened_coords))
                if (allocated(flattened_coords)) deallocate(flattened_coords)
                if (allocated(coords)) deallocate(coords)
                ! deallocate the coordinate memory
                ! print statement to check loop finished and memory deallocated
                ! sending end of frame message
                opcount=opcount+1
                ! check that end of frame message sent
            elseif (trim(header) == "") then ! if an empty message then do loop again
                write(*,*) "empty message"
            else 
                write(*,*) " unexpected header ", header
                stop "ended"
            endif
        enddo
    end subroutine driver_subroutine
    subroutine helpmessage
        write(*,*) " couldn't connect broski "
    end subroutine helpmessage
end module 