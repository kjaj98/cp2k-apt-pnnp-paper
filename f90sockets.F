module f90sockets
    use iso_c_binding
#include "./base/base_uses.f90"
    implicit none
    LOGICAL, PARAMETER, PRIVATE :: debug_this_module = .TRUE.
    CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'f90_sockets'

    interface writebuffer ! Generic functions for write; depending on the data type each of this ones is selected
        module procedure writebuffer_s, writebuffer_d, writebuffer_dv, writebuffer_i, writebuffer_sp, writebuffer_spv
    end interface

    interface readbuffer ! Generic functions for read; depending on the data type each of this ones is selected
        module procedure readbuffer_s, readbuffer_d, readbuffer_dv, readbuffer_i, readbuffer_sp, readbuffer_spv
    end interface 

    interface
        subroutine open_csocket(psockfd, inet, port, host) bind(c, name="open_socket")
            use iso_c_binding
            integer(kind=c_int)                      :: psockfd, inet, port
            character(kind=c_char), dimension(*)     :: host
        end subroutine open_csocket


        subroutine writebuffer_csocket(psockfd, pdata, plen) bind(c, name="writebuffer")
            use iso_c_binding
            integer(kind=c_int)                      :: psockfd
            type(c_ptr), value                       :: pdata
            integer(kind=c_int)                      :: plen
        end subroutine writebuffer_csocket       

        subroutine readbuffer_csocket(psockfd, pdata, plen) bind(c, name="readbuffer")
            use iso_c_binding
            integer(kind=c_int)                      :: psockfd
            type(c_ptr), value                       :: pdata
            integer(kind=c_int)                      :: plen
        end subroutine readbuffer_csocket   
    end interface

    interface
        subroutine close_csocket(psockfd) bind(c, name="close")
            use iso_c_binding
            integer(kind=c_int), value :: psockfd
        end subroutine close_csocket
    end interface
            

contains
    subroutine close_socket(psockfd)
        implicit none
        integer, intent(in) :: psockfd
        call close_csocket(psockfd)
    end subroutine
   
    subroutine open_socket(psockfd, inet, port, host)      
        implicit none
        integer, intent(in) :: inet, port
        integer, intent(out) :: psockfd
        character(len=1024), intent(in) :: host
        character(len=1,kind=c_char) :: chost(1024)

        call fstr2cstr(host, chost)
        call open_csocket(psockfd, inet, port, chost)
    end subroutine


    subroutine fstr2cstr(fstr, cstr, plen)
        implicit none
        character(len=*), intent(in) :: fstr
        character(len=1,kind=c_char), intent(out) :: cstr(:)
        integer, intent(in), optional :: plen

        integer i,n
        if (present(plen)) then
            n = plen
            do i=1,n
                cstr(i) = fstr(i:i)
            enddo
        else
            n = len_trim(fstr)
            do i=1,n
                cstr(i) = fstr(i:i)
            enddo
            cstr(n+1) = c_null_char
        end if
    end subroutine

    subroutine writebuffer_d(psockfd, fdata)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        real(kind=8), intent(in)                :: fdata

        real(kind=c_double), target             :: cdata

        cdata = fdata
        call writebuffer_csocket(psockfd, c_loc(cdata), 8)
    end subroutine

    subroutine writebuffer_sp(psockfd, fdata)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        real(kind=4), intent(in)                :: fdata

        real(kind=c_float), target             :: cdata

        cdata = fdata
        call writebuffer_csocket(psockfd, c_loc(cdata), 4)
    end subroutine


    subroutine writebuffer_i(psockfd, fdata)
        use iso_c_binding
        integer, intent(in)                     :: psockfd, fdata

        integer(kind=c_int), target             :: cdata

        cdata = fdata
        call writebuffer_csocket(psockfd, c_loc(cdata), 4)
    end subroutine

    subroutine writebuffer_s(psockfd, fstring, plen)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        character(len=*), intent(in)            :: fstring
        integer, intent(in)                     :: plen

        integer                                 :: i
        character(len=1, kind=c_char), target   :: cstring(plen)

        do i = 1,plen
            cstring(i) = fstring(i:i)
        enddo
        call writebuffer_csocket(psockfd, c_loc(cstring(1)), plen)
    end subroutine

    subroutine writebuffer_dv(psockfd, fdata, plen)
        use iso_c_binding  
        integer, intent(in)                     :: psockfd, plen
        real(kind=8), intent(in), target        :: fdata(plen)

        call writebuffer_csocket(psockfd, c_loc(fdata(1)), 8*plen)
    end subroutine

    subroutine writebuffer_spv(psockfd, fdata, plen)
        use iso_c_binding
        integer, intent(in)                     :: psockfd, plen
        real(kind=4), intent(in), target        :: fdata(plen)

        call writebuffer_csocket(psockfd, c_loc(fdata(1)), 4*plen)
    end subroutine

    subroutine readbuffer_d(psockfd, fdata)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        real(kind=8), intent(out)               :: fdata

        real(kind=c_double), target             :: cdata

        call readbuffer_csocket(psockfd, c_loc(cdata), 8)
        fdata=cdata
    end subroutine

    subroutine readbuffer_sp(psockfd, fdata)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        real(kind=4), intent(out)               :: fdata

        real(kind=c_float), target              :: cdata

        call readbuffer_csocket(psockfd, c_loc(cdata), 4)
        fdata=cdata
    end subroutine

    subroutine readbuffer_i(psockfd, fdata)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        integer, intent(out)                    :: fdata

        integer(kind=c_int), target             :: cdata

        call readbuffer_csocket(psockfd, c_loc(cdata), 4)
        fdata = cdata
    end subroutine

    subroutine readbuffer_s(psockfd, fstring, plen)
        use iso_c_binding
        integer, intent(in)                     :: psockfd
        character(len=*), intent(out)           :: fstring
        integer, intent(in)                     :: plen

        integer                                 :: i
        character(len=1, kind=c_char), target   :: cstring(plen)

        call readbuffer_csocket(psockfd, c_loc(cstring(1)), plen)
        fstring=""   
        do i = 1,plen
            fstring(i:i) = cstring(i)
        enddo
    end subroutine

    subroutine readbuffer_dv(psockfd, fdata, plen)
        use iso_c_binding  
        integer, intent(in)                     :: psockfd, plen
        real(kind=8), intent(out), target       :: fdata(plen)
    
        call readbuffer_csocket(psockfd, c_loc(fdata(1)), 8*plen)
    end subroutine

    subroutine readbuffer_spv(psockfd, fdata, plen)
        use iso_c_binding
        integer, intent(in)                     :: psockfd, plen
        real(kind=4), intent(out), target       :: fdata(plen)

        call readbuffer_csocket(psockfd, c_loc(fdata(1)), 4*plen)
    end subroutine
end module