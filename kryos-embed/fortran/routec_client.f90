! Route c client: per call, write inputs to /dev/shm, send a request on a FIFO, block on the
! reply FIFO, read outputs. Times the full round trip. usage: routec_client nx ny ncalls mode cmdfifo respfifo
program routec_client
    use iso_c_binding, only: c_double
    implicit none
    integer :: nx, ny, nc, mode, k, i, j, u
    character(len=512) :: a, cmdf, respf, resp
    real(c_double), allocatable :: zb(:,:), tb(:,:), zs(:,:), h(:,:), ts(:)
    integer(1), allocatable :: m(:,:)
    real(c_double) :: pi, x, y, r2, dx
    integer(8) :: c0, c1, cr
    call system_clock(count_rate=cr)
    call get_command_argument(1,a); read(a,*) nx
    call get_command_argument(2,a); read(a,*) ny
    call get_command_argument(3,a); read(a,*) nc
    call get_command_argument(4,a); read(a,*) mode
    call get_command_argument(5,cmdf); call get_command_argument(6,respf)
    pi = acos(-1.0_c_double); dx = 32.0e3_c_double
    allocate(zb(nx,ny), tb(nx,ny), zs(nx,ny), h(nx,ny), m(nx,ny), ts(nc))
    do j = 1, ny; do i = 1, nx
        x = (i-0.5_c_double)/nx; y = (j-0.5_c_double)/ny
        r2 = ((x-0.5_c_double)/0.35_c_double)**2 + ((y-0.5_c_double)/0.38_c_double)**2
        m(i,j) = merge(1_1, 0_1, r2 < 1.0_c_double)
        zb(i,j) = 200.0_c_double*sin(6*pi*x)*cos(4*pi*y) - 300.0_c_double*(x-0.5_c_double) + 100.0_c_double
        tb(i,j) = 1.0e5_c_double*(1.0_c_double + 0.5_c_double*sin(2*pi*x)*cos(2*pi*y))
    end do; end do
    open(newunit=u, file=trim(respf), action='read', form='formatted'); read(u,'(a)') resp; close(u)
    write(*,*) 'server: ', trim(resp)
    do k = 1, nc
        call system_clock(c0)
        open(newunit=u, file='/dev/shm/kp_in.bin', access='stream', form='unformatted', status='replace')
        write(u) zb, tb, m; close(u)
        open(newunit=u, file=trim(cmdf), action='write', form='formatted')
        write(u,'(a,2(1x,i0),2(1x,es12.5),1x,i0,a)') 'solve', nx, ny, dx, dx, mode, ' /dev/shm/kp_in.bin /dev/shm/kp_out.bin'
        close(u)
        open(newunit=u, file=trim(respf), action='read', form='formatted'); read(u,'(a)') resp; close(u)
        if (trim(resp) /= 'ok') then; write(*,*) trim(resp); stop 1; end if
        open(newunit=u, file='/dev/shm/kp_out.bin', access='stream', form='unformatted', status='old')
        read(u) zs, h; close(u)
        call system_clock(c1); ts(k) = real(c1-c0, c_double)/cr
    end do
    write(*,'(a,f8.4,a)') 'route c first request : ', ts(1), ' s'
    if (nc > 1) write(*,'(a,f8.4,a,f8.4,a)') 'route c steady min/max : ', minval(ts(2:)), ' / ', maxval(ts(2:)), ' s'
    write(*,'(a,f10.3,a,f12.6)') 'Hmax = ', maxval(h), '  Hmean = ', sum(h, mask=m/=0)/count(m/=0)
    open(newunit=u, file=trim(cmdf), action='write', form='formatted'); write(u,'(a)') 'quit'; close(u)
end program
