! Route b host: link the juliac --trim shared library (libplastic.so) and call kp_solve directly.
! usage: routeb_main nx ny ncalls mode
program routeb_main
    use iso_c_binding
    implicit none
    interface
        function kp_solve(zb, tb, m, zs, h, nx, ny, dx, dy, mode) bind(C, name="kp_solve") result(rc)
            import :: c_double, c_bool, c_int
            real(c_double), intent(in)  :: zb(*), tb(*)
            logical(c_bool), intent(in) :: m(*)
            real(c_double), intent(out) :: zs(*), h(*)
            integer(c_int), value :: nx, ny, mode
            real(c_double), value :: dx, dy
            integer(c_int) :: rc
        end function
    end interface
    integer :: nx, ny, nc, mode, k, rc, i, j
    character(len=64) :: a
    real(c_double), allocatable :: zb(:,:), tb(:,:), zs(:,:), h(:,:), ts(:)
    logical(c_bool), allocatable :: m(:,:)
    real(c_double) :: pi, x, y, r2, dx
    integer(8) :: c0, c1, cr
    call system_clock(count_rate=cr)
    call get_command_argument(1,a); read(a,*) nx
    call get_command_argument(2,a); read(a,*) ny
    call get_command_argument(3,a); read(a,*) nc
    call get_command_argument(4,a); read(a,*) mode
    pi = acos(-1.0_c_double); dx = 32.0e3_c_double
    allocate(zb(nx,ny), tb(nx,ny), zs(nx,ny), h(nx,ny), m(nx,ny), ts(nc))
    do j = 1, ny; do i = 1, nx
        x = (i-0.5_c_double)/nx; y = (j-0.5_c_double)/ny
        r2 = ((x-0.5_c_double)/0.35_c_double)**2 + ((y-0.5_c_double)/0.38_c_double)**2
        m(i,j) = r2 < 1.0_c_double
        zb(i,j) = 200.0_c_double*sin(6*pi*x)*cos(4*pi*y) - 300.0_c_double*(x-0.5_c_double) + 100.0_c_double
        tb(i,j) = 1.0e5_c_double*(1.0_c_double + 0.5_c_double*sin(2*pi*x)*cos(2*pi*y))
    end do; end do
    do k = 1, nc
        call system_clock(c0)
        rc = kp_solve(zb, tb, m, zs, h, int(nx,c_int), int(ny,c_int), dx, dx, int(mode,c_int))
        call system_clock(c1); ts(k) = real(c1-c0, c_double)/cr
    end do
    write(*,'(a,f8.4,a)') 'route b first call : ', ts(1), ' s'
    if (nc > 1) write(*,'(a,f8.4,a,f8.4,a)') 'route b steady min/max : ', minval(ts(2:)), ' / ', maxval(ts(2:)), ' s'
    write(*,'(a,f10.3,a,f12.6)') 'Hmax = ', maxval(h), '  Hmean = ', sum(h, mask=logical(m))/count(logical(m))
end program
