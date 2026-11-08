! Minimal kryos-style host: start Julia once in-process (libjulia), load KryosEmbed,
! fetch C function pointers, then call the plastic solver N times on Fortran arrays
! (no copies, no Julia objects crossing the boundary), optionally one gradient call.
!
! usage: embed_main nx ny ncalls mode(0=flat,1=full) do_grad(0/1/2=krylov) [sysimage.so]
program embed_main
    use iso_c_binding
    implicit none
    interface
        subroutine jl_init() bind(C, name="jl_init")
        end subroutine
        subroutine jl_init_with_image(bindir, image) bind(C, name="jl_init_with_image")
            import :: c_char
            character(kind=c_char), intent(in) :: bindir(*), image(*)
        end subroutine
        function jl_eval_string(s) bind(C, name="jl_eval_string") result(r)
            import :: c_char, c_ptr
            character(kind=c_char), intent(in) :: s(*)
            type(c_ptr) :: r
        end function
        function jl_exception_occurred() bind(C, name="jl_exception_occurred") result(r)
            import :: c_ptr
            type(c_ptr) :: r
        end function
        function jl_unbox_voidpointer(v) bind(C, name="jl_unbox_voidpointer") result(p)
            import :: c_ptr, c_funptr
            type(c_ptr), value :: v
            type(c_funptr) :: p
        end function
        function jl_string_ptr(v) bind(C, name="jl_string_ptr") result(p)
            import :: c_ptr
            type(c_ptr), value :: v
            type(c_ptr) :: p
        end function
        subroutine jl_atexit_hook(status) bind(C, name="jl_atexit_hook")
            import :: c_int
            integer(c_int), value :: status
        end subroutine
        function c_strlen(s) bind(C, name="strlen") result(n)
            import :: c_ptr, c_size_t
            type(c_ptr), value :: s
            integer(c_size_t) :: n
        end function
    end interface
    abstract interface
        function solve_i(zb, tb, m, zs, h, nx, ny, dx, dy, mode) bind(C) result(rc)
            import :: c_double, c_bool, c_int
            real(c_double), intent(in)  :: zb(*), tb(*)
            logical(c_bool), intent(in) :: m(*)
            real(c_double), intent(out) :: zs(*), h(*)
            integer(c_int), value :: nx, ny, mode
            real(c_double), value :: dx, dy
            integer(c_int) :: rc
        end function
        function grad_i(zb, tb, m, ht, g, loss, nx, ny, dx, dy, mode, solver) bind(C) result(rc)
            import :: c_double, c_bool, c_int
            real(c_double), intent(in)  :: zb(*), tb(*), ht(*)
            logical(c_bool), intent(in) :: m(*)
            real(c_double), intent(out) :: g(*), loss
            integer(c_int), value :: nx, ny, mode, solver
            real(c_double), value :: dx, dy
            integer(c_int) :: rc
        end function
    end interface

    integer :: nx, ny, ncalls, mode, dograd, k, rc, i, j
    character(len=512) :: arg, sysimg, bindir
    real(c_double), allocatable :: zb(:,:), tb(:,:), zs(:,:), h(:,:), g(:,:)
    logical(c_bool), allocatable :: m(:,:)
    real(c_double) :: dx, x, y, r2, loss, t0, t1, hmean, pi
    real(c_double), allocatable :: ts(:)
    type(c_ptr) :: v
    procedure(solve_i), pointer :: jsolve
    procedure(grad_i),  pointer :: jgrad
    integer(8) :: c0, c1, cr

    call system_clock(count_rate=cr)
    call get_command_argument(1, arg); read(arg, *) nx
    call get_command_argument(2, arg); read(arg, *) ny
    call get_command_argument(3, arg); read(arg, *) ncalls
    call get_command_argument(4, arg); read(arg, *) mode
    call get_command_argument(5, arg); read(arg, *) dograd
    call get_command_argument(6, sysimg)
    call get_environment_variable("JULIA_BINDIR", bindir)
    pi = acos(-1.0_c_double); dx = 32.0e3_c_double

    allocate(zb(nx,ny), tb(nx,ny), zs(nx,ny), h(nx,ny), g(nx,ny), m(nx,ny), ts(ncalls))
    do j = 1, ny
        do i = 1, nx
            x = (i - 0.5_c_double) / nx; y = (j - 0.5_c_double) / ny
            r2 = ((x - 0.5_c_double)/0.35_c_double)**2 + ((y - 0.5_c_double)/0.38_c_double)**2
            m(i,j) = r2 < 1.0_c_double
            zb(i,j) = 200.0_c_double*sin(6*pi*x)*cos(4*pi*y) - 300.0_c_double*(x - 0.5_c_double) + 100.0_c_double
            tb(i,j) = 1.0e5_c_double*(1.0_c_double + 0.5_c_double*sin(2*pi*x)*cos(2*pi*y))
        end do
    end do

    ! --- start Julia (once) ---
    call system_clock(c0)
    if (len_trim(sysimg) > 0) then
        call jl_init_with_image(trim(bindir)//c_null_char, trim(sysimg)//c_null_char)
    else
        call jl_init()
    end if
    call system_clock(c1); write(*,'(a,f8.3,a)') 'jl_init            : ', real(c1-c0)/cr, ' s'

    call system_clock(c0)
    v = jl_eval_string("using KryosEmbed"//c_null_char); call chk('using KryosEmbed')
    call system_clock(c1); write(*,'(a,f8.3,a)') 'using KryosEmbed   : ', real(c1-c0)/cr, ' s'
    v = jl_eval_string("KryosEmbed.entry_solve()"//c_null_char); call chk('entry_solve')
    call c_f_procpointer(jl_unbox_voidpointer(v), jsolve)
    v = jl_eval_string("KryosEmbed.entry_grad()"//c_null_char); call chk('entry_grad')
    call c_f_procpointer(jl_unbox_voidpointer(v), jgrad)

    ! --- solve N times ---
    do k = 1, ncalls
        call system_clock(c0)
        rc = jsolve(zb, tb, m, zs, h, int(nx,c_int), int(ny,c_int), dx, dx, int(mode,c_int))
        call system_clock(c1); ts(k) = real(c1-c0, c_double)/cr
        if (rc /= 0) call fail('solve')
    end do
    hmean = sum(h, mask=logical(m)) / count(logical(m))
    write(*,'(a,f8.4,a)') 'solve first call   : ', ts(1), ' s'
    if (ncalls > 1) write(*,'(a,f8.4,a,f8.4,a)') 'solve steady (min/med): ', minval(ts(2:)), ' / ', median(ts(2:)), ' s'
    write(*,'(a,f10.3,a,f12.6)') 'Hmax = ', maxval(h), '  Hmean = ', hmean

    ! --- one gradient call (target = 0.9 H) ---
    if (dograd > 0) then
        h = 0.9_c_double * h
        call system_clock(c0)
        rc = jgrad(zb, tb, m, h, g, loss, int(nx,c_int), int(ny,c_int), dx, dx, int(mode,c_int), int(dograd-1,c_int))
        call system_clock(c1)
        if (rc /= 0) call fail('grad')
        write(*,'(a,f8.3,a,es12.4,a,es12.4)') 'grad 1st call      : ', real(c1-c0)/cr, ' s  loss=', loss, ' sum|g|=', sum(abs(g))
        call system_clock(c0)
        rc = jgrad(zb, tb, m, h, g, loss, int(nx,c_int), int(ny,c_int), dx, dx, int(mode,c_int), int(dograd-1,c_int))
        call system_clock(c1)
        write(*,'(a,f8.3,a,es12.4,a,es12.4)') 'grad 2nd call      : ', real(c1-c0)/cr, ' s  loss=', loss, ' sum|g|=', sum(abs(g))
    end if
    call jl_atexit_hook(0_c_int)

contains
    subroutine chk(what)
        character(*), intent(in) :: what
        type(c_ptr) :: e
        e = jl_exception_occurred()
        if (c_associated(e)) then
            write(*,*) 'Julia exception in ', what; stop 2
        end if
    end subroutine
    subroutine fail(what)
        character(*), intent(in) :: what
        type(c_ptr) :: s
        character(kind=c_char), pointer :: cs(:)
        integer :: n, q
        character(len=1024) :: msg
        s = jl_eval_string("KryosEmbed.lasterr()"//c_null_char)
        n = int(min(c_strlen(jl_string_ptr(s)), 1023_c_size_t))
        call c_f_pointer(jl_string_ptr(s), cs, [n])
        msg = ''
        do q = 1, n; msg(q:q) = cs(q); end do
        write(*,*) what, ' failed: ', trim(msg); stop 3
    end subroutine
    function median(a) result(r)
        real(c_double), intent(in) :: a(:)
        real(c_double) :: r, b(size(a)), t
        integer :: p, q, n
        b = a; n = size(b)
        do p = 2, n
            t = b(p); q = p - 1
            do while (q >= 1)
                if (b(q) <= t) exit
                b(q+1) = b(q); q = q - 1
            end do
            b(q+1) = t
        end do
        r = b((n+1)/2)
    end function
end program
