"""
    KryosEmbed

C-callable entry points for driving PlasticIceSheet from a Fortran host (kryos).

The host obtains plain C function pointers once (`entry_solve()`, `entry_grad()`), then
calls them like any C routine: arrays are Fortran-owned memory wrapped without copying
(`unsafe_wrap`), exceptions are caught here and reported as a nonzero return plus
`lasterr()`. No Julia object crosses the boundary, so the host needs no GC rooting.
"""
module KryosEmbed

using PlasticIceSheet
using ImplicitDifferentiation, ADTypes, ForwardDiff   # activates the reverse-mode extension
using Zygote

export synthetic_case, entry_solve, entry_grad, lasterr

const LASTERR = Ref("")
lasterr() = LASTERR[]

mode_sym(m) = m == 0 ? :flat : :full

# Deterministic synthetic case (the Fortran test program evaluates the same formulas).
function synthetic_case(nx::Int, ny::Int)
    zb = Matrix{Float64}(undef, nx, ny); tb = similar(zb); mask = Matrix{Bool}(undef, nx, ny)
    for j in 1:ny, i in 1:nx
        x = (i - 0.5) / nx; y = (j - 0.5) / ny
        r2 = ((x - 0.5) / 0.35)^2 + ((y - 0.5) / 0.38)^2
        mask[i, j] = r2 < 1
        zb[i, j] = 200 * sin(6π * x) * cos(4π * y) - 300 * (x - 0.5) + 100
        tb[i, j] = 1e5 * (1 + 0.5 * sin(2π * x) * cos(2π * y))
    end
    return zb, tb, mask
end

wrap2(p::Ptr{T}, nx, ny) where {T} = unsafe_wrap(Array, p, (Int(nx), Int(ny)))

# forward: (z_b, τ_b, mask) -> (z_s, H); writes into host buffers
function _solve_c(zb_p::Ptr{Cdouble}, tb_p::Ptr{Cdouble}, m_p::Ptr{Bool},
                  zs_p::Ptr{Cdouble}, H_p::Ptr{Cdouble},
                  nx::Cint, ny::Cint, dx::Cdouble, dy::Cdouble, mode::Cint)::Cint
    try
        zb = wrap2(zb_p, nx, ny); tb = wrap2(tb_p, nx, ny); m = wrap2(m_p, nx, ny)
        zs, H = PlasticIceSheet.solve(zb, tb, m; dx, dy, mode = mode_sym(mode))
        copyto!(wrap2(zs_p, nx, ny), zs); copyto!(wrap2(H_p, nx, ny), H)
        return Cint(0)
    catch e
        LASTERR[] = sprint(showerror, e)
        return Cint(1)
    end
end

# gradient of L = mean_mask (H - H_target)^2 w.r.t. τ_b (reverse mode, implicit diff).
# solver: 0 = dense direct (small grids only), 1 = matrix-free Krylov.
function _grad_c(zb_p::Ptr{Cdouble}, tb_p::Ptr{Cdouble}, m_p::Ptr{Bool}, Ht_p::Ptr{Cdouble},
                 g_p::Ptr{Cdouble}, loss_p::Ptr{Cdouble},
                 nx::Cint, ny::Cint, dx::Cdouble, dy::Cdouble, mode::Cint, solver::Cint)::Cint
    try
        zb = wrap2(zb_p, nx, ny); tb = copy(wrap2(tb_p, nx, ny)); m = wrap2(m_p, nx, ny)
        Ht = wrap2(Ht_p, nx, ny)
        n = count(m)
        kw = solver == 0 ? (;) :
             (; representation = OperatorRepresentation(),
                linear_solver = IterativeLinearSolver(; maxiter = 2000, atol = 1e-8, rtol = 1e-8))
        loss(t) = sum(abs2, (PlasticIceSheet.differentiable_thickness(t, zb, m; dx, dy,
                      mode = mode_sym(mode), kw...) .- Ht) .* m) / n
        l, back = Zygote.withgradient(loss, tb)
        unsafe_store!(loss_p, l)
        copyto!(wrap2(g_p, nx, ny), back[1])
        return Cint(0)
    catch e
        LASTERR[] = sprint(showerror, e)
        return Cint(1)
    end
end

entry_solve() = @cfunction(_solve_c, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Bool}, Ptr{Cdouble},
    Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Cint))
entry_grad() = @cfunction(_grad_c, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Bool}, Ptr{Cdouble},
    Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cdouble, Cdouble, Cint, Cint))

# Run once at build time so the sysimage / pkgimage holds the compiled methods.
function warmup()
    zb, tb, m = synthetic_case(24, 30)
    H = zeros(24, 30); zs = similar(H); g = similar(H); l = Ref(0.0)
    GC.@preserve zb tb m H zs g l begin
        for mode in (0, 1)
            _solve_c(pointer(zb), pointer(tb), pointer(m), pointer(zs), pointer(H),
                     Cint(24), Cint(30), 32e3, 32e3, Cint(mode))
            _grad_c(pointer(zb), pointer(tb), pointer(m), pointer(H), pointer(g), Base.unsafe_convert(Ptr{Cdouble}, l),
                    Cint(24), Cint(30), 32e3, 32e3, Cint(mode), Cint(0))
        end
    end
    return nothing
end

end # module
