module PlasticIceSheetImplicitDiffExt

# Reverse-mode-differentiable thickness solve w.r.t. a full per-cell basal shear stress
# field `τ_b`, via implicit differentiation of the converged Godunov fixed point.
#
# The forward fast sweep is treated as a black box; only the optimality conditions
# `godunov_residual(w, τ_b) = 0` are differentiated (with ForwardDiff), and the implicit
# function theorem yields ∂w/∂τ_b from a single linear solve — independent of the number
# of sweeps / outer iterations, and scalable to a full `τ_b` field. The user's outer AD
# (e.g. Zygote) gets the gradient through ImplicitDifferentiation's ChainRules rule.

using PlasticIceSheet
using PlasticIceSheet: _solve_w, _bc_flat, godunov_residual, PlasticParams
using ImplicitDifferentiation: ImplicitFunction, MatrixRepresentation, DirectLinearSolver
using ADTypes: AutoForwardDiff
import ForwardDiff  # provides the backend used to differentiate the conditions

"""
    differentiable_thickness(τ_b, z_b, mask; dx, dy, mode=:full, params=PlasticParams(),
                             max_sweeps, tol, n_outer, outer_tol,
                             representation=MatrixRepresentation(),
                             linear_solver=DirectLinearSolver())

Reverse-mode-differentiable ice thickness `H` as a function of a full per-cell basal
shear stress field `τ_b` (with bed elevation `z_b`), via implicit differentiation of the
converged Godunov fixed point. Use as the inner solve of a basal-shear-stress inversion;
differentiate a scalar objective of `H` with any reverse-mode AD (e.g. Zygote).

The default linear solver forms the (sparse-ish) Jacobian explicitly and solves it
directly — robust and exact for the modest grids this first-guess tool targets. For very
large grids pass `representation=OperatorRepresentation()` and
`linear_solver=IterativeLinearSolver(; maxiter=...)` (matrix-free, may need tuning, as
the eikonal adjoint's characteristics span the domain).
"""
function PlasticIceSheet.differentiable_thickness(
        τ_b::AbstractMatrix, z_b::AbstractMatrix, mask::AbstractMatrix{Bool};
        dx::Real, dy::Real, mode::Symbol = :full,
        params::PlasticParams = PlasticParams(),
        max_sweeps::Int = 200, tol = 1e-6, n_outer::Int = 200, outer_tol = 1e-5,
        relax = 0.5,
        representation = MatrixRepresentation(),
        linear_solver = DirectLinearSolver())

    size(z_b) == size(mask) == size(τ_b) ||
        throw(DimensionMismatch("τ_b, z_b and mask must share the same shape"))

    bc = _bc_flat.(z_b, Ref(params))
    shape = size(z_b)
    # Constant (non-differentiated) context carried alongside the differentiated `τ_b`.
    # `:full` needs a tightly-converged forward solve (small residual) for the implicit
    # gradient to be exact, hence the firmer n_outer / outer_tol defaults here.
    ctx = (; z_b, mask, bc, dx, dy, params, mode, shape,
           max_sweeps, tol, n_outer, outer_tol, relax)

    # The implicit output `y` and the residual are kept as flat vectors so the Jacobian
    # operator ∂c/∂y is square (vector → vector) for both direct and iterative solvers.

    # solver(x, args...) -> (y, z): forward solve for w = H² (byproduct z unused).
    solver = function (x, c)
        w = _solve_w(x, c.z_b, c.mask, c.bc, c.dx, c.dy, c.params, c.mode;
                     max_sweeps = c.max_sweeps, tol = c.tol,
                     n_outer = c.n_outer, outer_tol = c.outer_tol, relax = c.relax)
        return vec(w), nothing
    end

    # conditions(x, y, z, args...) -> c: the residual that vanishes at the solution.
    conditions = function (x, y, _z, c)
        r = godunov_residual(reshape(y, c.shape), x, c.z_b, c.mask, c.bc,
                             c.dx, c.dy, c.params, c.mode)
        return vec(r)
    end

    implicit = ImplicitFunction(solver, conditions; representation, linear_solver,
                                backends = (; x = AutoForwardDiff(), y = AutoForwardDiff()))

    wvec, _ = implicit(τ_b, ctx)
    w = reshape(wvec, shape)
    # H = √w on ice, 0 elsewhere — written as a mask-multiply with a floored √ so the
    # reverse pass never evaluates sqrt'(0)=Inf (which would give Inf·0 = NaN under
    # Zygote). The floor (H_min²) is far below any real ice thickness.
    floorw = params.H_min^2
    return @. sqrt(max(w, floorw)) * mask
end

end # module
