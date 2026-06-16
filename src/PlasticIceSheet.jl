"""
    PlasticIceSheet

Perfectly-plastic, steady-state ice-sheet reconstruction by solving the plastic surface
equation `|∇z_s| = τ_b / (ρ_i g (z_s − z_b))` as a static Hamilton–Jacobi (eikonal) problem
on a grid with fast sweeping. A simplification and modernization of ICESHEET 2.0
(Gowan et al. 2016, <https://github.com/evangowan/icesheet>); see `docs/PLAN.md`.

Inputs: a grounded-ice `mask`, bed elevation `z_b`, and basal shear stress `τ_b`.
Outputs: ice surface elevation `z_s` and thickness `H`. AD-friendly w.r.t. `τ_b`.
"""
module PlasticIceSheet

include("constants.jl")
include("godunov.jl")
include("solve.jl")
include("rheology.jl")
include("velocity.jl")
include("balance.jl")
include("reconstruct.jl")
include("diagnostics.jl")
include("io.jl")

export PlasticParams, solve
export ice_volume, ice_area, surface_misfit
export flotation_thickness, godunov_eikonal, sweep_eikonal, godunov_residual
export differentiable_thickness
export load_plastic_inputs, save_reconstruction
export GlenRheology, deformational_velocity
export SlidingLaw, LinearSliding, WeertmanSliding, RegularizedCoulomb, basal_velocity
export flow_direction, diva_velocity, smb_from_velocity
export balance_flux, balance_velocity, implied_basal_velocity
export diva_reconstruct

# Friendly fallback: the real `differentiable_thickness` method ships in the extension
# activated by `ImplicitDifferentiation` (with an AD backend). The extension method is
# more specific, so it takes precedence whenever the trigger packages are loaded.
function differentiable_thickness(args...; kwargs...)
    error("`differentiable_thickness` (reverse-mode τ_b inversion) needs the implicit-" *
          "differentiation extension. Load its triggers first:\n" *
          "    using ImplicitDifferentiation, ADTypes, ForwardDiff")
end

end # module
