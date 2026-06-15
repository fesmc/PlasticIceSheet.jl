# NetCDF I/O convenience. Real methods live in the extension activated by `NCDatasets`;
# the base package carries only stubs so it stays dependency-light.

"""
    load_plastic_inputs(path; bed="bed", tau="tau", mask="mask", x="x", y="y")

Read reconstruction inputs from a NetCDF file. Returns a NamedTuple
`(; bed, τ, mask, dx, dy, x, y)` with `bed`, `τ` as matrices, `mask` a `BitMatrix`
(nonzero ⇒ grounded ice), and `dx`, `dy` inferred from the `x`/`y` coordinate vectors.

Requires `using NCDatasets`.
"""
function load_plastic_inputs end

"""
    save_reconstruction(path, E, H; x=nothing, y=nothing, attrib=Dict())

Write a reconstruction (surface `E`, thickness `H`) to a NetCDF file, with optional
`x`/`y` coordinate vectors. Requires `using NCDatasets`.
"""
function save_reconstruction end
