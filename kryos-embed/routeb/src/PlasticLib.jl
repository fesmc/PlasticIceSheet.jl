# Route b: forward solver as a trimmed shared library (juliac, Julia >= 1.12).
# Build:  julia --project=kryos-embed/routeb <julia>/share/julia/juliac/juliac.jl \
#           --output-lib libplastic --experimental --trim=safe --compile-ccallable routeb/src/PlasticLib.jl
module PlasticLib
using PlasticIceSheet

Base.@ccallable function kp_solve(zb_p::Ptr{Cdouble}, tb_p::Ptr{Cdouble}, m_p::Ptr{Bool},
        zs_p::Ptr{Cdouble}, H_p::Ptr{Cdouble}, nx::Cint, ny::Cint,
        dx::Cdouble, dy::Cdouble, mode::Cint)::Cint
    0 <= mode <= 2 || return Cint(1)   # 0 = :flat, 1 = :full, 2 = :surface
    sz = (Int(nx), Int(ny))
    zb = unsafe_wrap(Array, zb_p, sz); tb = unsafe_wrap(Array, tb_p, sz)
    m = unsafe_wrap(Array, m_p, sz)
    zs, H = PlasticIceSheet.solve(zb, tb, m; dx = dx, dy = dy, mode = (:flat, :full, :surface)[mode + 1])
    copyto!(unsafe_wrap(Array, zs_p, sz), zs); copyto!(unsafe_wrap(Array, H_p, sz), H)
    return Cint(0)
end

end
