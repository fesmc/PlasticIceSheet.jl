using PlasticIceSheet
const P = PlasticIceSheet
M = Matrix{Float64}; B = Matrix{Bool}
const p = PlasticParams()
rt(f, T) = println(nameof(f), " -> ", Base.return_types(f, T))
rt(P._bc_flat, (Float64, typeof(p)))
println("bc bcast: ", Base.return_types(x -> P._bc_flat.(x, Ref(p)), (M,)))
rt(P.sweep_eikonal, (M, B, M, Float64, Float64))
rt(P._driving_rhs, (M, M, M, Float64, Float64, typeof(p), Symbol))
println("_solve_w: ", Base.return_types((a,b,c,d,e) -> P._solve_w(a,b,c,d,e,e,p,:flat; max_sweeps=200,tol=1e-6,n_outer=100,outer_tol=1e-3), (M,M,B,M,Float64)))
println("_solve_w sym: ", Base.return_types((a,b,c,d,e,s) -> P._solve_w(a,b,c,d,e,e,p,s; max_sweeps=200,tol=1e-6,n_outer=100,outer_tol=1e-3), (M,M,B,M,Float64,Symbol)))
rt(P.flotation_thickness, (Float64, typeof(p)))
println("full solve flat: ", Base.return_types((a,b,c) -> P.solve(a,b,c; dx=1.0, dy=1.0, mode=:flat), (M,M,B)))
println("full solve full: ", Base.return_types((a,b,c) -> P.solve(a,b,c; dx=1.0, dy=1.0, mode=:full), (M,M,B)))
println("solve sym: ", Base.return_types((a,b,c,s) -> P.solve(a,b,c; dx=1.0, dy=1.0, mode=s), (M,M,B,Symbol)))
