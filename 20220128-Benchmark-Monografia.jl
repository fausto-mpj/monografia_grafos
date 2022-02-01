#-----------------------------------------------------------------------------#
# @1. Pacotes
#-----------------------------------------------------------------------------#

##
using BenchmarkTools

##

#-----------------------------------------------------------------------------#
# @2. Variáveis
#-----------------------------------------------------------------------------#

##
include("20220128-Monografia.jl")

##

#-----------------------------------------------------------------------------#
# @3. Benchmark
#-----------------------------------------------------------------------------#

##
bm_reliability = @benchmarkable Sᵣ(grafo) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_reliability)
bm_reliability_result = run(bm_reliability)

##
bm_reliability2 = @benchmarkable Sᵣ(grafo, unique(get_strategic_edges2(grafo).Edge)) samples = 1000 seconds = 180 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_reliability2)
bm_reliability_result2 = run(bm_reliability2)

##
bm_alpha = @benchmarkable α_strategy(grafo, get_possible_edges(grafo)) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_alpha)
bm_alpha_result = run(bm_alpha)

##
bm_phi = @benchmarkable ϕ_strategy(grafo, get_possible_edges(grafo)) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_phi)
bm_phi_result = run(bm_phi)

##
bm_beta = @benchmarkable β_strategy(grafo) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_beta)
bm_beta_result = run(bm_beta)

##
bm_gamma = @benchmarkable γ_strategy(grafo) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_gamma)
bm_gamma_result = run(bm_gamma)

##
bm_delta = @benchmarkable δ_strategy(grafo) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_delta)
bm_delta_result = run(bm_delta)

##
bm_r = @benchmarkable r_strategy(get_possible_edges(grafo)) samples = 1000 seconds = 120 setup = (grafo = erdos_renyi(15, 0.5))
tune!(bm_r)
bm_r_result = run(bm_r)
