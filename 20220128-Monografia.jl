#-----------------------------------------------------------------------------#
# @1. Pacotes
#-----------------------------------------------------------------------------#

##
using Chain
using Combinatorics
using DataFrames
using Distributions
using Graphs
using IntervalArithmetic
using LinearAlgebra
using Polynomials
using ProgressMeter
using Random
using SimpleWeightedGraphs
using SortingAlgorithms
using Statistics
using TypedPolynomials

##

#-----------------------------------------------------------------------------#
# @2. Variáveis
#-----------------------------------------------------------------------------#

##
path = pwd()

##

#-----------------------------------------------------------------------------#
# @3. Funções auxiliares
#-----------------------------------------------------------------------------#

##
"""
    get_possible_edges(G::SimpleGraph{Int64})
---

# Descrição

Retorna as arestas possíveis de serem adicionadas em um grafo `G`.

A operação retorna a lista de arestas de `G - Kₙ`, onde `Kₙ` é o grafo completo de mesma ordem que `G`.

# Fluxo dos Dados

- SimpleGraph{Int64} -> get_possible_edges() -> SimpleEdgeIter{SimpleGraph{Int64}}

---

"""
function get_possible_edges(G::SimpleGraph{Int64})
    return (edges(difference(complete_graph(nv(G)), G)))
end

##
"""
    second_smaller(vector::Vector{Float64})
---

# Descrição

Retorna todas as posições em um vetor de entrada em que a componente tem valor igual ao segundo menor valor dentre todas as componentes do vetor.

# Fluxo dos Dados

- Vector{Float64} -> second_smaller() -> Vector{Int64}

---

"""
function second_smaller(vector::Vector{Float64})
    aux = sort(vector)
    second = aux[2]
    return (findall(x -> x == second, vector))
end

##

#-----------------------------------------------------------------------------#
# @4. Estratégias
#-----------------------------------------------------------------------------#

##
"""
    α_strategy(G::SimpleGraph{Int64}, arestas::SimpleEdgeIter{SimpleGraph{Int64}})
---

# Descrição

Seleciona inserções de arestas em `G` pela estratégia do maior incremento da conectividade algébrica.

A função calculará os autovalores da matriz Laplaciana do grafo `G` e de todos os supergrafos resultantes pela adição de uma aresta dentre as presentes em `arestas`. Para este cálculo, usa-se um arredondamento dos valores com base nos 12 dígitos após o marcador de decimal. Caso múltiplas arestas apresentem o mesmo maior incremento, então todas serão retornadas.

# Saídas

- `Strategy` (Symbol): Símbolo da estratégia. Sempre `:α`.

- `Edges` (Vector{Vector{Int64}}): Arestas indicadas pela heurística.

- `Info` (Vector{Float64}): Incremento na confiabilidade algébrica da inserção.

- `Extra` (Vector{Int64}): Sempre `0`.

# Fluxo dos Dados

- (SimpleGraph{Int64}, SimpleEdgeIter{SimpleGraph{Int64}}) -> α_strategy() -> NamedTuple{(:Strategy, :Edges, :Info, :Extra), Tuple{Symbol, Vector{Vector{Int64}}, Vector{Float64}, Vector{Int64}}

---

"""
function α_strategy(G::SimpleGraph{Int64}, arestas::Graphs.SimpleGraphs.SimpleEdgeIter{SimpleGraph{Int64}})
    vals = round.(
        eigvals(Matrix(laplacian_matrix(G))),
        digits = 12
    )
    algebraic_connectivity = vals[2]
    mat = fill(0.0, (length(arestas), 3))
    line = 1
    for aresta ∈ arestas
        graph_aux = copy(G)
        add_edge!(graph_aux, aresta)
        val_aux = round.(
            eigvals(Matrix(laplacian_matrix(graph_aux))),
            digits = 12
        )[2]
        mat[line, :] = [
            src(aresta),
            dst(aresta),
            val_aux - algebraic_connectivity
        ]
        line += 1
    end
    aux = findall(x -> x == maximum(mat[:, 3]), mat[:, 3])
    return (
        Strategy = :α,
        Edges = Vector.(
            eachrow(
                Int64.(
                    mat[aux, 1:2]
                )
            )
        ),
        Info = mat[aux, 3],
        Extra = repeat([0], length(mat[aux, 3]))
    )
end

##
"""
    ϕ_strategy(G::SimpleGraph{Int64}, arestas::SimpleEdgeIter{SimpleGraph{Int64}})
---

# Descrição

Seleciona inserções de arestas em `G` pela estratégia da maior distância de Fiedler.

A função calculará os autovetores da matriz Laplaciana do grafo `G`. Para este cálculo, usa-se um arredondamento dos valores com base nos 12 dígitos após o marcador de decimal. Caso múltiplos vetores de Fiedler apresentem diferentes pares de vértices com maior distância, então todos serão retornadas.

# Saídas

- `Strategy` (Symbol): Símbolo da estratégia. Sempre `:ϕ`.

- `Edges` (Vector{Vector{Int64}}): Arestas indicadas pela heurística.

- `Info` (Vector{Float64}): Distância de Fiedler da inserção.

- `Extra` (Vector{Int64}): Número do autovetor de Fiedler. Somente diferente de `1` quando ocorre multiplicidade geométrica maior que `1`.

# Fluxo dos Dados

- (SimpleGraph{Int64}, SimpleEdgeIter{SimpleGraph{Int64}}) -> ϕ_strategy() -> NamedTuple{(:Strategy, :Edges, :Info, :Extra), Tuple{Symbol, Vector{Vector{Int64}}, Vector{Float64}, Vector{Int64}}

---

"""
function ϕ_strategy(G::SimpleGraph{Int64}, arestas::Graphs.SimpleGraphs.SimpleEdgeIter{SimpleGraph{Int64}})
    vals, vecs = eigen(Matrix(laplacian_matrix(G)))
    fiedler = round.(
        vecs[:, second_smaller(round.(vals, digits = 12))],
        digits = 12
    )
    line = 1
    idx = 1
    mat = fill(0.0, (length(arestas) * size(fiedler, 2), 4))
    for vec ∈ eachcol(fiedler)
        for aresta ∈ arestas
            mat[line, :] = [
                idx,
                src(aresta),
                dst(aresta),
                abs(vec[src(aresta)] - vec[dst(aresta)])
            ]
            line += 1
        end
        idx += 1
    end
    aux = findall(x -> x == maximum(mat[:, 4]), mat[:, 4])
    aux2 = mean(mat[:, 4])
    return (
        Strategy = :ϕ,
        Edges = Vector.(
            eachrow(
                Int64.(
                    mat[aux, 2:3]
                )
            )
        ),
        Info = mat[aux, 4] .- aux2,
        Extra = Int64.(mat[aux, 1])
    )
end

##
"""
    δ_strategy(G::SimpleGraph{Int64})
---

# Descrição

Seleciona as inserções de arestas candidatas em `G` para a estratégia de maior distância geodésica da maior centralidade de grau.

Para o cálculo, são selecionados todos os vértices com a mair centralidade e então identificados os vértices mais distantes destes. Caso múltiplas inserções apresentem viabilidade, então todos serão retornadas.

A função retorna uma tupla contendo os campos `Edges` como um vetor dos índices dos vértices e `Centrality` com os valores da centralidade de grau de cada um dos vértices de `G`.

# Saídas

- `Strategy` (Symbol): Símbolo da estratégia. Sempre `:δ`.

- `Edges` (Vector{Vector{Int64}}): Arestas indicadas pela heurística.

- `Info` (Vector{Float64}): Menor centralidade de intermediação da inserção. 

- `Extra` (Vector{Int64}): Distância geodédica da inserção.

# Fluxo dos Dados

- SimpleGraph{Int64} -> δ_strategy() -> NamedTuple{(:Strategy, :Edges, :Info, :Extra), Tuple{Symbol, Vector{Vector{Int64}}, Vector{Float64}, Vector{Int64}}

---

"""
function δ_strategy(G::SimpleGraph{Int64})
    possible = [[src(i), dst(i)] for i in get_possible_edges(G)]
    centrality = degree_centrality(G)
    candidates = Vector{Int64}[]
    score = Float64[]
    distance = Int64[]
    pid = unique(vcat(possible...))
    for δ ∈ findall(x -> x == maximum(centrality[pid]), centrality[pid])
        aux = gdistances(G, δ; sort_alg = RadixSort)
        for Δ ∈ findall(x -> x == maximum(aux), aux)
            if sort([δ, Δ]) ∈ possible
                push!(candidates, sort([δ, Δ]))
                push!(score, centrality[δ])
                push!(distance, aux[Δ])
            end
        end
    end
    candidates = unique(candidates)
    return (
        Strategy = :δ,
        Edges = candidates,
        Info = score,
        Extra = distance
    )
end

##
"""
    β_strategy(G::SimpleGraph{Int64})
---

# Descrição

Seleciona as inserções de arestas candidatas em `G` para a estratégia de menor centralidade de intermediação.

Para o cálculo, são selecionados todos os vértices com a menor centralidade e então identificados os vértices não-adjacentes a estes com a menor centralidade. O par com as menores centralidades são retornados. Caso múltiplas inserções apresentem viabilidade, então todos serão retornadas.

A função retorna uma tupla contendo os campos `Edges` como um vetor dos índices dos vértices e `Centrality` com os valores da centralidade de grau de cada um dos vértices de `G`.

# Saídas

- `Strategy` (Symbol): Símbolo da estratégia. Sempre `:β`.

- `Edges` (Vector{Vector{Int64}}): Arestas indicadas pela heurística.

- `Info` (Vector{Float64}): Soma da centralidade de intermediação dos vértices da inserção.

- `Extra` (Vector{Int64}): Sempre `0`.

# Fluxo dos Dados

- SimpleGraph{Int64} -> β_strategy() -> NamedTuple{(:Strategy, :Edges, :Info, :Extra), Tuple{Symbol, Vector{Vector{Int64}}, Vector{Float64}, Vector{Int64}}

---

"""
function β_strategy(G::SimpleGraph{Int64})
    possible = [[src(i), dst(i)] for i in get_possible_edges(G)]
    centrality = betweenness_centrality(G)
    score = Float64[]
    aux = findall(x -> x == minimum(centrality), centrality)
    aux2 = second_smaller(centrality)
    if length(aux) == 1
        candidates = [sort([aux..., i]) for i ∈ aux2]
    else
        candidates = sort.(collect(combinations(aux, 2)))
    end
    candidates = intersect(candidates, possible)
    if isempty(candidates)
        candidates = rand(possible, 1)
    end
    info = sum.([centrality[i] for i ∈ candidates])
    return (
        Strategy = :β,
        Edges = candidates,
        Info = info,
        Extra = repeat([0], length(candidates))
    )
end

##
"""
    γ_strategy(G::SimpleGraph{Int64})
---

# Descrição

Seleciona as inserções de arestas candidatas em `G` para a estratégia de menor centralidade de grau.

Para o cálculo, são selecionados todos os vértices com a menor centralidade e então identificados os vértices não-adjacentes a estes com a menor centralidade. O par com as menores centralidades são retornados. Caso múltiplas inserções apresentem viabilidade, então todos serão retornadas.

A função retorna uma tupla contendo os campos `Edges` como um vetor dos índices dos vértices e `Centrality` com os valores da centralidade de grau de cada um dos vértices de `G`.

# Saídas

- `Strategy` (Symbol): Símbolo da estratégia. Sempre `:γ`.

- `Edges` (Vector{Vector{Int64}}): Arestas indicadas pela heurística.

- `Info` (Vector{Float64}): Soma da centralidade de grau dos vértices da inserção.

- `Extra` (Vector{Int64}): Sempre `0`.

# Fluxo dos Dados

- SimpleGraph{Int64} -> δ_strategy() -> NamedTuple{(:Strategy, :Edges, :Info, :Extra), Tuple{Symbol, Vector{Vector{Int64}}, Vector{Float64}, Vector{Int64}}

---

"""
function γ_strategy(G::SimpleGraph{Int64})
    possible = [[src(i), dst(i)] for i in get_possible_edges(G)]
    centrality = degree_centrality(G)
    aux = findall(x -> x == minimum(centrality), centrality)
    aux2 = second_smaller(centrality)
    if length(aux) == 1
        candidates = [sort([aux..., i]) for i ∈ aux2]
    else
        candidates = sort.(collect(combinations(aux, 2)))
    end
    candidates = intersect(candidates, possible)
    if isempty(candidates)
        candidates = rand(possible, 1)
    end
    info = sum.([centrality[i] for i ∈ candidates])
    return (
        Strategy = :γ,
        Edges = candidates,
        Info = info,
        Extra = repeat([0], length(candidates))
    )
end

##
"""
    r_strategy(arestas::SimpleEdgeIter{SimpleGraph{Int64}})
---

# Descrição

Seleciona uma inserção de aresta em `G` aleatoriamente.

A função retorna uma inserção qualquer dentre as inserções viáveis.


# Saídas

- `Strategy` (Symbol): Símbolo da estratégia. Sempre `:r`.

- `Edges` (Vector{Vector{Int64}}): Aresta indicada pela heurística.

- `Info` (Vector{Float64}): Sempre `0.0`.

- `Extra` (Vector{Int64}): Sempre `0`.

# Fluxo dos Dados

- SimpleEdgeIter{SimpleGraph{Int64}} -> r_strategy() -> NamedTuple{(:Strategy, :Edges, :Info, :Extra), Tuple{Symbol, Vector{Vector{Int64}}, Vector{Float64}, Vector{Int64}}

---

"""
function r_strategy(arestas::Graphs.SimpleGraphs.SimpleEdgeIter{SimpleGraph{Int64}})
    candidatos = Vector{Int64}[]
    aux = first(rand(collect(arestas), 1))
    push!(candidatos, [src(aux), dst(aux)])
    return (
        Strategy = :r,
        Edges = candidatos,
        Info = Float64[0.0],
        Extra = Int64[0]
    )
end

##

#-----------------------------------------------------------------------------#
# @5. Processamento de dados
#-----------------------------------------------------------------------------#

## 
"""
    Sᵣ(G::SimpleGraph)
---

# Descrição

Retorna um dicionário em que a chave é o número de vértices usados para induzir os subgrafos e o valor é o número destes subrafos que são conexos.

Por padrão, a função retornará somente o dicionário.

Caso a opção `insertions` receba valor `true`, então a função retornará um vetor em que os elementos são tuplas contendo a todas as aresta possíveis de serem adicionada ao grafo G e o dicionário associado ao supergrafo G com a inserção de cada aresta.

Caso a opção `parallel` receba valor `true`, então as combinações de vértices para produzir e testar os subgrafos induzidos de `G` utilizarão todos os núcleos disponíveis do CPU. Verificar com `Threads.nthread()` a disponibilidade dos núcleos.

# Argumentos

- `insertions` (Bool). Default: `false`.

- `parallel` (Bool). Default: `true`.

# Fluxo dos Dados

- SimpleGraph{Int64} -> Sᵣ(; insertions = false) -> Dict{Int64, Int64}

- SimpleGraph{Int64} -> Sᵣ(; insertions = true) -> Vector{Tuple{SimpleGraphs.SimpleEdge{Int64}, Dict{Int64, Int64}}}

"""
function Sᵣ(G::SimpleGraph{Int64}, insertions::Vector{Vector{Int64}} = empty([[0, 0]]))
    # Note que:
    # (1) Todo subgrafo induzido por um único vértice é conexo por definição, logo
    # para um grafo de ordem n há sempre n subgrafos induzidos por um único vértice
    # que são conexos.
    # (2) O subgrafo induzido por todos os vértices de um grafo G é o próprio G,
    # portanto se o supergrafo é conexo então há sempre 1 subgrafo induzido por
    # todos os vértices conexo.
    # Logo, o dicionário `num_connected` que associa cada número de vértices ao número
    # de subgrafos induzidos conexos necessariamente tem os pares chave-valor `1 => n`
    # e `n => 1`.
    n = nv(G)
    num_connected = Dict(zip(1:n, vcat(n, repeat([0], n - 2), 1)))
    if isempty(insertions)
        # Esse bloco pega todas as combinações de 2-a-2 até (n-1)-a-(n-1) de
        # vértices do grafo e, para cada um dos subconjuntos, produz o subgrafo
        # induzido. Cada um dos subgrafos induzidos é testado para ver se é ou
        # não conexo. Caso seja, adicionamos 1 ao valor associado à chave com
        # a ordem do subgrafo. O bloco retorna um dicionário com o número de
        # subrafos induzidos conexos associados a cada número de vértices.
        Threads.@threads for num_vtx ∈ 2:(n-1)
            aux_disconnect = Vector{Int64}[]
            for induced_vtx ∈ combinations(1:n, num_vtx)
                if is_connected(induced_subgraph(G, induced_vtx)[1])
                    num_connected[num_vtx] += 1
                end
            end
        end
        return (num_connected)
    else
        # Esse bloco pega todas as combinações 2-a-2 até (n-1)-a-(n-1) de
        # vértices do grafo e, para cada um dos subconjuntos, produz o subgrafo
        # induzido. Cada um dos subgrafos induzidos é testado para ver se é ou
        # não conexo. Caso seja, adicionamos 1 ao valor associado à chave com
        # a ordem do subgrafo; já caso contrário, adicionamos os vértices que
        # induziram o subgrafo a um vetor para sabermos quais são os subgrafos
        # que podem se tornar conexos por meio de uma inserção de aresta.
        disconnected_vtx = Dict(2:(n-1) .=> [repeat([[0, 0],], 1)])
        Threads.@threads for num_vtx ∈ 2:(n-1)
            aux_disconnect = Vector{Int64}[]
            for induced_vtx ∈ combinations(1:n, num_vtx)
                if is_connected(induced_subgraph(G, induced_vtx)[1])
                    num_connected[num_vtx] += 1
                else
                    push!(aux_disconnect, induced_vtx)
                end
            end
            disconnected_vtx[num_vtx] = aux_disconnect
        end
        disconnected_vtx = reduce(vcat, values(disconnected_vtx))
        num_connected_insertions = Dict(insertions .=> repeat([Dict(0 => 0)], length(insertions)))
        Threads.@threads for insertion ∈ insertions
            # Esse subbloco gera todas as arestas possíveis de serem adicionadas
            # ao grafo G. Para cada uma destas arestas, realizamos uma inserção
            # para gerar um supergrafo, copiamos o dicionário do grafo
            # G e verificamos se os conjuntos de vértices que induziram subgrafos
            # desconexos em G agoram induzem um subgrafo conexo no supergrafo.
            # Caso isso ocorra, adicionamos 1 ao valor associado à chave com a
            # ordem do subgrafo em cada ocorrência.
            aux = copy(G)
            add_edge!(aux, insertion...)
            connected_subgraph = copy(num_connected)
            for induced_vtx ∈ disconnected_vtx
                if insertion ⊆ induced_vtx && is_connected(induced_subgraph(aux, induced_vtx)[1])
                    connected_subgraph[length(induced_vtx)] += 1
                end
            end
            num_connected_insertions[insertion] = connected_subgraph
        end
        return (num_connected, num_connected_insertions)
    end
end

##

#-----------------------------------------------------------------------------#
# @6. Confiabilidade
#-----------------------------------------------------------------------------#

##
"""
    float_node_reliability(S::Dict{Int64, Int64}, p::Float64)
---

# Descrição

Calcula o valor do polinômio de confiabilidade para uma probabilidade de falha `1-p`, com `p` específico, com base no dicionário de subgrafos induzidos conexos resultante da função `Sᵣ`.

Para o cálculo, utiliza-se a relação dada por Goldschmidt em "On Reliability of Graphs with Node Failure" (1994).

# Fluxo dos Dados

- (Dict{Int64, Int64}, Float64) -> float_node_reliability() -> Float64

"""
function float_node_reliability(S::Dict{Int64,Int64}, p::Float64)
    n = length(S)
    return (sum([S[r] * (1 - p)^(n - r) * p^r for r ∈ 1:n]))
end

##
"""
    node_reliability(S::Dict{Int64, Int64})
---

# Descrição

Calcula o polinômio de confiabilidade para uma probabilidade de falha `1-p` com base no dicionário de subgrafos induzidos conexos resultante da função `Sᵣ`.

Para o cálculo, utiliza-se a relação dada por Goldschmidt em "On Reliability of Graphs with Node Failure" (1994).

# Fluxo dos Dados

- Dict{Int64, Int64} -> node_reliability() -> Polynomials.Polynomial

"""
function node_reliability(S::Dict{Int64,Int64})
    @polyvar p
    n = length(S)
    eq = @chain begin
        sum([S[r] * (1 - p)^(n - r) * p^r for r ∈ 1:n])
    end
    coeficientes = repeat([0], n)
    for ordem ∈ 1:n
        # De modo a utilizar a integração eficiente do pacote Polynomials.jl,
        # é necessário que a soma para cada `n` de 1 até a ordem do grafo
        # seja expandida até a forma a₀ + a₁x + a₂x² + ... + aₙxⁿ e só então
        # transformada em um objeto do tipo Polynomials.Polynomial. Para isso,
        # usamos o pacote TypedPolynomials.jl de modo que a simplificação seja
        # mais rápida.
        coeficientes[ordem] = TypedPolynomials.coefficient(eq, p^ordem)
    end
    prepend!(coeficientes, 0)
    return (Polynomials.Polynomial(coeficientes, :p))
end

##
"""
    area(f::Polynomials.Polynomial, inf::Float64, sup::Float64)
---

# Descrição

Retorna o resultado da integral univariada de `inf` até `sup` de um polinômio `f`.

Para o cálculo, a função utiliza o polinômio resultante da integração de `f`, sem a constante, e retorna ``F(b) - F(a)``, onde `a` e `b` recebem o menor e o maior valor, respectivamente, entre `inf` e `sup`.

# Fluxo dos Dados

- Polynomials.Polynomial -> area() -> Float64

"""
function area(f::Polynomials.Polynomial, inf::Float64 = 0.0, sup::Float64 = 1.0)
    if sup < inf
        aux = sup
        sup = inf
        inf = aux
    end
    I = integrate(f)
    return (I(sup) - I(inf))
end

##

#-----------------------------------------------------------------------------#
# @7. Medidas de Desempenho
#-----------------------------------------------------------------------------#

##
function apply_strategies(G::SimpleGraph{Int64})
    possible_edges = get_possible_edges(G)
    return (
        vcat(
        α_strategy(G, possible_edges),
        ϕ_strategy(G, possible_edges),
        β_strategy(G),
        γ_strategy(G),
        δ_strategy(G),
        r_strategy(possible_edges)
    )
    )
end

##
function get_strategic_edges(G::SimpleGraph{Int64}, id::Int64 = 0)
    table = DataFrame(
        GraphID = Int64[],
        Strategy = Symbol[],
        Edge = Vector{Int64}[],
        Info = Float64[],
        Extra = Int64[],
        Score = Float64[],
        Poly = Polynomials.Polynomial[]
    )
    for strategy ∈ apply_strategies(G)
        for idx ∈ 1:length(strategy.Edges)
            push!(
                table,
                [
                    id,
                    strategy.Strategy,
                    strategy.Edges[idx],
                    strategy.Info[idx],
                    strategy.Extra[idx],
                    0.0,
                    Polynomials.Polynomial([0], :p)
                ]
            )
        end
    end
    return (table)
end

##
function get_possible_edges2(G::SimpleGraph{Int64})
    arestas = edges(difference(complete_graph(nv(G)), G))
    aux = [[src(x), dst(x)] for x ∈ arestas]
    return (
        Strategy = :a,
        Edges = aux,
        Info = repeat([0.0], length(arestas)),
        Extra = repeat([0], length(arestas))
    )
end

##
function apply_strategies2(G::SimpleGraph{Int64})
    possible_edges = get_possible_edges(G)
    return (
        vcat(
        α_strategy(G, possible_edges),
        ϕ_strategy(G, possible_edges),
        β_strategy(G),
        γ_strategy(G),
        δ_strategy(G),
        r_strategy(possible_edges),
        get_possible_edges2(G)
    )
    )
end

##
function get_strategic_edges2(G::SimpleGraph{Int64}, id::Int64 = 0)
    table = DataFrame(
        GraphID = Int64[],
        Strategy = Symbol[],
        Edge = Vector{Int64}[],
        Info = Float64[],
        Extra = Int64[],
        Score = Float64[],
        Poly = Polynomials.Polynomial[]
    )
    for strategy ∈ apply_strategies2(G)
        for idx ∈ 1:length(strategy.Edges)
            push!(
                table,
                [
                    id,
                    strategy.Strategy,
                    strategy.Edges[idx],
                    strategy.Info[idx],
                    strategy.Extra[idx],
                    0.0,
                    Polynomials.Polynomial([0], :p)
                ]
            )
        end
    end
    return (table)
end

##
"""
    relative_deviation(G::SimpleGraph{Int64}, inf::Float64, sup::Float64)
---

# Descrição

Calcula o Índice de Desvio Relativo (RDI) das inserções de arestas sugeridas pelas heurísticas em `α_strategy`, `ϕ_strategy`, `β_strategy`, `γ_strategy`, `δ_strategy` e `r_strategy` em um grafo `G` usando como função escore a saída de `area`.

Considerando `F_{B}(G)` o maior valor e `F_{W}(G)` o pior, então o RDI da i-ésima estratégia é dado por

`RDI_{i} = \\frac{F_{B}(G) - F_{i}(G)}{F_{B} - F_{W}}`

onde `F_{i}` é o valor da função escore na i-ésima estratégia.

A ordem das estratégias é sempre: α, ϕ, β, γ, δ, r. A saída é um objeto do tipo `DataFrame`.

# Colunas da Saída

- `GraphID` (Int64): ID do grafo, caso tenha.

- `Strategy` (Symbol): Heurística que indicou a inserção.

- `Edge` ({Vector{Int64}): Aresta indicada.

- `Info` (Float64): Informação dependente da heurística. Ver heurísticas.

- `Extra` (Int64): Informação dependente da heurística. Ver heurísticas.

- `Score` (Float64): Área abaixo do polinômio de confiabilidade de vértice no intervalo [inf, sup].

- `Poly` (Polynomial): Polinômio de confiabilidade de vértice da inserção.

- `RDI` (Float64): RDI da inserção.

# Fluxo dos Dados

- (SimpleGraph{Int64}, Float64, Float64) -> relative_deviation() -> DataFrame

---

"""
function relative_deviation(G::SimpleGraph{Int64}, inf::Float64 = 0.0, sup::Float64 = 1.0; all_insertions::Bool = false)
    if all_insertions
        table = get_strategic_edges2(G)
    else
        table = get_strategic_edges(G)
    end
    aux = Sᵣ(G, unique(table.Edge))[2]
    memoise = Dict{
        Vector{Int64},
        Polynomials.Polynomial
    }()
    for idx ∈ 1:length(table.Edge)
        poly = node_reliability(aux[table.Edge[idx]])
        table[idx, :Score] = area(poly, inf, sup)
        table[idx, :Poly] = poly
    end
    max_score, min_score = (maximum(table.Score), minimum(table.Score))
    if max_score ≠ min_score
        transform!(table, :Score => ByRow(x -> (max_score - x) / (max_score - min_score)) => :RDI)
    else
        transform!(table, :Score => ByRow(x -> 0) => :RDI)
    end
    return (table)
end

##
"""
    mean_relative_deviation(grafos::Vector{SimpleGraph{Int64}}, inf::Float64, sup::Float64)
---

# Descrição

Calcula o Índice de Desvio Relativo Médio (RDIM) das heurísticas de inserções de arestas usando a soma das RDI calculadas por `relative_deviation` em um vetor de grafos.

A função retorna três tabela do tipo `DataFrame`: `Table` com todas as informações contidas na tabela resultante da função `relative_deviation`, `Score` e `RDIM`, respectivamente, com a média, mediana, mínimo e máximo dos Escores e RDI de cada heurística.

# Fluxo dos Dados

- (Vector{SimpleGraph{Int64}}, Float64, Float64) -> mean_relative_deviation() -> DataFrame

---

"""
function mean_relative_deviation(graphs::Vector{SimpleGraph{Int64}}, inf::Float64, sup::Float64; allinsertions::Bool = false)
    table = DataFrame(
        GraphID = Int64[],
        Strategy = Symbol[],
        Edge = Vector{Int64}[],
        Info = Float64[],
        Extra = Int64[],
        Score = Float64[],
        Poly = Polynomials.Polynomial[],
        RDI = Float64[]
    )
    for idx ∈ 1:length(graphs)
        aux = relative_deviation(graphs[idx], inf, sup; all_insertions = allinsertions)
        aux.GraphID .= idx
        append!(table, aux)
    end
    groupedtable = groupby(table, :Strategy)
    return (
        Table = table,
        RDIM = combine(
            groupedtable,
            nrow,
            :RDI => mean => :MeanRDI,
            :RDI => median => :MedianRDI,
            :RDI => std => :StdRDI,
            :RDI => (x -> std(x) / mean(x)) => :CVRDI,
            :RDI => maximum => :MaxRDI,
            :RDI => minimum => :MinRDI
        ),
        Score = combine(
            groupedtable,
            nrow,
            :Score => mean => :MeanScore,
            :Score => median => :MedianScore,
            :Score => std => :StdScore,
            :Score => (x -> std(x) / mean(x)) => :CVScore,
            :Score => maximum => :MaxScore,
            :Score => minimum => :MinScore
        )
    )
end

##

#-----------------------------------------------------------------------------#
# @8. Protótipo de Simulação
#-----------------------------------------------------------------------------#

## Gertsbakh & Shpungin - Models of Network Reliability, pp. 104-105.
function netlife(G::SimpleGraph{Int64}, p::Float64, M::Int64)
    modelo = Geometric(1 - p)
    arestas = edges(G)
    n = nv(G)
    τₑ = Vector{Int64}[]
    τ = Int64[]
    for iteração ∈ 1:M
        # (1) Para cada vértice, simular o tempo de vida tᵥ
        tᵥ = rand(modelo, n)
        # (2) Para cada aresta, calcular o peso como mínimo dos tᵥ dentre os vértices aos quais incide
        WG = SimpleWeightedGraph(src.(arestas), dst.(arestas), minimum.(x -> tᵥ[x], [[src(i), dst(i)] for i ∈ arestas]))
        # (3) Construir a árvore geradora máxima (Algoritmo de Kruskal)
        agm = kruskal_mst(WG; minimize = false)
        # (4) Identificar a aresta de menor peso na árvore geradora máxima
        push!(τₑ, [last(agm).src, last(agm).dst])
        push!(τ, last(agm).weight)
    end
    # (5) Ordenar as m réplicas do tempo de vida do grafo
    sort!(τ)
    # (6) Estimar Fₙ(t) = #(τ ≤ t)/m; para t = 1,2,...,k
    Fn = Float64[]
    for num ∈ unique(τ)
        push!(Fn, sum(τ[1:findlast(x -> x == num, τ)]) / M)
    end
    # (7) Estimar Var(Fₙ(t)) = (Fₙ(t) * (1 - Fₙ(t)))/m
    VFn = (Fn .* (1 .- Fn)) ./ M
    return (Est = Fn, Var = VFn, Edg = τₑ, Wgh = τ)
end

## Gertsbakh & Shpungin - Models of Network Reliability, p. 109.
function netlife_spectrum(G::SimpleGraph{Int64}, p::Float64, M::Int64)
    arestas = edges(G)
    m = ne(G)
    modelo = Binomial(m, p)
    N = repeat([0], m)
    for iteração ∈ 1:M
        agm = kruskal_mst(SimpleWeightedGraph(src.(arestas), dst.(arestas), randperm(m)); minimize = false)
        N[last(agm).weight] += 1
    end
    fn = N ./ M
    Fn = sum([fn[r] * cdf(modelo, m - r) for r ∈ 1:m])
    soma = 0.0
    for j ∈ 2:m
        for i ∈ 1:(j-1)
            soma += cdf(modelo, m - i) * cdf(modelo, m - j) * fn[i] * fn[j]
        end
    end
    VFn = (sum([fn[r] * (1 - fn[r]) * cdf(modelo, m - r)^(2) for r ∈ 1:m]) - 2 * soma) / M
    return (Fn, VFn, fn)
end

##
function intervalo(θ::Float64, Vθ::Float64, fn::Vector{Float64}, M::Int64; α::Float64 = 0.05, conservador::Bool = false)
    if conservador
        return (θ ± (quantile(Normal(), 1 - α / 2) / (2 * sqrt(M))))
    else
        return (θ ± ((quantile(TDist(M - 1), 1 - α / 2) * sqrt(Vθ)) / sqrt(M - 1)))
    end
end

##
function avaliar_escore(G::SimpleGraph{Int64}, id::Int64, p::Float64, M::Int64; conservador::Bool = true)
    tabela = get_heuristics(G, id)
    arestas = unique(tabela.Edge)
    dados = Dict{Vector{Int64},Interval{Float64}}()
    for aresta ∈ arestas
        aux = copy(G)
        add_edge!(aux, aresta...)
        get!(dados, aresta, 1 - intervalo(netlife_spectrum(aux, p, M)...; conservador = conservador))
    end
    dados = DataFrame(Edge = collect(keys(dados)), Estimate = collect(values(dados)))
    aux = innerjoin(
        tabela,
        dados,
        on = :Edge
    )
    min_score, max_score = extrema(aux.Estimate)
    if sup(max_score) ≠ sup(min_score)
        transform!(aux, :Estimate => ByRow(x -> (sup(max_score) - x) / (sup(max_score) - sup(min_score))) => :RDI)
    else
        transform!(aux, :Estimate => ByRow(x -> 0 .. 0) => :RDI)
    end
    return (aux)
end

##
function simular_rdim(graphs::Vector{SimpleGraph{Int64}}, p::Float64, M::Int64; conservador::Bool = true)
    table = DataFrame(
        GraphID = Int64[],
        Strategy = Symbol[],
        Edge = Vector{Int64}[],
        Estimate = Interval{Float64}[],
        RDI = Interval{Float64}[]
    )
    barra = Progress(length(graphs))
    Threads.@threads for idx ∈ 1:length(graphs)
        aux = avaliar_escore(graphs[idx], idx, p, M; conservador = conservador)
        append!(table, aux)
        next!(barra)
    end
    groupedtable = groupby(table, :Strategy)
    return (
        Table = table,
        RDIM = combine(
            groupedtable,
            nrow,
            :RDI => mean => :MeanRDI
        )
    )
end

##
