#------------------------------------------------------------------------------#
# @1. Pacotes
#------------------------------------------------------------------------------#

##
using Combinatorics
using CSV
using DelimitedFiles
using FileIO
using JLD2
using Graphs
using OhMyREPL
using Primes
using ProgressMeter
using Random

##
const path = "/home/fausto-mpj/Programming/Julia/Julia_projects/ENCE_IC_Grafos"

##
cd(path)

##

#------------------------------------------------------------------------------#
# @2. Funções auxiliares
#------------------------------------------------------------------------------#

##
function reverselookup(d, v)
    """
    Função para inverter dicionário, ou seja, a partir dos valores obter
    as chaves. Esta função retorna erro quando o dicionário não corresponde
    a uma função inversível.
    """
    for k in keys(d)
        if d[k] == v
            return (k)
        end
    end
    error("ERROR: This dictionary isn't a reversible function.")
end

##
function combination2graph(vector::Array{Array{Tuple{Int64,Int64},1},1}, n::Int64)
    """
    A função `combination2graph` é uma função de apoio para a `generate graph`,
    facilitando a paralelização ao colocar os subconjuntos com k-elementos em
    processos distintos. Em cada escopo local uma variável `aux` é criada e
    manipulada quanto a adição de arestas sem afetar os demais processos. A
    penalidade é o maior custo em memória, dado que cada escopo criará uma
    cópia dos subconjuntos de k-elementos. Dada a quantidade de elementos que
    cada subconjunto pode ter [n! / k!(n-k)!], então o custo em memória pode
    ser proibitivo.
    """
    # Criando vetor (vazio) com elementos do tipo grafo
    graphs = SimpleGraph[]
    # Povoando vetor `graphs` com os grafos de `n` vértices que são conexos
    # e possuem todos os vértices com grau maior do que 1 (sem "folhas").
    for edges ∈ vector
        aux = SimpleGraph(n)
        for edge ∈ edges
            add_edge!(aux, edge)
        end
        if all(LightGraphs.degree(aux) .> 1) && is_connected(aux) && ne(aux) < ((nv(aux)^2 - nv(aux)) / 2)
            push!(graphs, aux)
        end
    end
    # Retornando os grafos do vetor de combinações
    return (graphs)
end

##
function gerar_arestas_primos(n::Int64)
    # Gerando todas as arestas para grafo de ordem n
    edges = collect(combinations(1:n, 2))
    # Gerando os números primos associados a cada aresta
    prime_id = BigInt.(primes(prime(length(edges))))
    # Criando dicionário <primo> => <aresta>
    dictionary = Dict(zip(prime_id, edges))
    # Escrevendo arquivo com os códigos
    open("grafos_ordem_$(n)_dicionario.csv", "w") do io
        writedlm(
            io,
            hcat(
                prime_id,
                reduce(vcat, edges')
            )
        )
    end
    # Retornando o dicionário
    return (dictionary)
end

##

#------------------------------------------------------------------------------#
# @3. Geração direta
#------------------------------------------------------------------------------#

# Queremos gerar *todos* os grafos conexos, sem vértices-folha, de ordem `n`.
# O problema é garantir que o tempo de execução, a quantidade de memória e o
# espaço de armazenamento seja razoável para a escala em que estamos operando.
# Ao contrário do código anterior em Python, não estamos verificando o
# isomorfismo dos grafos, ou seja, estamos considerando grafos etiquetados!
# Perceba que isto significa que há um número bem mais elevado de grafos em
# cada ordem `n` em comparação com a abordagem anterior.

##
function gerar_grafos(n::Int64)
    """
       A função `gerar_grafos` gera todos os subgrafos (etiquetados) com `n`
       vértices do grafo completo Kₙ de modo serial. O algoritmo utiliza o
       conjunto das partes das arestas de Kₙ.
       A complexidade computacional é O(2ⁿ), portanto exige cuidado nos valores
       de `n`.
    """
    grafos = SimpleGraph[]
    combinações = collect(combinations(
        [(src(x), dst(x)) for x ∈ edges(complete_graph(n))]
    ))
    for arestas ∈ combinações
        if length(arestas) ≥ (n - 1)
            aux = SimpleGraph(n)
            for aresta ∈ arestas
                add_edge!(aux, aresta...)
            end
            if all(LightGraphs.degree(aux) .> 1) && is_connected(aux) && ne(aux) < ((nv(aux)^2 - nv(aux)) / 2)
                push!(grafos, aux)
            end
        end
    end
    return (grafos)
end

##

#------------------------------------------------------------------------------#
# @4. Geração serial
#------------------------------------------------------------------------------#

##
function gerar_grafos_serial(n::Int64)
    """
    A função `gerar_grafos_serial` gera todos os subconjuntos de arestas para
    um grafo com `n` vértices tal que o grafo seja conexo e não tenha vértices
    com grau igual a 1. O algoritmo utiliza as combinações do conjunto de
    arestas do grafo completo Kₙ. Como não realiza a enumeração de uma só
    vez do conjunto das partes, então a utilização de memória é razoavelmente
    menor do que a das demais funções. Em contrapartida, como precisa escrever
    no disco a cada iteração quais são os subconjuntos de arestas, então o
    tempo de execução é bem mais elevado e o tamanho em disco destes
    subconjuntos deve ser considerado.
    """
    open("grafos_ordem_$(n).csv", "w") do io
        for id_in ∈ Int64(n - 1):Int64((n^2 - n) / 2)
            combinações = collect(combinations(
                [(src(x), dst(x)) for x ∈ edges(complete_graph(n))],
                id_in
            ))
            for arestas ∈ combinações
                aux = SimpleGraph(n)
                for aresta ∈ arestas
                    add_edge!(aux, aresta)
                end
                if all(LightGraphs.degree(aux) .> 1) && is_connected(aux) && ne(aux) < ((nv(aux)^2 - nv(aux)) / 2)
                    writedlm(io, [arestas], ',')
                end
            end
        end
    end
end

##

#------------------------------------------------------------------------------#
# @5. Geração paralela
#------------------------------------------------------------------------------#

##
function generate_graphs(n::Int64; parallel::Bool = true)
    """
    A função `generate_graphs` gera todos os subgrafos (etiquetados) com `n`
    vértices do grafo completo Kₙ usando paralelização. O algoritmo não utiliza
    o conjunto das partes das arestas de Kₙ, selecionando somente as
    combinações a partir de `n - 1` e até a cardinalidade do conjuntos das
    arestas de Kₙ.
    A complexidade computacional ainda assim é elevada, e portanto exige
    cuidado nos valores de `n`, sob risco adicional de falha devido ao uso
    excessivo de memória. Ver `combination2graph` para mais detalhes.
    """
    # Criando vetor (vazio) com elementos do tipo grafo
    graphs = SimpleGraph[]
    # Criando vetor com todas a duplas ordenadas com as arestas de Kₙ
    kn_edge = Tuple.(collect(combinations(1:n, 2)))
    # Criando vetor (vazio) com elementos do tipo vetor (de inteiros)
    combination = Array{Array{Tuple{Int64,Int64},1},1}[]
    # Povoando vetor `combination` com combinações de `n-1` a `length(kn_edge)`
    # Note: Geramos somente as combinações com número de arestas maior ou igual
    # a `(n-1)` porque todo grafo de `n` vértices com menos de `(n-1)` arestas
    # é necessariamente desconexo.
    Threads.@threads for num ∈ (n-1):length(kn_edge)
        push!(combination, collect(combinations(kn_edge, num)))
    end
    # Povoando vetor `graphs` com os grafos de `n` vértices que são conexos e
    # possuem todos os vértices com grau maior do que 1 (sem "folhas").
    # Se `parallel` for `true` (default), usaremos a paralelização para as
    # inserções e verificações, caso contrário usaremos o modo serial.
    if parallel
        Threads.@threads for vector ∈ combination
            append!(graphs, combination2graph(vector, n))
        end
    else
        for vector ∈ combination
            for edges ∈ vector
                aux = SimpleGraph(n)
                for edge ∈ edges
                    add_edge!(aux, edge)
                end
                if all(LightGraphs.degree(aux) .> 1) && is_connected(aux) && ne(aux) < ((nv(aux)^2 - nv(aux)) / 2)
                    push!(graphs, aux)
                end
            end
        end
    end
    # Retornando vetor de grafos
    return (graphs)
end

##

#------------------------------------------------------------------------------#
# @6. Geração por primos
#------------------------------------------------------------------------------#

##
function gerar_grafos_primos(n::Int64)
    dictionary = gerar_arestas_primos(n)
    open("grafos_ordem_$(n)_id.csv", "w") do io
        for num ∈ (n-1):Int((n^2 - n) / 2)
            for prime_id ∈ prod.(combinations(collect(keys(dictionary)), num))
                graph = SimpleGraph(n)
                edges = [dictionary[prime_factor] for prime_factor ∈ factor(Vector, prime_id)]
                for edge ∈ edges
                    add_edge!(graph, edge...)
                end
                if all(LightGraphs.degree(graph) .> 1) && is_connected(graph) && ne(graph) < ((nv(graph)^2 - nv(graph)) / 2)
                    writedlm(io, prime_id)
                end
            end
        end
    end
end

##

#------------------------------------------------------------------------------#
# @7. Benchmark
#------------------------------------------------------------------------------#

##
@time gerar_grafos(7)

# A `gerar_grafos` é a nossa função base e executou para `n = 7` em 4 segundos.

##
@time generate_graphs(7; parallel = false)

# A `generate_graphs` sem utilização de paralelização para a geração dos grafos
# demorou 4 segundos de tempo de execução para `n = 7`, obtendo mesmo
# desempenho que a função `gerar_grafos`.

##
@time generate_graphs(7)

# A `generate_graphs` utiliza paralelização para a geração dos grafos e
# executou para `n = 7` em 2 segundos.
# O limite que conseguimos rodar acima foi com `n = 7`. Tentando com `n = 8`
# em ambas as funções levaram a reboot devido ao uso excessivo de memória.
# A máquina de teste é um Intel i7 7700K, 32GB DDR4 2400Mhz e GPU RTX2080,
# com OS GNU-Linux (Fedora 34) e utilização padrão de memória ao iniciar na
# casa dos 500MB.

##
@time gerar_grafos_serial(7)

# Com a `gerar_grafos_serial` conseguimos rodar com `n = 8`, no entanto o
# tempo de execução foi de 1307 segundos (~21 minutos). Em comparação, para
# `n = 7`, a mesma função teve tempo de execução de 6 segundos. Além disso,
# o arquivo de texto com os dados das arestas em `n = 8` totalizou 16.8GB.
# Novamente, para o caso de `n = 7`, a mesma função retornou um arquivo com
# tamanho de 83.9MB. É provável que o caso de `n = 9` não irá caber nos 1.5TB
# de armazenamento disponíveis na máquina de teste.
# Por outro lado, o uso máximo de memória em `n = 8` foi de 19GB, logo
# conseguimos diminuir razoavelmente a quantidade de memória utilizada.
# Não sabemos se seria o suficiente para `n = 9`, porque o problema da memória
# foi deslocado pelo problema do espaço de armazenamento.

##
@time gerar_grafos_primos(7)

# Com a `gerar_grafos_primos` conseguimos com que o arquivo com os grafos
# ficasse com 17.8MB para `n = 7` e tempo de execução de 54 segundos.
# Com `n = 8`, a função teve como tempo de execução 13570 segundos
# (~226 minutos) e gerou um arquivo de texto totalizando 4.1GB. O uso máximo
# de memória foi inferior ao de `gerar_grafos_serial`, tendo como pico a
# utilização de 12GB.
# Note que para ler o arquivo gerado é necessário estabelecer no `readdlm()`
# o tipo `BigInt`, caso contrário a função interpretará como `Float64` e pode
# incorrer em erro de aproximação de ponto flutuante.

##

#------------------------------------------------------------------------------#
# @8. Salvando dados
#------------------------------------------------------------------------------#

##
@showprogress 1 "Computando..." for num ∈ 4:7
    aux = generate_graphs(num)
    filename = "grafos_ordem_$(num).jld2"
    save_object(filename, aux)
end

##

#------------------------------------------------------------------------------#
# @9. Grafos aleatorios
#------------------------------------------------------------------------------#

# Amostra de 1000 de cada ordem a partir de 8 para cada modelo.

## Gerando grafos pelo modelo Erdös-Renyi
for num ∈ 9:30
    println(num)
    prob = rand(0.25:0.05:0.75, 1000)
    lista = SimpleGraph[]
    for idx ∈ 1:1000
        println(idx)
        while true
            aux = erdos_renyi(num, prob[idx])
            if all(degree(aux) .> 1) && is_connected(aux) && ne(aux) ≠ ((nv(aux)^2 - nv(aux)) / 2)
                if length(lista) >= 1 && all(x -> !Graphs.Experimental.could_have_isomorph(aux, x), lista)
                    push!(lista, aux)
                    break
                elseif length(lista) < 1
                    push!(lista, aux)
                    break
                end
            end
        end
    end
    filename = "grafos_iso_erdos_ordem_$(num).jld2"
    params = "grafos_iso_erdos_ordem_$(num)_params.jld2"
    save_object(filename, lista)
    save_object(params, prob)
end

## Gerando grafos pelo modelo Erdös-Renyi (Alternativo)
for num ∈ 10:20
    println(num)
    lista = SimpleGraph[]
    problista = Float64[]
    U = Uniform(0.1, 0.5)
    for idx ∈ 1:1000
        println(idx)
        while true
            prob = rand(U)
            aux = erdos_renyi(num, prob)
            if all(Graphs.degree(aux) .> 1) && is_connected(aux) && ne(aux) ≠ ((nv(aux)^2 - nv(aux)) / 2)
                if length(lista) >= 1 && all(x -> !Graphs.Experimental.could_have_isomorph(aux, x), lista)
                    push!(lista, aux)
                    push!(problista, prob)
                    break
                elseif length(lista) < 1
                    push!(lista, aux)
                    push!(problista, prob)
                    break
                end
            end
        end
    end
    filename = "grafos_iso_erdos2_ordem_$(num).jld2"
    params = "grafos_iso_erdos2_ordem_$(num)_params.jld2"
    save_object(filename, lista)
    save_object(params, problista)
end

## Gerando grafos pelo modelo Barabási-Albert
for num ∈ 10:30
    println(num)
    lista = SimpleGraph[]
    param = Int64[]
    for idx ∈ 1:500
        println(idx)
        while true
            k = rand(2:Int(num - 2))
            aux = barabasi_albert(num, k)
            if all(degree(aux) .> 1) && is_connected(aux) && ne(aux) ≠ ((nv(aux)^2 - nv(aux)) / 2)
                if length(lista) >= 1 && all(x -> !Graphs.Experimental.could_have_isomorph(aux, x), lista)
                    push!(lista, aux)
                    push!(param, k)
                    break
                elseif length(lista) < 1
                    push!(lista, aux)
                    push!(param, k)
                    break
                end
            end
        end
    end
    filename = "grafos_iso_barabasi_ordem_$(num).jld2"
    params = "grafos_iso_barabasi_ordem_$(num)_params.jld2"
    save_object(filename, lista)
    save_object(params, param)
end

## Gerando grafos pelo modelo Watts-Strogatz
for num ∈ 10:30
    println(num)
    prob = rand(0.25:0.05:0.75)
    lista = SimpleGraph[]
    paramMD = Int64[]
    paramP = Float64[]
    for idx ∈ 1:500
        println(idx)
        while true
            prob = rand(0.25:0.05:0.75)
            md = rand(2:(num-2))
            aux = watts_strogatz(num, md, prob)
            if all(degree(aux) .> 1) && is_connected(aux) && ne(aux) ≠ ((nv(aux)^2 - nv(aux)) / 2)
                if length(lista) >= 1 && all(x -> !Graphs.Experimental.could_have_isomorph(aux, x), lista)
                    push!(lista, aux)
                    push!(paramMD, md)
                    push!(paramP, prob)
                    break
                elseif length(lista) < 1
                    push!(lista, aux)
                    push!(paramMD, md)
                    push!(paramP, prob)
                    break
                end
            end
        end
    end
    filename = "grafos_iso_strogatz_ordem_$(num).jld2"
    paramsMD = "grafos_iso_strogatz_ordem_$(num)_paramsMD.jld2"
    paramsP = "grafos_iso_strogatz_ordem_$(num)_paramsP.jld2"
    save_object(filename, lista)
    save_object(paramsMD, paramMD)
    save_object(paramsP, paramP)
end
##

## Gerando grafos pelo modelo Watts-Strogatz (Alternativo)
for num ∈ 10:20
    println(num)
    U = Uniform(0.25, 0.75)
    lista = SimpleGraph[]
    paramMD = Int64[]
    paramP = Float64[]
    for idx ∈ 1:500
        println(idx)
        while true
            prob = rand(U)
            md = rand(2:(num-2))
            aux = watts_strogatz(num, md, prob)
            if all(Graphs.degree(aux) .> 1) && is_connected(aux) && ne(aux) ≠ ((nv(aux)^2 - nv(aux)) / 2)
                if length(lista) >= 1 && all(x -> !Graphs.Experimental.could_have_isomorph(aux, x), lista)
                    push!(lista, aux)
                    push!(paramMD, md)
                    push!(paramP, prob)
                    break
                elseif length(lista) < 1
                    push!(lista, aux)
                    push!(paramMD, md)
                    push!(paramP, prob)
                    break
                end
            end
        end
    end
    filename = "grafos_iso_strogatz2_ordem_$(num).jld2"
    paramsMD = "grafos_iso_strogatz2_ordem_$(num)_paramsMD.jld2"
    paramsP = "grafos_iso_strograz2_ordem_$(num)_paramsP.jld2"
    save_object(filename, lista)
    save_object(paramsMD, paramMD)
    save_object(paramsP, paramP)
end

#------------------------------------------------------------------------------#
# @10. Loucura
#------------------------------------------------------------------------------#

# Esgotada as boas idéias, partirei para a loucura.

# Seja a lista de números primos associada à lista de arestas de um grafo
# completo K_{n}. Se cada grafo G é um número composto e cada aresta é um fator
# primo desta lista de números primos, então adicionar arestas é o mesmo que
# multiplicar o número que representa o grafo por um número primo que não é um
# de seus fatores e que está presente na lista de números primos. Analogamente
# para a remoção de arestas e a divisão inteira.

# O número que representa um grafo G conexo tem, no mínimo, (n - 1) fatores
# primos, onde n é a cardinalidade do conjunto de vértices do grafo (|V|).
# Todo grafo conexo tem número com no mínimo (n - 1) fatores primos. Condição
# necessária, e não suficiente. É possível que um grafo tenha (n - 1) arestas
# e ainda assim seja desconexo. Exemplo: Grafo com n = 4, m = 3 e um ciclo,
# onde m é a cardinalidade do conjunto de arestas do grafo (|E|). Se um grafo G
# tem (n - 1) arestas e é conexo, então dizemos que esse grafo é uma árevore.

# Seja T uma árvore e subgrafo de G. Então todo subgrafo de G cujo número é um
# múltiplo do número de T também é conexo.

# O número de árvores em G é \binom(m, n-1), a cardinalidade do
# conjunto de combinações (n-1)-a-(n-1) das m arestas de G. Se G é o K_{n},
# então m = (n^2 - n)/2.

# Seja τ o conjunto de todos os subgrafos árvore de G. Então todo subgrafo conexo
# de G de ordem n é múltiplo de ao menos um elemento de τ.
# A partir de τ podemos gerar todos os subgrafos conexos ao multiplicar cada
# membro de τ pelos fatores primos que não o compõem e que compõem G.

##
n = 5
# Gerando todas as arestas para grafo de ordem n
edges = collect(combinations(1:n, 2))
# Gerando os números primos associados a cada aresta
prime_id = BigInt.(primes(prime(length(edges))))
# Criando dicionário <primo> => <aresta>
dictionary = Dict(zip(prime_id, edges))
# Criando todos os grafos com (n - 1) arestas
candidates = prod.(combinations(prime_id, n - 1))

# Agora queremos saber quantos e quais destes candidatos são árvores
# Se um candidato é conexo, então é uma árvore

##
# Supondo n = 5, queremos agrupar as arestas da seguinte maneira:
# g₁ = [[1,2], [1,3], [1,4], [1,5]] = [2, 3, 5, 7]
# g₂ = [[2,3], [2,4], [2,5]] = [11, 13, 17]
# g₃ = [[3,4], [3,5]] = [19, 23]
# g₄ = [[4,5]] = [29]
# Temos que nos 4 grupos estão todas as arestas de K_{5} e cada um
# dos grupos tem um número decrescente de elementos, começando com
# (n-1) até 1, onde os elementos do primeiro grupo são os de índice
# 1 até (n-1), do segundo grupo são os de índice n até 2(n-1)-1, dado
# que queremos que tenha um elemento a menos que o anterior, e assim
# por diante.
# Por outro lado, para as mesmas arestas, vamos criar os seguintes grupos:
# G₁ = [[1,2], [1,3], [1,4], [1,5]]
# G₂ = [[2,1], [2,3], [2,4], [2,5]]
# G₃ = [[3,1], [3,2], [3,4], [3,5]]
# G₄ = [[4,1], [4,2], [4,3], [4,5]]
# G₅ = [[5,1], [5,2], [5,3], [5,4]]
# Temos que o grupo Gₖ contém todas as arestas que incidem no vértice k.

##
predecessor(n::Int64) = n - 1

sucessor(n::Int64) = n + 1

##
function diminishing_groups(n::Int64, vector::Vector)
    id_in = 1
    id_out = n - 1
    k = 1
    groups = Vector[]
    while (id_in <= id_out)
        push!(groups, vector[id_in:id_out])
        id_in = sucessor(id_out)
        id_out = predecessor((n - 1) - k + id_in)
        k += 1
    end
    return (groups)
end

##
function maximal_groups(n::Int64, dictionary::Dict)
    groups = Vector[]
    for idx ∈ 1:n
        push!(groups, collect(keys(dictionary))[idx.∈values(dictionary)])
    end
    return (groups)
end

##
groups = diminishing_groups(n, prime_id)

big_groups = maximal_groups(n, dictionary)

##
# Dada a função acima, temos que todos os elementos do gᵢ são vizinhos
# de quaisquer outros elementos também pertencentes ao Gᵢ. Além disso,
# o j-ésimo elemento de gᵢ é vizinho dos elementos pertencentes ao G(ᵢ+ⱼ).
# Temos que um grafo é conexo se a partir de um fator primo qualquer
# conseguimos construir um conjunto com todos os fatores do grafo apenas
# pela relação de vizinhança.

##
# Grafo conexo [[1,3], [1,4], [2,4], [3,5]]:
grafo_1 = BigInt(3 * 5 * 13 * 23)
# Grafo desconexo [[1,2], [1,3], [2,3], [4,5]]:
grafo_2 = BigInt(2 * 3 * 11 * 29)

# O fator 3 pertence ao g₁ e está na posição 2 de g₁, logo temos que 3
# é vizinho de qualquer outro fator que esteja em G₁ ou G₃.
# O fator 23 pertence ao g₃ e está na posição 2 de g₃, logo temos que 23
# é vizinho de qualquer outro fator que esteja em G₃ ou G₅.
# Se a partir de um vértice qualquer conseguimos alcançar todos os n grupos Gᵢ,
# então o grafo é conexo.

##
function prime_connectivity(grafo::BigInt, n::Int64, small_groups::Vector, big_groups::Vector)
    control_Group = fill(0, n)
    candidates = BigInt[]
    prime_factors = collect(keys(factor(grafo)))
    push!(candidates, first(prime_factors))
    while !isempty(candidates) || all(control_Group .== 0)
        i = findfirst(first(candidates) .∈ small_groups)
        j = indexin(first(candidates), small_groups[i])[1]
        popfirst!(candidates)
        if control_Group[i] == 0
            control_Group[i] = 1
            append!(candidates, intersect(prime_factors, small_groups[i]))
        end
        if control_Group[i+j] == 0
            control_Group[i+j] = 1
            append!(candidates, intersect(prime_factors, big_groups[i+j]))
        end
        if all(control_Group .== 1)
            return (true)
        end
    end
    return (false)
end

##
prime_connectivity(grafo_1, n, groups, big_groups)

prime_connectivity(grafo_2, n, groups, big_groups)

##
