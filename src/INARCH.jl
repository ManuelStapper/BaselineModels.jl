abstract type Baseline end

mutable struct inarchModel <: Baseline
    p::Int64
    m::Int64
    k::Int64
    nb::Bool
    function inarchModel(p, m = 0, k = 1, nb = true)
        if m < 0
            m = 0
        end
        if k < 0
            k = 1
        end
        if m <= 0
            k = 0
        end
        new(p, m, k, nb)
    end
end

mutable struct inarchParameter
    β0::Float64
    α::Vector{Float64}
    ϕ::Float64
    γ::Vector{Float64}
    function inarchParameter(β0, α, ϕ, γ)
        if !(typeof(α) <: Vector) & (length(α) == 1)
            α = [α]
        end
        if length(α) == 0
            α = Float64[]
        end
        if length(γ) == 0
            γ = Float64[]
        end

        new(float(β0), float.(α), float(ϕ), float.(γ))   
    end
end



mutable struct inarchFitted
    x::Vector{T} where {T <: Real}
    model::inarchModel
    par::inarchParameter
end

function inarchTF(θ, x::Vector{Int64}, model::inarchModel, returnPars::Bool = false)
    T = length(x)
    p = model.p
    m = model.m
    k = model.k
    nb = model.nb

    nPar = 1 + p + 2*k + nb

    β0 = θ[1]
    α = θ[2:1+p]
    ϕ =  ifelse(nb, θ[2 + p], Inf)
    γ = θ[2 + p + nb:end]
    if returnPars
        return inarchParameter(β0, α, ϕ, γ)
    end
    if (β0 <= 0) | any(α .< 0) | any(α .> 1) | (sum(α) > 1) | (ϕ <= 0)
        return Inf
    end

    λ = fill(β0, T - p)
    seas = zeros(T)

    if m > 0
        tSeq = (1:T) ./ m .* (2π)
        @inbounds for kk in 1:k
            ωk = kk * tSeq
            seas .+= γ[kk] .* sin.(ωk) .+ γ[k + kk] .* cos.(ωk)
        end
    end
    seas = exp.(seas)
    xScl = x ./ seas

    @inbounds for i in 1:p
        λ .+= α[i] .* @view xScl[(p+1:T) .- i]
    end

    λ = λ .* seas[p+1:end]

    if nb
        pp = ϕ ./ (ϕ .+ λ)
        if any(.!(0 .< pp .< 1))
            return Inf
        end
        out = -sum(logpdf.(NegativeBinomial.(ϕ, pp), @view x[p+1:end]))
    else
        out = -sum(logpdf.(Poisson.(λ), @view x[p+1:end]))
    end

    out
end

function fit(x::Vector{T1}, model::inarchModel) where {T1 <: Real}
    inits = [mean(x)/2; fill(0.5 ./ model.p, model.p); ifelse(model.nb, 3.0, zeros(0)); zeros(model.k*2)]
    optRaw = optimize(vars -> inarchTF(vars, x, model), inits)
    par = inarchTF(optRaw.minimizer, x, model, true)
    inarchFitted(x, model, par)
end

function predict(fit::inarchFitted,
                 h::Int64,
                 quant::Vector{Float64},
                 nChain::Int64 = 10000)
    #
    x = fit.x
    T = length(x)
    model = fit.model
    p = model.p
    m = model.m
    k = model.k
    nb = model.nb

    β0 = fit.par.β0
    α = fit.par.α
    ϕ = fit.par.ϕ
    γ = fit.par.γ

    quant = sort(unique(round.([quant; 1 .- quant; 0.5], digits = 4)))

    X = zeros(Int64, nChain, h)
    tSeq = (T+1-p:T+h) ./ m .* (2*π)
    seas = zeros(h + p)
    if m > 0
        for kk = 1:k
            seas .+= γ[kk] .* sin.(tSeq)
            seas .+= γ[k + kk] .* cos.(tSeq)
        end
    end
    
    seas = exp.(seas)
    xOld = x[end-p + 1:end] ./ seas[1:p]
    seas = seas[p+1:end]

    for hh in 1:h
        λ = fill(β0, nChain)
        for i = 1:p
            if hh - i <= 0
                λ .+= xOld[p + hh - i] * α[i]
            else
                λ .+= X[:, hh - i] .* α[i]
            end
        end

        λ .*= seas[hh]

        if nb
            pp = ϕ ./ (ϕ .+ λ)
            X[:, hh] = rand.(NegativeBinomial.(ϕ, pp))
        else
            X[:, hh] = rand.(Poisson.(λ))
        end
    end

    meanY = mean(X, dims = 1)[1, :]
    Q = zeros(length(quant), h)
    for i = 1:h
        Q[:, i] = quantile(X[:, i], quant)
    end

    interval = forecastInterval[]
    for hh = 1:h
        ls = Q[1:Int64((size(Q)[1] - 1)/2), hh]
        us = reverse(Q[Int64((size(Q)[1] - 1)/2) + 2:end, hh])
        αs = quant[1:Int64((size(Q)[1] - 1)/2)]*2
        push!(interval, forecastInterval(αs, ls, us))
    end

    medY = Q[Int64((size(Q)[1] - 1)/2) + 1, :]


    return forecast(1:h, mean = meanY, median = medY, interval = interval)
end


