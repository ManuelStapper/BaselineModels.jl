#########################
### Marginal Forecast ###
#########################

struct marginalModel <: Baseline
    p::Int64
    isPos::Bool
    function marginalModel(p::Int64, isPos::Bool = true)
        new(p, isPos)
    end
end

mutable struct marginalParameter
    μ::T where {T <: Real}
end

mutable struct marginalFitted
    x::Vector{T} where {T <: Real}
    model::marginalModel
    par::marginalParameter
end

function fit(x::Vector{T}, model::marginalModel) where {T <: Real}
    pp = minimum([model.p, length(x)])
    est = mean(x[end-pp + 1:end])
    marginalFitted(x, model, marginalParameter(est))
end

function predict(fit::marginalFitted,
                 h::Int64,
                 quant::Vector{Float64})
    #
    out = fill(fit.par.μ, h)
    quant = sort(unique(round.([quant; 1 .- quant; 0.5], digits = 4)))
    
    # Get intervals from historic h-step-ahead forecast errors
    Q = zeros(length(quant), h)
    ϵ = zeros(length(fit.x) - h, h)
    for t = 1:length(fit.x) - h
        tStart = maximum([1, t - fit.model.p])
        temp = mean(fit.x[tStart:t])
        ϵ[t, :] = fit.x[t:t+h-1] .- temp
    end
    

    for hh = 1:h
        pL = fit.model.p + hh
        pU = 1 + hh
        ϵh = (t -> fit.x[t] - mean(fit.x[t - pL:t-pU])).(fit.model.p + hh + 1:length(fit.x))
        ϵh = [ϵh; -ϵh]
        for q = 1:length(quant)
            Q[q, hh] = out[hh] + quantile(ϵh, quant[q])
        end   
    end
    
    if fit.model.isPos
        Q[Q .< 0] .= 0
    end

    interval = forecastInterval[]
    for hh = 1:h
        ls = Q[1:Int64((size(Q)[1] - 1)/2), hh]
        us = reverse(Q[Int64((size(Q)[1] - 1)/2) + 2:end, hh])
        αs = quant[1:Int64((size(Q)[1] - 1)/2)]*2
        push!(interval, forecastInterval(αs, ls, us))
    end

    return forecast(1:h, mean = out, median = out, interval = interval)
end