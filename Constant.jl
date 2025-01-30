#########################
### Constant Forecast ###
#########################

# Constant forecasting simply returns the last observed value
# Forecast intervals are based on historic h-step-ahead forecast errors

struct constantModel <: Baseline
    isPos::Bool
    function constantModel()
        new(true)
    end
end

mutable struct constantParameter
    μ::T where {T <: Real}
end

mutable struct constantFitted
    x::Vector{T} where {T <: Real}
    model::constantModel
    par::constantParameter
end

function fit(x::Vector{T},
             model::constantModel) where {T <: Real}
    constantFitted(x, model, constantParameter(x[end]))
end

function predict(fit::constantFitted,
                 h::Int64,
                 quant::Vector{Float64})
    out = fill(fit.par.μ, h)
    Q = zeros(length(quant), h)

    for hh = 1:h
        ϵh = fit.x[1:end-hh] .- fit.x[hh+1:end]
        ϵh = [ϵh; -ϵh]
        for q = 1:length(quant)
            Q[q, hh] = out[hh] + quantile(ϵh, quant[q])
        end   
    end
    if fit.model.isPos
        Q[Q .< 0] .= 0
    end
    return forecast(Q, quant, collect(1:h))
end