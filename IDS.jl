###########################
### Extrapolation Model ###
###########################

# Idea: If the last 3 observations "go into the same direction",
#       forward that trend by fitting an OLS
#       Otherwise, return constant forecast

struct idsModel <: Baseline
    p::Int64
    isPos::Bool
    function idsModel(p::Int64 = 3, isPos::Bool = true)
        new(p, isPos)
    end
end

mutable struct idsParameter
    a::T where {T <: Real} # Intercept
    b::T where {T <: Real} # Slope
end

mutable struct idsFitted
    x::Vector{T} where {T <: Real}
    model::idsModel
    par::idsParameter
end

# Notation: Last observation has time index 0 for regression
function fit(x::Vector{T},
             model::idsModel) where {T <: Real}
    #
    y = x[end-model.p + 1:end]
    s = sign.(diff(y))
    if length(unique(s)) == 1
        X = [ones(model.p) 1 - model.p:0]
        β = inv(X'X)*X'y
        par = idsParameter(β[1], β[2])
    else
        par = idsParameter(mean(y), 0)
    end
    idsFitted(x, model, par)
end

function predict(fit::idsFitted,
                 h::Int64,
                 quant::Vector{Float64})
    #
    out = fit.par.a .+ fit.par.b .* collect(1:h)

    # For prediction errors, use previous h-step-ahead errors
    T = length(fit.x)
    p = fit.model.p

    ϵ = zeros(T - p - h, h)
    for i = 1:T - h - p
        xx = fit.x[i:i+p-1]

        s = sign.(diff(xx[end-p+1:end]))
        y = fit.x[end-p + 1:end]

        if length(unique(s)) == 1
            X = [ones(p) 1 - p:0]
            β = inv(X'X)*X'y
        else
            β = [mean(y), 0]
        end

        fc = β[1] .+ β[2] .* (1:h)
        ϵ[i, :] = fit.x[i + p:i + p + h - 1] .- fc
    end
    ϵ = [ϵ; -ϵ]
    
    Q = zeros(length(quant), h)
    for hh = 1:h
        for q = 1:length(quant)
            Q[q, hh] = out[hh] + quantile(ϵ[:, hh], quant[q])
        end
    end

    if fit.model.isPos
        Q[Q .< 0] .= 0
    end

    return forecast(Q, quant, collect(1:h))
end