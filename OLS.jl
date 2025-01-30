#################
### OLS Model ###
#################

struct olsModel <: Baseline
    p::Int64 # Temporal lag
    d::Int64 # Dimension (1 = linear, 2 = square, ...)
    isPos::Bool
    function olsModel(p::Int64 = 3, d::Int64 = 1, isPos::Bool = true)
        if p < 1 + d
            error("Insufficient temporal lag.")
        end
        new(p, d, isPos)
    end
end

mutable struct olsParameter
    β::Vector{T} where {T <: Real} # Regression coefficients
end

mutable struct olsFitted
    x::Vector{T} where {T <: Real}
    model::olsModel
    par::olsParameter
end

# Notation: Last observation has time index 0 for regression
function fit(x::Vector{T},
             model::olsModel) where {T <: Real}
    #
    y = x[end-model.p + 1:end]
    
    X = zeros(model.p, 1 + model.d)
    X[:, 1] .= 1.0
    tSeq = 1 - model.p:0
    for i = 1:model.d
        X[:, 1 + i] = tSeq .^ i
    end
    β = inv(X'X)*X'y
    par = olsParameter(β)
    olsFitted(x, model, par)
end


function predict(fit::olsFitted,
                 h::Int64,
                 quant::Vector{Float64})
    #
    T = length(fit.x)
    p = fit.model.p
    d = fit.model.d

    out = fill(fit.par.β[1], h)
    for i = 1:d
        out .+= fit.par.β[1 + i] * collect(1:h) .^ i
    end

    # For prediction errors, use previous h-step-ahead errors
    ϵ = zeros(T - p - h, h)

    tSeq = 1 - p:0
    X = zeros(p, 1 + d)
    X[:, 1] .= 1.0
    for j = 1:d
        X[:, 1 + j] = tSeq .^ j
    end
    
    for i = 1:T - h - p
        xx = fit.x[i:i+p-1]
        y = xx[end-p + 1:end]

        β = inv(X'X)*X'y

        fc = fill(β[1], h)
        for j = 1:d
            fc .+= β[1 + j] .* (1:h) .^ j
        end
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
