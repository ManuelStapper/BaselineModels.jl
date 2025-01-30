###############################
### Very general ARMA(p, q) ###
###############################

# X_t - μ_t = ϵ_t + \sum_{i = 1}^p α_i (X_{t-i} - μ_{t-i}) + \sum_{i = 1}^q β_{i} ϵ_{t-i}
# μ_t = μ(t, θ) some function
# Note X_t - μ_t as Y_t

struct armaModel <: Baseline
    p::Int64
    q::Int64
    μ::Function
    μDim::Int64
    isPos::Bool
    function armaModel(p::Int64, q::Int64, μ::Function, μDim::Int64; isPos::Bool = true)
        new(p, q, μ, μDim, isPos)
    end
end

# Sloppy version of a wrapper function (unnamed μ function)
# Make constant seasonality default (not zero)
function armaModel(p::Int64, q::Int64; m::Int64 = 0, trend::Bool = false, isPos::Bool = true)
    if m[1] == 0
        if trend
            μ = (θ, t) -> θ[1] + θ[2]*t
            μDim = 2
        else
            μ = (θ, t) -> θ[1]
            μDim = 1
        end
    else
        if trend
            μ = (θ, t) -> θ[1] + θ[2] * sin(t/m*2*π) + θ[3] * cos(t/m*2*π) + θ[4] * t
            μDim = 4
        else
            μ = (θ, t) -> θ[1] + θ[2] * sin(t/m*2*π) + θ[3] * cos(t*2*π)
            μDim = 3
        end
    end
    return armaModel(p, q, μ, μDim, isPos = isPos)
end


mutable struct armaParameter
    α::Vector{Float64}
    β::Vector{Float64}
    μ::Vector{Float64}
    σ²::Float64
end

mutable struct armaFitted
    x::Vector{T} where {T <: Real}
    model::armaModel
    par::armaParameter
end

# Some helper functions:

# Function to translate ARMA(p, q) to MA(∞)
function ARMAtoMA(α::Vector{Float64},
                  β::Vector{Float64},
                  iMax::Int64 = 100)
    ψ = zeros(iMax)
    p = length(α)
    q = length(β)
    M = maximum([p, q+1])
    # Make α and β the same length
    a = [α; zeros(M - p)]
    b = [β; zeros(M - q)]
    for j = 1:M-1
        ψ[j] = b[j] + sum(a[1:j-1] .* ψ[j-1:-1:1]) + a[j]
    end
    for j = M:iMax
        if j == p
            ψ[j] =  sum(a[1:p-1] .* ψ[j-1:-1:1]) + a[p]
        else
            ψ[j] = sum(a[1:p] .* ψ[j-1:-1:j-p])
        end
    end
    return ψ
end

# Function that translates a vector to ARMA parameters
# Only needed for easy back and forth in optimisation
function armaθ2par(θ::Vector{T}, model::armaModel) where {T <: Real}
    p = model.p
    q = model.q
    armaParameter(θ[1:p], θ[p+1:p+q], θ[p+q+1:end-1], θ[end])
end

function armapar2θ(par::armaParameter, model::armaModel)
    [par.α; par.β; par.μ; par.σ²]
end

function fit(x::Vector{T1},
             model::armaModel) where {T1 <: Real}
    # (Negative) Log-likelihood function for optimisation
    function tf(θ, x, model)
        par = armaθ2par(θ, model)
        # Only check for positive variance, other restrictions not imposed
        if par.σ² <= 0
            return Inf
        end

        p = model.p
        q = model.q
        M = maximum([p, q])
        T = length(x)

        m = (t -> model.μ(par.μ, t)).(1:T)
        y = x .- m
        
        yHat = zeros(T)
        yHat[1:M] = zeros(M)
        for t = M+1:T
            yHat[t] = sum(par.α .* y[t-1:-1:t-p]) .+ sum(par.β .* (y[t-1:-1:t-q] .- yHat[t-1:-1:t-q]))
        end
        d = Normal.(yHat, sqrt(par.σ²))
        -sum(logpdf.(d, y))
    end

    # No dependency for initialisation
    inits = [zeros(model.p + model.q); zeros(model.μDim); var(x)]
    est = optimize(vars -> tf(vars, x, model), inits).minimizer
    estPar = armaθ2par(est, model)

    armaFitted(x, model, estPar)
end

# Function for forecasting
# Note: If we log-transform the time series, we should to a sample
#       based forecasts.
function predict(fit::armaFitted,
                 h::Int64,
                 quant::Vector{Float64})
    #
    α = fit.par.α
    β = fit.par.β

    p = length(α)
    q = length(β)
    M = maximum([p, q])
    
    μ = fit.model.μ
    σ² = fit.par.σ²

    T = length(fit.x)
    tSeq = 1:T+h

    m = (t -> μ(fit.par.μ, t)).(tSeq)
    y = fit.x .- m[1:T]
    
    yHat = zeros(T + h)
    e = zeros(T + h)
    for t = M+1:T
        yHat[t] = sum(α .* y[t-1:-1:t-p]) .+ sum(β .* (y[t-1:-1:t-q] .- yHat[t-1:-1:t-q]))
        e[t] = y[t] - yHat[t]
    end

    out = zeros(T + h)
    out[1:T] = y
    for t = T+1:T+h
        out[t] = sum(α .* out[t-1:-1:t-p]) + sum(β .* e[t-1:-1:t-p])
    end
    
    out = out .+ m
    
    if h > 1
        MAcoef = ARMAtoMA(α, β, h-1)
        MAcoef = [1; MAcoef]
    else
        MAcoef = [1]
    end
    predVar = σ² .* cumsum(MAcoef.^2)
    dVec = Normal.(out[end-h+1:end], sqrt.(predVar))
    Q = zeros(length(quant), h)
    for hh = 1:h
        Q[:, hh] = quantile.(dVec[hh], quant)
    end

    if fit.model.isPos
        Q[Q .< 0] .= 0
    end
    
    return forecast(Q, quant, collect(1:h))
end