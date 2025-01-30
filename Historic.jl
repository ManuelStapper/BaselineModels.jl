##########################
### Last Similar Dates ###
##########################

# Model architecture:
# - S: Periodicity
# - w: Window size: For t use for forecast t - n*S ± w for integer n

struct lsdModel <: Baseline
    S::Int64
    w::Int64
    isPos::Bool
    function lsdModel(S::Int64 = 1, w::Int64 = 0, isPos::Bool = true)
        if (w < 0) | (w >= S)
            error("Invalid window size")
        end
        if S < 1
            error("Invalid periodicity")
        end

        new(S, w, true)
    end
end

mutable struct lsdParameter
    μ::Vector{T} where {T <: Real}
end

mutable struct lsdFitted
    x::Vector{T} where {T <: Real}
    model::lsdModel
    par::lsdParameter
end

function trunc(x::Vector{Int64}, T::Int64)::Vector{Int64}
    x[0 .< x .<= T]
end

function fit(x::Vector{T1}, model::lsdModel) where {T1 <: Real}
    w = model.w
    S = model.S
    T = length(x)

    # Create a set of indices for s = 1 and then shift along the time series
    # Should first contain negative indices (up to -S+1)
    cen = collect(1:S:T+w)
    ind = vcat((cc -> cc-w:cc+w).(cen)...)
    μ = (s -> mean(x[trunc(ind .+ s, T)])).(0:S-1)
    
    lsdFitted(x, model, lsdParameter(μ))
end

# Functions to compute leave one out forecasts
# Returns the forecast error
function dropMissingMean(x::Vector{Union{T1, Missing}}) where {T1 <: Real}
    mean(x[.!ismissing.(x)])
end

# x:     Original time series
# t:     Time index of observation that is left out
# model: Model
function fitLOO(x::Vector{T1}, t::Int64, model::lsdModel) where {T1 <: Real}
    w = model.w
    S = model.S
    T = length(x)

    xx = [x[1:t-1]; missing; x[t+1:T]]

    # Create a set of indices for s = 1 and then shift along the time series
    # Should first contain negative indices (up to -S+1)
    cen = collect(1:S:T+w)
    ind = vcat((cc -> cc-w:cc+w).(cen)...)

    sOut = mod(t, S)
    sOut = ifelse(sOut == 0, S, sOut)

    μ = (s -> dropMissingMean(xx[trunc(ind .+ s, T)]))(sOut - 1)
    
    return x[t] - μ
end

function predict(fit::lsdFitted,
                 h::Int64,
                 quant::Vector{Float64})
    #
    x = fit.x
    T = length(x)
    S = fit.model.S
    w = fit.model.w

    μ = repeat(fit.par.μ, ceil(Int64, (T + h)/S))[1:T+h]
    
    # Point forecast
    out = μ[T+1:T+h]

    # Forecast intervals do not really depend on horizon
    # More important is the time
    # Compute intervals by leave one out
    ϵ = zeros(T)
    
    for t = 1:T
        s = mod(t, S)
        s = ifelse(s == 0, S, s)
        ϵ[t] = fitLOO(x, t, fit.model)
    end

    cen = collect(1:S:T+w)
    ind = vcat((cc -> cc-w:cc+w).(cen)...)

    

    Q = zeros(length(quant), h)
    for hh = 1:h
        s = mod(T + hh, S)
        s = ifelse(s == 0, S, s)
        ϵh = ϵ[trunc(ind .+ s .- 1, T)]
        ϵh = [ϵ[h]; -ϵh]

        Q[:, hh] = out[hh] .+ quantile(ϵh, quant)
    end

    if fit.model.isPos
        Q[Q .< 0] .= 0
    end

    return forecast(Q, quant, collect(1:h))
end