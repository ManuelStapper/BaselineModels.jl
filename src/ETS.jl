################################
### Error-Trend-Season model ###
################################

# Notaion:
# x_t = wfun(z_{t-1}) + r(x_{t-1})ϵ_t
# z_t = f(z_{t-1}) + g(x_{t-1})ϵ_t

# Hidden states contain level-, trend- and season-component
# Error terms are assumed white noise
# Errors can be Additive ("A") or Multiplicative ("M")
# Seasonal component can be either None ("N"), Additive ("A") or Multiplicative ("M")
# Trend component can be None ("N"), Additive ("A") or Multiplicative ("M") or
# for the latter two a damped version ("Ad" and "Md")

# A maximum of four tuning coefficients control the smoothness of the approach
# Tuning constants and the initial states z_0 are estimated, the rest is simple filtering

# For details, see "forecast" package in R

# Additionally include pre-filtering of time series to remove seasonality
# Assumes sine-cosine seasonality structure
# That way, the approach is applicable with short training data


abstract type errorType end
abstract type trendType end
abstract type seasonType end

struct Aerror <: errorType end
struct Merror <: errorType end

struct Ntrend <: trendType end
struct Atrend <: trendType end
struct Adtrend <: trendType end
struct Mtrend <: trendType end
struct Mdtrend <: trendType end

struct Nseason <: seasonType end
struct Aseason <: seasonType
    m::Int64 # Periodicity
end
struct Mseason <: seasonType
    m::Int64 # Periodicity
end

struct etsModel{T1, T2, T3} <: Baseline where {T1 <: errorType, T2 <: trendType, T3 <: seasonType}
    error::T1
    trend::T2
    season::T3
end

function etsModel(;error::String = "A",
                  trend::String = "N",
                  season::String = "N",
                  m::Int64 = 1)
    if error == "N"
        et = Merror()
    else
        et = Aerror()
    end

    if trend == "A"
        tt = Atrend()
    elseif trend == "Ad"
        tt = Adtrend()
    elseif trend == "M"
        tt = Mtrend()
    elseif trend == "Md"
        tt = Mdtrend()
    else
        tt = Ntrend()
    end

    if season == "A"
        st = Aseason(m)
    elseif season == "M"
        st = Mseason(m)
    else
        st = Nseason()
    end
    return etsModel(et, tt, st)
end

# Notation for example x_t
struct space
    l::Float64          # l_t
    b::Float64          # b_t
    s::Vector{Float64}  # s_t, s_{t-1}, ..., s_{t-m+1}
end

mutable struct etsTuning
    α::Float64
    β::Float64
    γ::Float64
    ϕ::Float64
end

struct etsParameter
    θ::etsTuning
    z0::space
end

# 30 different setting combinations, all need:
# - w, r, f, and g functions
# - Filtering functions (can then be general?)
# - later: foreacsting and interval function

function etsθ2par(θ::Vector{T}, model::etsModel{<: errorType, Ntrend, Nseason}) where {T <: Real}
    etsParameter(etsTuning(θ[1], 0, 0, 0), space(θ[2], 0, [0]))
end

function etsθ2par(θ::Vector{T}, model::etsModel{<: errorType, Ntrend, <: Union{Aseason, Mseason}}) where {T <: Real}
    etsParameter(etsTuning(θ[1], 0, θ[2], 0), space(θ[3], 0, θ[4:3 + model.season.m]))
end

function etsθ2par(θ::Vector{T}, model::etsModel{<: errorType, <: Union{Atrend, Mtrend}, Nseason}) where {T <: Real}
    etsParameter(etsTuning(θ[1], θ[2], 0, 0), space(θ[3], θ[4], [0]))
end

function etsθ2par(θ::Vector{T}, model::etsModel{<: errorType, <: Union{Atrend, Mtrend}, <: Union{Aseason, Mseason}}) where {T <: Real}
    etsParameter(etsTuning(θ[1], θ[2], θ[3], 0), space(θ[4], θ[5], θ[6:5 + model.season.m]))
end

function etsθ2par(θ::Vector{T}, model::etsModel{<: errorType, <: Union{Adtrend, Mdtrend}, Nseason}) where {T <: Real}
    etsParameter(etsTuning(θ[1], θ[2], 0, θ[3]), space(θ[4], θ[5], [0]))
end

function etsθ2par(θ::Vector{T}, model::etsModel{<: errorType, <: Union{Adtrend, Mdtrend}, <: Union{Aseason, Mseason}}) where {T <: Real}
    etsParameter(etsTuning(θ[1], θ[2], θ[3], θ[4]), space(θ[5], θ[6], θ[7:6 + model.season.m]))
end

function etspar2θ(par::etsParameter, model::etsModel{<: errorType, Ntrend, Nseason})
    [par.θ.α; par.z0.l]
end

function etspar2θ(par::etsParameter, model::etsModel{<: errorType, Ntrend, <: Union{Aseason, Mseason}})
    [par.θ.α; par.θ.β; par.z0.l; par.z0.s]
end

function etspar2θ(par::etsParameter, model::etsModel{<: errorType, <: Union{Atrend, Mtrend}, Nseason})
    [par.θ.α; par.θ.β; par.z0.l; par.z0.b]
end

function etspar2θ(par::etsParameter, model::etsModel{<: errorType, <: Union{Atrend, Mtrend}, <: Union{Aseason, Mseason}})
    [par.θ.α; par.θ.β; par.θ.γ; par.z0.l; par.z0.b; par.z0.s]
end

function etspar2θ(par::etsParameter, model::etsModel{<: errorType, <: Union{Adtrend, Mdtrend}, Nseason})
    [par.θ.α; par.θ.β; par.θ.ϕ; par.z0.l; par.z0.b]
end

function etspar2θ(par::etsParameter, model::etsModel{<: errorType, <: Union{Adtrend, Mdtrend}, <: Union{Aseason, Mseason}})
    [par.θ.α; par.θ.β; par.θ.γ; par.θ.ϕ; par.z0.l; par.z0.b; par.z0.s]
end


# A-N-N
function wfun(z::space, model::etsModel{Aerror, Ntrend, Nseason}, coef::etsTuning)
    z.l
end

function f(z::space, model::etsModel{Aerror, Ntrend, Nseason}, coef::etsTuning)
    return z
end

function g(z::space, model::etsModel{Aerror, Ntrend, Nseason}, coef::etsTuning)
    space(coef.α, 0.0, [0.0])
end

# A-N-A
function wfun(z::space, model::etsModel{Aerror, Ntrend, Aseason}, coef::etsTuning)
    z.l + z.s[end]
end

function f(z::space, model::etsModel{Aerror, Ntrend, Aseason}, coef::etsTuning)
    space(z.l, 0.0, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Ntrend, Aseason}, coef::etsTuning)
    space(coef.α, 0.0, [coef.γ; zeros(length(z.s) - 1)])
end

# A-N-M
function wfun(z::space, model::etsModel{Aerror, Ntrend, Mseason}, coef::etsTuning)
    z.l * z.s[end]
end

function f(z::space, model::etsModel{Aerror, Ntrend, Mseason}, coef::etsTuning)
    space(z.l, 0.0, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Ntrend, Mseason}, coef::etsTuning)
    space(coef.α / z.s[end], 0.0, [coef.γ / z.l; zeros(length(z.s) - 1)])
end

# A-A-N
function wfun(z::space, model::etsModel{Aerror, Atrend, Nseason}, coef::etsTuning)
    z.l + z.b
end

function f(z::space, model::etsModel{Aerror, Atrend, Nseason}, coef::etsTuning)
    space(z.l + z.b, z.b, [0.0])
end

function g(z::space, model::etsModel{Aerror, Atrend, Nseason}, coef::etsTuning)
    space(coef.α, coef.β, [0.0])
end

# A-A-A
function wfun(z::space, model::etsModel{Aerror, Atrend, Aseason}, coef::etsTuning)
    z.l + z.b + z.s[end]
end

function f(z::space, model::etsModel{Aerror, Atrend, Aseason}, coef::etsTuning)
    space(z.l + z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Atrend, Aseason}, coef::etsTuning)
    space(coef.α, coef.β, [coef.γ; zeros(length(z.s) - 1)])
end

# A-A-M
function wfun(z::space, model::etsModel{Aerror, Atrend, Mseason}, coef::etsTuning)
    (z.l + z.b) * z.s[end]
end

function f(z::space, model::etsModel{Aerror, Atrend, Mseason}, coef::etsTuning)
    space(z.l + z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Atrend, Mseason}, coef::etsTuning)
    space(coef.α/z.s[end], coef.β/z.s[end], [coef.γ / (z.l + z.b); zeros(length(z.s) - 1)])
end

# A-Ad-N
function wfun(z::space, model::etsModel{Aerror, Adtrend, Nseason}, coef::etsTuning)
    z.l + coef.ϕ * z.b
end

function f(z::space, model::etsModel{Aerror, Adtrend, Nseason}, coef::etsTuning)
    space(z.l + coef.ϕ * z.b, coef.ϕ * z.b, [0.0])
end

function g(z::space, model::etsModel{Aerror, Adtrend, Nseason}, coef::etsTuning)
    space(coef.α, coef.β, [0.0])
end

# A-Ad-A
function wfun(z::space, model::etsModel{Aerror, Adtrend, Aseason}, coef::etsTuning)
    z.l + coef.ϕ * z.b + z.s[end]
end

function f(z::space, model::etsModel{Aerror, Adtrend, Aseason}, coef::etsTuning)
    space(z.l + coef.ϕ * z.b, coef.ϕ * z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Adtrend, Aseason}, coef::etsTuning)
    space(coef.α, coef.β, [coef.γ; zeros(length(z.s) - 1)])
end

# A-Ad-M
function wfun(z::space, model::etsModel{Aerror, Adtrend, Mseason}, coef::etsTuning)
    (z.l + coef.ϕ * z.b) * z.s[end]
end

function f(z::space, model::etsModel{Aerror, Adtrend, Mseason}, coef::etsTuning)
    space(z.l + coef.ϕ * z.b, coef.ϕ * z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Adtrend, Mseason}, coef::etsTuning)
    space(coef.α/z.s[end], coef.β/z.s[end], [coef.γ / (z.l + coef.ϕ * z.b); zeros(length(z.s) - 1)])
end

# A-M-N
function wfun(z::space, model::etsModel{Aerror, Mtrend, Nseason}, coef::etsTuning)
    z.l * z.b
end

function f(z::space, model::etsModel{Aerror, Mtrend, Nseason}, coef::etsTuning)
    space(z.l * z.b, z.b, [0.0])
end

function g(z::space, model::etsModel{Aerror, Mtrend, Nseason}, coef::etsTuning)
    space(coef.α, coef.β / z.l, [0.0])
end

# A-M-A
function wfun(z::space, model::etsModel{Aerror, Mtrend, Aseason}, coef::etsTuning)
    z.l * z.b + z.s[end]
end

function f(z::space, model::etsModel{Aerror, Mtrend, Aseason}, coef::etsTuning)
    space(z.l * z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Mtrend, Aseason}, coef::etsTuning)
    space(coef.α, coef.β / z.l, [coef.γ; zeros(length(z.s) - 1)])
end

# A-M-M
function wfun(z::space, model::etsModel{Aerror, Mtrend, Mseason}, coef::etsTuning)
    z.l * z.b * z.s[end]
end

function f(z::space, model::etsModel{Aerror, Mtrend, Mseason}, coef::etsTuning)
    space(z.l * z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Mtrend, Mseason}, coef::etsTuning)
    space(coef.α/z.s[end], coef.β/(z.s[end]*z.l), [coef.γ / (z.l * z.b); zeros(length(z.s) - 1)])
end

# A-Md-N
function wfun(z::space, model::etsModel{Aerror, Mdtrend, Nseason}, coef::etsTuning)
    z.l * z.b ^ coef.ϕ
end

function f(z::space, model::etsModel{Aerror, Mdtrend, Nseason}, coef::etsTuning)
    space(z.l * z.b ^ coef.ϕ, z.b ^ coef.ϕ, [0.0])
end

function g(z::space, model::etsModel{Aerror, Mdtrend, Nseason}, coef::etsTuning)
    space(coef.α, coef.β/z.l, [0.0])
end

# A-Md-A
function wfun(z::space, model::etsModel{Aerror, Mdtrend, Aseason}, coef::etsTuning)
    z.l * z.b ^ coef.ϕ + z.s[end]
end

function f(z::space, model::etsModel{Aerror, Mdtrend, Aseason}, coef::etsTuning)
    space(z.l * z.b ^ coef.ϕ, z.b ^ coef.ϕ, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Mdtrend, Aseason}, coef::etsTuning)
    space(coef.α, coef.β / z.l, [coef.γ; zeros(length(z.s) - 1)])
end

# A-Md-M
function wfun(z::space, model::etsModel{Aerror, Mdtrend, Mseason}, coef::etsTuning)
    (z.l * z.b ^ coef.ϕ) * z.s[end]
end

function f(z::space, model::etsModel{Aerror, Mdtrend, Mseason}, coef::etsTuning)
    space(z.l * z.b ^ coef.ϕ, z.b ^ coef.ϕ, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Aerror, Mdtrend, Mseason}, coef::etsTuning)
    space(coef.α/z.s[end], coef.β/(z.s[end] * z.l), [coef.γ / (z.l * z.b^coef.ϕ); zeros(length(z.s) - 1)])
end

### Multiplicative errors

# M-N-N
function wfun(z::space, model::etsModel{Merror, Ntrend, Nseason}, coef::etsTuning)
    z.l
end

function f(z::space, model::etsModel{Merror, Ntrend, Nseason}, coef::etsTuning)
    space(z.l, 0.0, [0.0])
end

function g(z::space, model::etsModel{Merror, Ntrend, Nseason}, coef::etsTuning)
    space(coef.α * z.l, 0.0, [0.0])
end

# M-N-A
function wfun(z::space, model::etsModel{Merror, Ntrend, Aseason}, coef::etsTuning)
    z.l + z.s[end]
end

function f(z::space, model::etsModel{Merror, Ntrend, Aseason}, coef::etsTuning)
    space(z.l, 0.0, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Ntrend, Aseason}, coef::etsTuning)
    space(coef.α * (z.l + z.s[end]), 0.0, [coef.γ * (z.l + z.s[end]); zeros(length(z.s) - 1)])
end

# M-N-M
function wfun(z::space, model::etsModel{Merror, Ntrend, Mseason}, coef::etsTuning)
    z.l * z.s[end]
end

function f(z::space, model::etsModel{Merror, Ntrend, Mseason}, coef::etsTuning)
    space(z.l, 0.0, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Ntrend, Mseason}, coef::etsTuning)
    space(z.l * coef.α, 0.0, [coef.γ * z.s[end]; zeros(length(z.s) - 1)])
end

# M-A-N
function wfun(z::space, model::etsModel{Merror, Atrend, Nseason}, coef::etsTuning)
    z.l + z.b
end

function f(z::space, model::etsModel{Merror, Atrend, Nseason}, coef::etsTuning)
    space(z.l + z.b, z.b, [0.0])
end

function g(z::space, model::etsModel{Merror, Atrend, Nseason}, coef::etsTuning)
    space(coef.α * (z.l + z.b), coef.β * (z.l + z.b), [0.0])
end

# M-A-A
function wfun(z::space, model::etsModel{Merror, Atrend, Aseason}, coef::etsTuning)
    z.l + z.b + z.s[end]
end

function f(z::space, model::etsModel{Merror, Atrend, Aseason}, coef::etsTuning)
    space(z.l + z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Atrend, Aseason}, coef::etsTuning)
    scl = (z.l + z.b + z.s[end])
    space(coef.α * scl, coef.β * scl, [coef.γ * scl; zeros(length(z.s) - 1)])
end

# M-A-M
function wfun(z::space, model::etsModel{Merror, Atrend, Mseason}, coef::etsTuning)
    (z.l + z.b) * z.s[end]
end

function f(z::space, model::etsModel{Merror, Atrend, Mseason}, coef::etsTuning)
    space(z.l + z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Atrend, Mseason}, coef::etsTuning)
    scl = z.l + z.b
    space(coef.α * scl, coef.β * scl, [coef.γ * z.s[end]; zeros(length(z.s) - 1)])
end

# M-Ad-N
function wfun(z::space, model::etsModel{Merror, Adtrend, Nseason}, coef::etsTuning)
    z.l + coef.ϕ * z.b
end

function f(z::space, model::etsModel{Merror, Adtrend, Nseason}, coef::etsTuning)
    space(z.l + coef.ϕ * z.b, coef.ϕ * z.b, [0.0])
end

function g(z::space, model::etsModel{Merror, Adtrend, Nseason}, coef::etsTuning)
    scl = z.l + coef.ϕ * z.b
    space(coef.α * scl, coef.β * scl, [0.0])
end

# M-Ad-A
function wfun(z::space, model::etsModel{Merror, Adtrend, Aseason}, coef::etsTuning)
    z.l + coef.ϕ * z.b + z.s[end]
end

function f(z::space, model::etsModel{Merror, Adtrend, Aseason}, coef::etsTuning)
    space(z.l + coef.ϕ * z.b, coef.ϕ * z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Adtrend, Aseason}, coef::etsTuning)
    scl = z.l + coef.ϕ * z.b + z.s[end]
    space(coef.α * scl, coef.β * scl, [coef.γ * scl; zeros(length(z.s) - 1)])
end

# M-Ad-M
function wfun(z::space, model::etsModel{Merror, Adtrend, Mseason}, coef::etsTuning)
    (z.l + coef.ϕ * z.b) * z.s[end]
end

function f(z::space, model::etsModel{Merror, Adtrend, Mseason}, coef::etsTuning)
    space(z.l + coef.ϕ * z.b, coef.ϕ * z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Adtrend, Mseason}, coef::etsTuning)
    scl = z.l + coef.ϕ * z.b
    space(coef.α * scl, coef.β * scl, [coef.γ * z.s[end]; zeros(length(z.s) - 1)])
end

# M-M-N
function wfun(z::space, model::etsModel{Merror, Mtrend, Nseason}, coef::etsTuning)
    z.l * z.b
end

function f(z::space, model::etsModel{Merror, Mtrend, Nseason}, coef::etsTuning)
    space(z.l * z.b, z.b, [0.0])
end

function g(z::space, model::etsModel{Merror, Mtrend, Nseason}, coef::etsTuning)
    space(coef.α * z.l * z.b, coef.β * z.b, [0.0])
end

# M-M-A
function wfun(z::space, model::etsModel{Merror, Mtrend, Aseason}, coef::etsTuning)
    z.l * z.b + z.s[end]
end

function f(z::space, model::etsModel{Merror, Mtrend, Aseason}, coef::etsTuning)
    space(z.l * z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Mtrend, Aseason}, coef::etsTuning)
    scl = z.l * z.b + z.s[end]
    space(coef.α * scl, coef.β * scl / z.l, [coef.γ * scl; zeros(length(z.s) - 1)])
end

# M-M-M
function wfun(z::space, model::etsModel{Merror, Mtrend, Mseason}, coef::etsTuning)
    z.l * z.b * z.s[end]
end

function f(z::space, model::etsModel{Merror, Mtrend, Mseason}, coef::etsTuning)
    space(z.l * z.b, z.b, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Mtrend, Mseason}, coef::etsTuning)
    space(coef.α * z.l * z.b, coef.β * z.b, [coef.γ * z.s[end]; zeros(length(z.s) - 1)])
end

# M-Md-N
function wfun(z::space, model::etsModel{Merror, Mdtrend, Nseason}, coef::etsTuning)
    z.l * z.b ^ coef.ϕ
end

function f(z::space, model::etsModel{Merror, Mdtrend, Nseason}, coef::etsTuning)
    space(z.l * z.b ^ coef.ϕ, z.b ^ coef.ϕ, [0.0])
end

function g(z::space, model::etsModel{Merror, Mdtrend, Nseason}, coef::etsTuning)
    space(coef.α * z.l * z.b ^ coef.ϕ, coef.β * z.b ^ coef.ϕ, [0.0])
end

# M-Md-A
function wfun(z::space, model::etsModel{Merror, Mdtrend, Aseason}, coef::etsTuning)
    z.l * z.b ^ coef.ϕ + z.s[end]
end

function f(z::space, model::etsModel{Merror, Mdtrend, Aseason}, coef::etsTuning)
    space(z.l * z.b ^ coef.ϕ, z.b ^ coef.ϕ, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Mdtrend, Aseason}, coef::etsTuning)
    scl = z.l * z.b ^ coef.ϕ + z.s[end]
    space(coef.α * scl, coef.β * scl / z.l, [coef.γ * scl; zeros(length(z.s) - 1)])
end

# M-Md-M
function wfun(z::space, model::etsModel{Merror, Mdtrend, Mseason}, coef::etsTuning)
    (z.l * z.b ^ coef.ϕ) * z.s[end]
end

function f(z::space, model::etsModel{Merror, Mdtrend, Mseason}, coef::etsTuning)
    space(z.l * z.b ^ coef.ϕ, z.b ^ coef.ϕ, [z.s[end]; z.s[1:end-1]])
end

function g(z::space, model::etsModel{Merror, Mdtrend, Mseason}, coef::etsTuning)
    space(coef.α * z.l * z.b ^ coef.ϕ, coef.β * z.b ^ coef.ϕ, [coef.γ * z.s[end]; zeros(length(z.s) - 1)])
end

function r(z::space, model::etsModel{Aerror, <: Union{Ntrend, Atrend, Mtrend, Adtrend, Mdtrend}, <: Union{Nseason, Aseason, Mseason}}, coef::etsTuning)
    1.0
end

function r(z::space, model::etsModel{Merror, <: Union{Ntrend, Atrend, Mtrend, Adtrend, Mdtrend}, <: Union{Nseason, Aseason, Mseason}}, coef::etsTuning)
    wfun(z, model, coef)
end


# Arithmetics for "space" objects
import Base.+
+(s1::space, s2::space) = begin
    space(s1.l + s2.l, s1.b + s2.b, s1.s .+ s2.s)
end

import Base.*
*(s1::space, s2::space) = begin
    space(s1.l * s2.l, s1.b * s2.b, s1.s .* s2.s)
end

*(s1::space, s2::T) where {T <: Real} = begin
    space(s1.l * s2, s1.b * s2, s1.s .* s2)
end

mutable struct etsFiltered
    z0::space
    z::Vector{space}
    xHat::Vector{T} where {T <: Real}
    ϵ::Vector{T} where {T <: Real}
end

# Function to carry out filtering
function etsFilter(x::Vector{T1}, model::etsModel, par::etsParameter) where {T1 <: Real}
    T = length(x)
    xHat = zeros(T)
    ϵ = zeros(T)
    z0 = par.z0
    z = Vector{typeof(z0)}(undef, T)

    xHat[1] = wfun(par.z0, model, par.θ)
    if typeof(model.error) == Aerror
        ϵ[1] = x[1] - xHat[1]
    else
        ϵ[1] = (x[1] - xHat[1])/xHat[1]
    end    
    z[1] = f(z0, model, par.θ) + g(z0, model, par.θ)*ϵ[1]

    for t = 2:T
        xHat[t] = wfun(z[t-1], model, par.θ)
        if typeof(model.error) == Aerror
            ϵ[t] = x[t] - xHat[t]
        else
            ϵ[t] = (x[t] - xHat[t])/xHat[t]
        end    
        z[t] = f(z[t-1], model, par.θ) + g(z[t-1], model, par.θ)*ϵ[t]
    end
    etsFiltered(z0, z, xHat, ϵ)
end

# Getting initial values for z0 and coefficients:
# See article on "forecast" package for details

function getInitial(x::Vector{T1}, model::etsModel) where {T1 <: Real}
    coef = etsTuning(0.1, 0.01, 0.01, 0.99)
    T = length(x)
    if typeof(model.season) <: Union{Aseason, Mseason}
        m = model.season.m
        if length(x) > 2*m
            ff = (t -> mean(x[t:t+2*m-1])).(1:T-2*m+1)
            tSeq = (1:T-2*m+1) .+ m
            if typeof(model.season) == Aseason
                xx = x[tSeq] .- ff
                sRaw = (ss -> mean(xx[ss:m:length(xx)])).(1:m)
                s = sRaw ./ sum(sRaw)
            else
                xx = x[tSeq] ./ ff
                sRaw = (ss -> mean(xx[ss:m:length(xx)])).(1:m)
                s = sRaw ./ sum(sRaw) .* m
            end
        else
            xx = x
            s = abs.(rand(Normal(), m)) ./ 10
        end
    else
        xx = x
        s = [0.0]
    end

    nxx = minimum([10, length(xx)])
    yy = xx[1:nxx]
    X = [ones(nxx) collect(1:nxx)]
    beta = inv(X'X)*X'yy
    l = beta[1]
    if typeof(model.trend) == Atrend
        b = beta[2]
    elseif typeof(model.trend) == Mtrend
        b = 1 + beta[2]/beta[1]
    else
        b = 0.0
    end
    etsParameter(coef, space(l, b, s))
end

# Estimating coefficients and z0:
function L(x::Vector{T1}, model::etsModel, parVec::Vector{T2}) where {T1 <: Real, T2 <: Real}
    par = etsθ2par(parVec, model)
    if !(0 <= par.θ.α <= 1) | !(0 <= par.θ.β <= par.θ.α) | !(0 <= par.θ.γ <= 1 - par.θ.α) | !(0 <= par.θ.ϕ <= 1)
        return Inf
    end

    filtered = etsFilter(x, model, par)
    ϵ = filtered.ϵ
    rr = [r(filtered.z0, model, par.θ); (zz -> r(zz, model, par.θ)).(filtered.z[1:end-1])]
    return length(x)*log(sum(ϵ.^2)) + 2*sum(log.(abs.(rr)))
end

mutable struct etsFitted
    x::Vector{T} where {T <: Real}
    model::etsModel
    par::etsParameter
end

function fit(x::Vector{T1}, model::etsModel) where {T1 <: Real}
    init = getInitial(x, model)
    initVec = etspar2θ(init, model)
    estVec = optimize(vars -> L(x, model, vars), initVec).minimizer
    etsFitted(x, model, etsθ2par(estVec, model))
end

function predict(fit::etsFitted,
                 h::Int64,
                 quant::Vector{Float64},
                 nChains::Int64 = 10000)
    #
    x = fit.x
    T = length(x)
    out = zeros(h)
    filtered = etsFilter(x, fit.model, fit.par)
    quant = sort(unique(round.([quant; 1 .- quant; 0.5], digits = 4)))

    σ = std(filtered.ϵ) + 0.01
    zCurr = filtered.z[end]
    for hh = 1:h
        out[hh] = wfun(zCurr, fit.model, fit.par.θ)
        zCurr = f(zCurr, fit.model, fit.par.θ)
    end

    Y = zeros(nChains, h)
    for i = 1:nChains
        ϵ = rand(Normal(0, σ), h + 1)
        zCurr = filtered.z[end]
        for hh = 1:h
            Y[i, hh] = wfun(zCurr, fit.model, fit.par.θ) + r(zCurr, fit.model, fit.par.θ)*ϵ[hh]
            zCurr = f(zCurr, fit.model, fit.par.θ) + g(zCurr, fit.model, fit.par.θ)*ϵ[hh+1]
        end
    end

    Q = zeros(length(quant), h)
    for i = 1:h
        Q[:, i] = quantile(Y[:, i], quant)
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