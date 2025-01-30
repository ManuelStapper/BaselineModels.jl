# Functions to remove seasonality before fitting
# Goal: Fit a sine-cosine seasonality to avoid neccessity of long training data

mutable struct seasonalityParameter
    θ::Vector{T1} where {T1 <: Real}
    κ::Vector{T2} where {T2 <: Real}
    function seasonalityParameter(θ, κ)
        if !(typeof(θ) <: Vector)
            θ = [θ]
        end
        if !(typeof(κ) <: Vector)
            κ = [κ]
        end
        if length(θ) == length(κ)
            return new(θ, κ)
        else
            error("Invalid dimensions")
        end
    end
end


function getS(pars::seasonalityParameter, m::Int64, T::Int64)
    k = length(pars.θ)
    tSeq = 1:T
    out = zeros(T)
    for i = 1:k
        out = out .+ pars.θ[i] .* sin.((i * 2 * π / m) .* tSeq)
        out = out .+ pars.κ[i] .* cos.((i * 2 * π / m) .* tSeq)
    end
    out
end

function fitS(x::Vector{T1}, m::Int64, k::Int64) where {T1 <: Real}
    T = length(x)
    tSeq = 1:T
    X = zeros(T, 2*k)
    for i = 1:k
        X[:, i] = sin.((i * 2 * π / m) .* tSeq)
        X[:, k+i] = cos.((i * 2 * π / m) .* tSeq)
    end
    X = [X ones(length(x))]
    β = inv(X'X)*X'x
    return seasonalityParameter(β[1:k], β[k+1:end - 1])
end

# Function that takes a time series and returns a de-seasonalised time series and the fitted
function preFilter(x::Vector{T1}, m::Int64, k::Int64) where {T1 <: Real}
    res = fitS(log.(x .+ 1) .- mean(log.(x .+ 1)), m, k)
    μ = getS(res, m, length(x))
    μ = exp.(μ .+ mean(log.(x .+ 1)))
    return x ./ μ, μ, res
end


function postFilter(x::Vector{T1}, fc::forecast, m::Int64, est::seasonalityParameter, xTruth::Vector{T2}) where {T1, T2 <: Real}
    h = maximum(fc.horizon)
    μNew = getS(est, m, length(x) + h)[end-h+1:end]
    μNew = μNew[fc.horizon]
    mlx = mean(log.(x .+ 1))
    for i = 1:size(fc.Qmat)[1]
        fc.Qmat[i, :] = fc.Qmat[i, :] .* exp.((μNew .+ mlx))
    end
    fc.truth = xTruth
    return fc
end