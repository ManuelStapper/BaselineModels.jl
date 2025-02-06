###########
### STL ###
###########

# "Model" specified by the smoothing coefficients
struct stlModel <: Baseline
    p::Int64 # Periodicity
    i::Int64 # Nr of inner loops
    o::Int64 # Nr of outer loops
    l::Int64 # Low pass filter coefficient
    t::Int64 # Smoothing coef for trend
    s::Int64 # Smoothing coef for seasonality
    isPos::Bool
end

# Constructor with default values
function stlModel(;p::Int64 = 1,
                  i::Int64 = 1,
                  o::Int64 = 5,
                  l::Int64 = -1,
                  t::Int64 = -1,
                  s::Int64 = 7,
                  isPos::Bool = true)
    #
    if l == -1
        l = p + iseven(p)
    end
    if t == -1
        t = ceil(Int64, 3/2*p/(1 - 3/(2*s)))
        t = t + iseven(t)
        if t < 5
            t = 5
        end
    end
    return stlModel(p, i, o, l, t, s, isPos)
end

mutable struct stlParameter
    S::Vector{T1} where {T1 <: Real} # Season
    T::Vector{T1} where {T1 <: Real} # Trend
    R::Vector{T1} where {T1 <: Real} # Remainder
end

# LOESS regression. Should return one value before and one after observed
# Input:
# y time series to be smoothed
# q window coefficient
# w weights

function LOESS(y::Vector{T1}, q::Int64, w::Vector{T2}) where {T1, T2 <: Real}
    T = length(y)
    out = zeros(T+2) # Include buffer for start & end
    # Double check if that should be more?
    m = Int64((q+1)/2) # Nr of observations to each side

    # Start (Asymmetric)
    # For the first observations, always use the first q observations
    for s = 0:m
        tSeq = 1:q
        yy = y[tSeq]
        # Time distances
        λ = abs.(tSeq .- s)
        # Compute weights and normalise
        u = λ./maximum(λ)
        ww = (1 .- u.^3).^3
        ww = ww .* w[tSeq]
        ww = ww ./ sum(ww)

        # Weighted regression
        tBar = ww'tSeq
        yBar = ww'yy
        bHat = (ww'*((tSeq .- tBar) .* (yy .- yBar))) / (ww'*(tSeq .- tBar).^2)
        aHat = yBar .- bHat*tBar
        out[s+1] = aHat + bHat*s
    end
    # Middle
    for s = m+1:T-m
        tSeq = s-m:s+m
        yy = y[tSeq]
        λ = abs.(tSeq .- s)
        u = λ./maximum(λ)
        ww = (1 .- u.^3).^3
        ww = ww .* w[tSeq]
        ww = ww ./ sum(ww)
        out[s+1] = sum(yy .* ww)
    end
    # End
    for s = T-m+1:T+1
        tSeq = T-q+1:T
        yy = y[tSeq]
        λ = abs.(tSeq .- s)
        u = λ./maximum(λ)
        ww = (1 .- u.^3).^3
        ww = ww .* w[tSeq]
        ww = ww ./ sum(ww)
        tBar = ww'tSeq
        yBar = ww'yy
        bHat = (ww'*((tSeq .- tBar) .* (yy .- yBar))) / (ww'*(tSeq .- tBar).^2)
        aHat = yBar .- bHat*tBar
        out[s+1] = aHat + bHat*s
    end
    out
end

# Simple moving average
function movingAverage(y::Vector{T1}, q::Int64) where {T1 <: Real}
    (i -> mean(y[(1:q) .+ i])).(0:length(y) - q)
end

mutable struct stlFitted
    x::Vector{T} where {T <: Real}
    model::stlModel
    par::stlParameter
end

# Does not really work well in case of no seasonality
# Adapt for p < 2

function fit(x::Vector{T1}, model::stlModel) where {T1 <: Real}
    nObs = length(x)
    T = zeros(nObs)
    S = zeros(nObs)
    R = zeros(nObs)
    L = zeros(nObs)
    w = ones(nObs)

    xx = x

    # There seems to be an NA problem here?!
    # Only in case of no log transformation
    # Weights can be zero -> add min weight to avoid div by zero
    for iOuter = 1:model.o
        for iInner = 1:model.i
            # Step 1: De-trending
            dt = xx .- T
            # Step 2: Cycle-subseries smoothing
            C = zeros(nObs + 2*model.p)
            if model.p >= 2
                for s = 1:model.p
                    yy = dt[s:model.p:end]
                    ind = (s - model.p:model.p:length(dt) + model.p) .+ model.p
                    C[ind] = LOESS(yy, model.s, w[s:model.p:end])
                end
                # Step 3: Low pass filter
                L = movingAverage(C, model.p)
                L = movingAverage(L, model.p)
                L = movingAverage(L, 3)
                L = LOESS(L, model.l, ones(length(L)))[2:end-1]
            end
            
            # Step 4: De-trending smoothed cycle-subseries
            S = C[model.p+1:model.p+nObs] .- L
            # Step 5: De-seasonalising
            ds = xx .- S
            # Step 6: Trend-smoothing
            T = LOESS(ds, model.t, w)[2:end-1]
        end
        # If there are multiple outer iterations:
        R = xx .- T .- S
        if model.o > 1
            h = 6*median(abs.(R))
            u = abs.(R)./h
            w = (1 .- u.^2).^2
            w[u .> 1] .= 0
            w[w .== 0] .= minimum(w[w .> 0])
        end
    end
    stlFitted(xx, model, stlParameter(S, T, R))
end

# Forecasting uses average trend increments for trend component
# Point forecasts by averaging weeks
# Forecast intervals by cumsumming remainders
# Starting point for point forecast is the last observation without remainder
function predict(fit::stlFitted,
                 h::Int64,
                 quant::Vector{Float64} = [0.5],
                 nChains::Int64 = 10000)
    #
    nObs = length(fit.x)
    quant = sort(unique(round.([quant; 1 .- quant; 0.5], digits = 4)))
    init = fit.par.S[end] + fit.par.T[end]
    Tavg = mean(diff(fit.par.T))
    Savg = (i -> mean(fit.par.S[i:fit.model.p:end])).(1:fit.model.p)
    out = init .+ repeat(Savg, ceil(Int64, (nObs + h)/fit.model.p))[nObs .+ (1:h)]
    out .+= Tavg .* (1:h)

    Y = zeros(nChains, h)
    ϵ = fit.par.R
    ϵ = ϵ .- mean(ϵ)
    ϵ = [ϵ; -ϵ]

    for i = 1:nChains
        Y[i, :] = cumsum(sample(ϵ, h)) .+ out
    end
    Q = zeros(length(quant), h)
    for hh = 1:h
       Q[:, hh] = (q -> quantile(Y[:, hh], q)).(quant)
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