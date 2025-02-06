# Data type to store forecasts and observed values

mutable struct forecastInterval
    α::Vector{T1} where {T1 <: Real}
    l::Vector{T2} where {T2 <: Real}
    u::Vector{T3} where {T3 <: Real}
    function forecastInterval(α, l, u)
        if !(typeof(α) <: Vector)
            α = [α]
        end
        if !(typeof(l) <: Vector)
            l = [l]
        end
        if !(typeof(u) <: Vector)
            u = [u]
        end
        if length(unique([length(α), length(l), length(u)])) != 1
            error("Invalid dimensions")
        end
        if any(.!(0 .<= α .<= 1))
            error("Invalid level") 
        end
        if any(l .> u)
            error("Invalid interval")
        end
        o = sortperm(α)
        new(α[o], l[o], u[o])
    end
end

mutable struct forecast
    horizon::Vector{Int64}
    mean::Vector{T1} where {T1 <: Real}
    median::Vector{T2} where {T2 <: Real}
    interval::Vector{forecastInterval}
    truth::Vector{T3} where {T3 <: Real}
    trajectory::Matrix{T4} where {T4 <: Real}
    
    function forecast(horizon = Int64[];
        mean = Float64[],
        median = Float64[],
        interval = forecastInterval[],
        truth = Float64[],
        trajectory = zeros(0, 0))

        horizon = Int64.(horizon)
        if typeof(horizon) == Int64
            horizon = [horizon]
        end

        nh = length(horizon)

        if typeof(interval) == forecastInterval
            interval = [interval]
        end
        if (length(interval) != nh) & (length(interval) > 0)
            error("Forecast intervals do not match forecast horizon.")
        end

        if (length(truth) != nh) & (length(truth) > 0)
            error("Truth data does not match forecast horizons.")
        end

        if (length(mean) != nh) & (length(mean) > 0)
            error("Point forecast (mean) does not match forecast horizons.")
        end

        if (length(median) != nh) & (length(median) > 0)
            error("Median forecast does not match forecast horizons.")
        end

        if (length(trajectory) > 0) & (size(trajectory)[2] != nh)
            if size(trajectory[1] == nh)
                trajectory = t(trajectory)
            else
                error("Trajectory dimensions do not match forecast horizons.")
            end
        end
        new(horizon, mean, median, interval, truth, trajectory)
    end
end


function addTruth(x::forecast, truth::Vector{T1}) where {T1 <: Real}
    forecast(x.horizon, mean = x.mean, median = x.median, interval = x.interval, truth = truth, trajectory = x.trajectory)
end

function addTrajectory(x::forecast, trajectory::Matrix{T1}) where {T1 <: Real}
    forecast(x.horizon, mean = x.mean, median = x.median, interval = x.interval, truth = x.truth, trajectory = trajectory)
end

function addMean(x::forecast, trajectory::Vector{T1}) where {T1 <: Real}
    forecast(x.horizon, mean = mean, median = x.median, interval = x.interval, truth = x.truth, trajectory = x.trajectory)
end

function addMedian(x::forecast, trajectory::Vector{T1}) where {T1 <: Real}
    forecast(x.horizon, mean = x.mean, median = median, interval = x.interval, truth = x.truth, trajectory = x.trajectory)
end


function getQuantiles(x::forecast)
    (i -> i.α).(x.interval)
end

# Translating between intervals and matrix
function getQmat(x::forecast)
    quants = getQuantiles(x)
    if length(unique(quants)) > 1
        error("Quantiles of intervals do not match")
    end
    nq = length(quants[1])
    hasMedian = length(x.median) > 0
    if (length(x.interval) == 0) & !hasMedian
        error("No interval or median forecasts found")
    end
    out = zeros(2*nq + hasMedian, length(x.horizon))
    for i in 1:length(x.interval)
        out[1:nq, i] = x.interval[i].l
        out[end - nq + 1:end, i] = reverse(x.interval[i].u)
    end
    if hasMedian
        out[nq + 1, :] = x.median
    end
    return out
end

