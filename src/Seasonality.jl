# Functions to remove seasonality before fitting
# Goal: Fit a sine-cosine seasonality to avoid neccessity of long training data
# Two versions: One for sine-cosine and one for group means
# Should also have trend
"""
    STTransform(;s = 0, k = 1, trend = false, additive = true)

Seasonal-Trend transformation for pre/post-processing time series.

Enables fitting of seasonal patterns and trends to time series data before
applying forecasting models, then adding the patterns back to forecasts.
Uses harmonic (sine-cosine) representation for seasonal components.

# Fields
- `s::Float64`: Seasonal period (0 for no seasonality)
- `k::Int`: Number of harmonic components (default: 1)
- `trend::Bool`: Include linear trend component (default: false)
- `additive::Bool`: Use additive vs. multiplicative decomposition (default: true)

# Mathematical Model
The seasonal-trend component is:
μ(t) = β₀ + trend×(b×t) + ∑ⱼ₌₁ᵏ [θⱼsin(2πjt/s) + κⱼcos(2πjt/s)]

# Decomposition Types
- **Additive**: X_t = Trend_t + Seasonal_t + Remainder_t
- **Multiplicative**: X_t = Trend_t × Seasonal_t × Remainder_t
"""
struct STTransform
    s::Float64
    k::Int
    trend::Bool
    additive::Bool
    function STTransform(;s = 0, k = 1, trend = false, additive = true)
        s isa Real || throw(ArgumentError("Periodicity must be a real number"))
        k isa Int || throw(ArgumentError("Number of sine-cosine waves must be a positive integer"))

        ((s > 0) || (trend)) || throw(ArgumentError("Specify either seasonality or trend"))
        new(s, k, trend, additive)
    end
end

"""
    STParameter(β₀, θ, κ, b)

Parameters for fitted seasonal-trend component.

# Fields
- `β₀::Float64`: Baseline level (intercept)
- `θ::Vector{Float64}`: Sine coefficients [θ₁, θ₂, ..., θₖ]
- `κ::Vector{Float64}`: Cosine coefficients [κ₁, κ₂, ..., κₖ]
- `b::Float64`: Linear trend coefficient (0 if no trend)
"""
struct STParameter
    β0::Float64
    θ::Vector{Float64}
    κ::Vector{Float64}
    b::Float64

    function STParameter(β0, θ, κ, b)
        (b isa Real) || throw(ArgumentError("Trend parameter must be a real number"))
        if !(θ isa Vector)
            θ = [θ]
        end
        if !(κ isa Vector)
            κ = [κ]
        end
        β0 = Float64(β0)
        if length(θ) == length(κ)
            return new(β0, θ, κ, Float64(b))
        else
            throw(ArgumentError("Invalid dimensions"))
        end
    end
end

"""
    getST(pars::STParameter, STsetting::STTransform, T::Int) -> Vector{Float64}

Evaluate seasonal-trend component at specified time points.

# Arguments
- `pars::STParameter`: Fitted seasonal-trend parameters
- `STsetting::STTransform`: Transformation specification
- `T::Int`: Number of time points to evaluate

# Returns
- `Vector{Float64}`: Seasonal-trend values for times 1, 2, ..., T
"""
function getST(pars::STParameter, STsetting::STTransform, T::Int)
    k = STsetting.k
    s = STsetting.s
    tSeq = collect(1:T)

    out = fill(pars.β0, T) .+ pars.b .* tSeq

    for i = 1:k
        out = out .+ pars.θ[i] .* sin.((i * 2 * π / s) .* tSeq)
        out = out .+ pars.κ[i] .* cos.((i * 2 * π / s) .* tSeq)
    end

    out
end

"""
    fitST(x::Vector{T1}, STsetting::STTransform) -> STParameter

Fit seasonal-trend component using least squares regression.

# Arguments
- `x::Vector{T1}`: Time series data where T1 <: Real
- `STsetting::STTransform`: Transformation specification

# Returns
- `STParameter`: Fitted parameters for seasonal-trend component

# Fitting Method
1. **Design Matrix**: Construct X with columns for:
   - Intercept (constant)
   - Sine terms: sin(2πjt/s) for j = 1, ..., k
   - Cosine terms: cos(2πjt/s) for j = 1, ..., k  
   - Linear trend: t (if trend=true)
2. **Least Squares**: Solve β̂ = (X'X)⁻¹X'x
3. **Extract Parameters**: Parse coefficients into STParameter structure
"""
function fitST(x::Vector{T1}, STsetting::STTransform) where {T1 <: Real}
    T = length(x)
    tSeq = 1:T
    k = STsetting.k
    s = STsetting.s

    X = ones(T, 2*STsetting.k .+ STsetting.trend + 1)
    for i = 1:k
        X[:, i + 1] = sin.((i * 2 * π / s) .* tSeq)
        X[:, k+i + 1] = cos.((i * 2 * π / s) .* tSeq)
    end
    if STsetting.trend
        X[:, end] = collect(1:T)
    end
    β = inv(X'X)*X'x

    β0 = β[1]
    if s > 0
        θ = β[2:1 + k]
        κ = β[(2:1 + k) .+ k]
    else
        θ = Float64[]
        κ = Float64[]
    end
    if STsetting.trend
        b = β[end]
    else
        b = 0.0
    end
    STParameter(β0, θ, κ, b)
end

"""
    preFilter(x::Vector{T1}, STsetting::STTransform) -> (Vector{Float64}, STParameter)

Remove seasonal-trend component from time series.

# Arguments
- `x::Vector{T1}`: Original time series where T1 <: Real
- `STsetting::STTransform`: Transformation specification

# Returns
- `Tuple` containing:
  - `Vector{Float64}`: Deseasonalised/detrended series
  - `STParameter`: Fitted seasonal-trend parameters

# Method
1. **Fit Component**: Estimate seasonal-trend parameters using `fitST`
2. **Evaluate Component**: Compute fitted values using `getST`
3. **Remove Component**:
   - **Additive**: filtered = x - fitted
   - **Multiplicative**: filtered = x / fitted

# Data Requirements
- **Multiplicative**: Requires x ≥ 0 (division by seasonal component)
- **Additive**: No restrictions on x values
"""
function preFilter(x::Vector{T1},
    STsetting::STTransform) where {T1 <: Real}
    (STsetting.additive || all(x .>= 0)) || throw(ArgumentError("Only additive filter possible with negative values in time series"))
    
    pars = fitST(x, STsetting)
    fitted = getST(pars, STsetting, length(x))

    if STsetting.additive
        return x .- fitted, pars
    else
        return x ./ fitted, pars
    end
end

"""
    postFilter(x::Vector{T1}, forecast::Forecast, setting::STTransform, pars::STParameter) -> Forecast

Add seasonal-trend component back to forecasts.

# Arguments  
- `x::Vector{T1}`: Original time series (for computing forecast origins)
- `forecast::Forecast`: Forecasts on filtered (deseasonalized) scale
- `setting::STTransform`: Transformation specification (same as preFilter)
- `pars::STParameter`: Fitted parameters (from preFilter)

# Returns
- `Forecast`: Transformed forecast with seasonality/trend restored

# Method
1. **Compute Future Seasonality**: Evaluate seasonal-trend component at forecast horizons
2. **Transform Forecasts**:
   - **Additive**: new_forecast = old_forecast + seasonal_component
   - **Multiplicative**: new_forecast = old_forecast × seasonal_component
3. **Transform All Components**: Apply to mean, median, intervals, trajectories

# Component Transformation
- **Point forecasts**: Direct addition/multiplication
- **Intervals**: Transform both lower and upper bounds
- **Trajectories**: Transform entire trajectory matrix
- **Median**: Transform if available
"""
function postFilter(x::Vector{T1}, forecast::Forecast,
    setting::STTransform, pars::STParameter) where {T1 <: Real}
    h = maximum(forecast.horizon)

    meanVals = getST(pars, setting, length(x) + h)[forecast.horizon .+ length(x)]

    if setting.additive
        if has_mean(forecast)
            meanOut = forecast.mean .+ meanVals
        else
            meanOut = nothing
        end
        if has_median(forecast)
            medianOut = forecast.median .+ meanVals
        else
            medianOut = nothing
        end
        if has_intervals(forecast)
            intervalsOut = (h -> ForecastInterval(forecast.intervals[h].lower .+ meanVals[h], forecast.intervals[h].upper .+ meanVals[h], forecast.intervals[h].levels)).(1:length(forecast.horizon))
        else
            intervalsOut = nothing
        end
        if has_trajectories(forecast)
            trajectoriesOut = forecast.trajectories .+ meanVals'
        else
            trajectoriesOut = nothing
        end
    else
        all(meanVals .> 0) || throw(DomainError("Seasonality has non-positive values"))
        if has_mean(forecast)
            meanOut = forecast.mean .* meanVals
        else
            meanOut = nothing
        end
        if has_median(forecast)
            medianOut = forecast.median .* meanVals
        else
            medianOut = nothing
        end
        if has_intervals(forecast)
            intervalsOut = (h -> ForecastInterval(forecast.intervals[h].lower .* meanVals[h], forecast.intervals[h].upper .* meanVals[h], forecast.intervals[h].levels)).(1:length(forecast.horizon))
        else
            intervalsOut = nothing
        end
        if has_trajectories(forecast)
            trajectoriesOut = forecast.trajectories .* meanVals'
        else
            trajectoriesOut = nothing
        end
    end
    
    Forecast(horizon = forecast.horizon,
        mean = meanOut,
        median = medianOut,
        intervals = intervalsOut,
        truth = forecast.truth,
        trajectories = trajectoriesOut,
        reference_date = forecast.reference_date,
        target_date = forecast.target_date,
        resolution = forecast.resolution,
        model_name = forecast.model_name)
end
