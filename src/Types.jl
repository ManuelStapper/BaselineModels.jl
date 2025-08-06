"""
    AbstractBaselineModel

Abstract supertype for all baseline forecasting models.

Defines the interface that all forecasting models must implement. Models inheriting
from this type can be used with the standard fitting and forecasting pipeline.

# Required Interface
Models must implement:
- `fit_baseline(x, model; setting, temporal_info)` → fitted model
- `point_forecast(fitted, horizon)` → point forecasts

# Optional Interface  
Models may implement:
- `interval_forecast(fitted, method, horizon, levels; ...)` → prediction intervals

If interval methods are not implemented, `EmpiricalInterval` can be used automatically.
"""
abstract type AbstractBaselineModel end

"""
    AbstractIntervalMethod

Abstract supertype for prediction interval computation methods.

Defines different approaches to quantifying forecast uncertainty and generating
prediction intervals from fitted models.

# Concrete Types
- `NoInterval`: Point forecasts only (fastest)
- `EmpiricalInterval`: Bootstrap from historical errors (most flexible)
- `ParametricInterval`: Model-specific analytical formulas (exact when available)
- `ModelTrajectoryInterval`: Simulation-based intervals (full uncertainty)

# Interface
All interval methods must implement:
`interval_forecast(fitted, method, horizon, levels; ...)` → (point, median, intervals, trajectories)

# Method Selection Guidelines
- **Speed priority**: `NoInterval`
- **Flexibility**: `EmpiricalInterval` (works with any model)
- **Accuracy**: `ParametricInterval` (when analytical formulas available)
- **Full uncertainty**: `ModelTrajectoryInterval` (captures all sources)
"""
abstract type AbstractIntervalMethod end

"""
    AbstractEstimationSetting

Abstract supertype for model estimation settings.

Provides model-specific configuration options for parameter estimation.
Each model type typically defines its own concrete estimation setting type.

# Purpose
- Configure optimisation algorithms and constraints
- Control robustness and stability features
- Specify validation and convergence criteria
- Handle model-specific estimation options
"""
abstract type AbstractEstimationSetting end

"""
    AbstractFittedModel

Abstract supertype for fitted forecasting models.

Container for fitted models that stores all information needed for forecasting
and analysis. All fitted models follow a consistent structure.

# Standard Fields
All fitted models contain:
- `x::Vector{Float64}`: Original time series data
- `model::AbstractBaselineModel`: Model specification used
- `par::AbstractModelParameters`: Estimated model parameters
- `estimation_setting::AbstractEstimationSetting`: Settings used during fitting
- `temporal_info::TemporalInfo`: Temporal metadata
"""
abstract type AbstractFittedModel end

"""
    AbstractModelParameters

Abstract supertype for model parameter objects.

Stores estimated parameters in model-specific formats. Each model type
defines its own parameter structure optimized for that model's needs.
"""
abstract type AbstractModelParameters end

"""
    AbstractForecast

Abstract supertype for forecast objects.

Base type for forecast containers. Currently only implemented by `Forecast`,
but provides extensibility for specialised forecast types in the future.
"""
abstract type AbstractForecast end

"""
    ForecastInterval(lower, upper, levels)

Prediction interval representation for a single forecast horizon.

Stores multiple confidence levels and their corresponding interval bounds
for a single time point. Supports flexible confidence level specifications.

# Fields
- `lower::Vector{Float64}`: Lower bounds for each confidence level
- `upper::Vector{Float64}`: Upper bounds for each confidence level
- `levels::Vector{Float64}`: Confidence levels (e.g., [0.5, 0.8, 0.95])

# Examples
```julia
# Single 95% interval
interval = ForecastInterval([2.1], [4.7], [0.95])

# Multiple confidence levels
interval = ForecastInterval(
    [1.5, 2.1, 2.3],  # lower bounds
    [5.2, 4.7, 4.3],  # upper bounds  
    [0.5, 0.8, 0.95]  # confidence levels
)
```
"""
struct ForecastInterval
    lower::Vector{Float64}
    upper::Vector{Float64}
    levels::Vector{Float64}
    
    function ForecastInterval(lower, upper, levels)
        if !(lower isa Vector) & (length(lower) == 1)
            lower = [lower]
        end
        if !(upper isa Vector) & (length(upper) == 1)
            upper = [upper]
        end
        if !(levels isa Vector) & (length(levels) == 1)
            levels = [levels]
        end

        all(lower .<= upper) ||
            throw(ArgumentError("Lower bounds must be smaller than upper bounds"))
        length(lower) == length(upper) == length(levels) || 
            throw(ArgumentError("All vectors must have the same length"))
        all(0 .<= levels .<= 1) || 
            throw(ArgumentError("Confidence levels must be between 0 and 1"))
        new(lower, upper, levels)
    end
end

function validate_forecast_consistency(horizon, mean, median, intervals, truth, trajectories, target_date)
    provided_lengths = Int[]
    
    !isnothing(horizon) && push!(provided_lengths, length(horizon))
    !isnothing(mean) && push!(provided_lengths, length(mean))
    !isnothing(median) && push!(provided_lengths, length(median))
    !isnothing(intervals) && push!(provided_lengths, length(intervals))
    
    if length(unique(provided_lengths)) > 1
        throw(ArgumentError("Provided forecast components must have consistent lengths: $provided_lengths"))
    end
    
    n = isempty(provided_lengths) ? 0 : first(provided_lengths)
    
    if !isnothing(truth) && length(truth) != n
        throw(ArgumentError("Truth length ($(length(truth))) must match forecast length ($n)"))
    end
    
    if !isnothing(trajectories) && size(trajectories, 2) != n
        throw(ArgumentError("Trajectories columns ($(size(trajectories, 2))) must match forecast length ($n)"))
    end
    
    if !isnothing(target_date) && length(target_date) != n
        throw(ArgumentError("Target date length ($(length(target_date))) must match forecast length ($n)"))
    end
end

function generate_target_dates(reference_date::Union{DateTime,Date,Int}, horizon::Vector{Int}, 
                              resolution::Union{DatePeriod,Int})
    if reference_date isa Int
        return [reference_date + h * resolution for h in horizon]
    else
        if resolution isa Int
            period = Week(resolution)
        else
            period = resolution
        end
        return [reference_date + h * period for h in horizon]
    end
end

function validate_temporal_consistency(horizon, reference_date, target_date, resolution)
    if !isnothing(horizon) && !isnothing(reference_date) && !isnothing(target_date)
        expected_targets = generate_target_dates(reference_date, horizon, resolution)
        
        if typeof(target_date) != typeof(expected_targets)
            @warn "Target date type ($(typeof(target_date))) doesn't match expected type ($(typeof(expected_targets))) based on reference_date type"
            return  
        end
        
        if target_date != expected_targets
            @warn "Target dates don't match expected dates based on reference_date, horizon, and resolution"
            @warn "Expected: $expected_targets"
            @warn "Got: $target_date"
        end
    end
end

"""
    Forecast(;horizon = nothing, mean = nothing, median = nothing, 
             intervals = nothing, truth = nothing, trajectories = nothing,
             reference_date = nothing, target_date = nothing, 
             resolution = 1, model_name = "Unnamed Model")

Main forecast container with comprehensive metadata and validation.

Central data structure containing all forecast information: predictions,
uncertainty quantification, temporal metadata, and validation data.

# Core Forecast Fields
- `horizon::Vector{Int}`: Forecast horizons (e.g., [1, 2, 3, ...])
- `mean::Vector{Float64}`: Point forecasts
- `median::Vector{Float64}`: Median forecasts (may differ from mean)
- `intervals::Vector{ForecastInterval}`: Prediction intervals for each horizon
- `truth::Vector{Float64}`: Observed values for evaluation (optional)
- `trajectories::Matrix{Float64}`: Sample trajectories (rows=samples, cols=horizons)

# Temporal/Metadata Fields
- `reference_date::Union{DateTime, Date, Time, Int, Nothing}`: Date of last observation
- `target_date::Vector{Union{DateTime, Date, Time, Int}}`: Dates being forecast
- `resolution::Union{DatePeriod, TimePeriod, Int}`: Time step between observations
- `model_name::String`: Descriptive name for the forecast

# Validation Rules
- All forecast components (mean, median, intervals) must have same length
- Truth values must match forecast length if provided
- Trajectories must have columns matching forecast length
- Target dates must match forecast length if provided
- Temporal consistency checks between reference/target dates

# Examples
```julia
# Simple point forecast
fc = Forecast(horizon=1:3, mean=[2.1, 2.3, 2.5], model_name="Simple Model")

# Complete forecast with intervals and temporal info
fc = Forecast(
    horizon = 1:12,
    mean = monthly_forecasts,
    intervals = prediction_intervals,
    truth = observed_values,
    reference_date = Date("2024-12-31"),
    resolution = Month(1),
    model_name = "ARMA(2,1) Monthly"
)

# Trajectory-based forecast
fc = Forecast(
    horizon = 1:6,
    mean = point_forecasts,
    trajectories = simulation_matrix,  # 1000 x 6 matrix
    model_name = "Monte Carlo Forecast"
)
```
"""
struct Forecast <: AbstractForecast
    horizon::Union{Vector{Int}, Nothing}
    mean::Union{Vector{Float64}, Nothing}
    median::Union{Vector{Float64}, Nothing}
    intervals::Union{Vector{ForecastInterval}, Nothing}
    truth::Union{Vector{Float64}, Nothing}
    trajectories::Union{Matrix{Float64}, Nothing}
    reference_date::Union{DateTime, Date, Time, Int, Nothing}
    target_date::Union{Vector{DateTime}, Vector{Date}, Vector{Time}, Vector{Int}, Nothing}
    resolution::Union{DatePeriod, TimePeriod, Int}
    model_name::String
    
    function Forecast(;horizon = nothing, mean = nothing, median = nothing, intervals = nothing, truth=nothing, 
                     trajectories=nothing, reference_date = nothing, target_date = nothing, resolution = 1, model_name="Unnamed Model")
        if horizon isa Integer
            horizon = [horizon]
        elseif !(horizon isa Vector{Int}) & !isnothing(horizon)
            throw(ArgumentError("'horizon' must be an integer vector"))
        end

        if mean isa Real
            mean = [mean]
        elseif !(mean isa Vector{T} where {T <: Real}) & !isnothing(mean)
            throw(ArgumentError("'mean' must be a real-valued vector"))
        end

        if median isa Real
            median = [median]
        elseif !(median isa Vector{T} where {T <: Real}) & !isnothing(median)
            throw(ArgumentError("'median' must be a real-valued vector"))
        end

        if truth isa Real
            truth = [truth]
        elseif !(truth isa Vector{T} where {T <: Real}) & !isnothing(truth)
            throw(ArgumentError("'truth' must be a real-valued vector"))
        end

        if trajectories isa AbstractVector
            trajectories = reshape(trajectories, (length(trajectories), 1))
        elseif !(trajectories isa Matrix{T} where {T <: Real}) & !isnothing(trajectories)
            throw(ArgumentError("'trajectories must be a real-valued matrix"))
        end

        if intervals isa ForecastInterval
            intervals = [intervals]
        end

        forecast_fields = [horizon, mean, median, intervals]
        has_forecast_data = any(!isnothing, forecast_fields)
        
        if has_forecast_data
            validate_forecast_consistency(horizon, mean, median, intervals, truth, trajectories, target_date)
            
            if !isnothing(reference_date) && !isnothing(horizon) && isnothing(target_date)
                target_date = generate_target_dates(reference_date, horizon, resolution)
            end
            
            validate_temporal_consistency(horizon, reference_date, target_date, resolution)
        end
        
        new(horizon, mean, median, intervals, truth, trajectories, 
            reference_date, target_date, resolution, model_name)
    end
end

function string_to_start(s::String)
    if length(split(s, "T")) > 1
        return DateTime(s)
    end
    if length(split(s, "-")) > 1
        return Date(s)
    end
    return Time(s)
end

"""
    TemporalInfo(start = 1, resolution = 1)

Temporal metadata for time series and forecasts.

Stores information about time indexing, dating, and resolution for proper
temporal handling throughout the forecasting pipeline.

# Fields
- `start::Union{TimeType, Int}`: Starting time/date or integer index
- `resolution::Union{DatePeriod, TimePeriod, Int}`: Time step between observations

# Supported Time Types
- **Dates**: `Date`, `DateTime`, `Time` from Dates.jl
- **Integers**: Simple integer indexing (1, 2, 3, ...)
- **Periods**: `Day`, `Month`, `Year`, `Hour`, `Minute`, etc.

# Constructor Features
- **String Parsing**: Automatically parses date/time strings
- **Type Consistency**: Ensures start and resolution types are compatible
- **Validation**: Checks for valid combinations of start and resolution types

# Type Compatibility Rules
- `Int` start requires `Int` resolution
- `Date` start works with `DatePeriod` resolution (Day, Month, Year, etc.)
- `DateTime` start works with any period type
- `Time` start requires `TimePeriod` resolution (Hour, Minute, Second, etc.)

# Examples
```julia
# Integer indexing (default)
temporal = TemporalInfo()  # start=1, resolution=1

# Daily data starting from specific date
temporal = TemporalInfo(Date("2024-01-01"), Day(1))

# Monthly data with string parsing
temporal = TemporalInfo("2024-01-01", Month(1))

# High-frequency data
temporal = TemporalInfo(DateTime("2024-01-01T09:00:00"), Minute(15))

# Business time indexing
temporal = TemporalInfo(1, 1)  # Simple integer sequence
```

# Integration
Used throughout the forecasting pipeline:
- Model fitting: Stores original series temporal information
- Forecasting: Generates proper target dates for forecasts
- Evaluation: Aligns forecasts with observation timing
- Visualisation: Enables proper time axis construction

# Automatic Date Generation
When used with forecast generation, automatically computes target dates:
```julia
temporal = TemporalInfo(Date("2024-12-31"), Month(1))
# Forecasting with horizon 1:6 generates targets:
# 2025-01-31, 2025-02-28, 2025-03-31, 2025-04-30, 2025-05-31, 2025-06-30
```
"""
struct TemporalInfo
    start::Union{TimeType, Int}
    resolution::Union{DatePeriod, TimePeriod, Int}
    function TemporalInfo(start = 1, resolution = 1)
        if start isa String
            start = string_to_start(start)
        end
        ((start isa Int) || (start isa TimeType)) || throw(ArgumentError("Invalid start, must be time (string) or integer"))
        ((resolution isa Int) || (resolution isa DatePeriod) || (resolution isa TimePeriod)) || throw(ArgumentError("Invalid resolution"))

        !xor(start isa Int,  resolution isa Int) || throw(ArgumentError("Start and resolution do not match"))

        if (start isa Date) & (resolution isa TimePeriod)
            start = DateTime(start)
        end

        if (start isa Time) & (resolution isa DatePeriod)
            throw(ArgumentError("Start and resolution do not match"))
        end

        new(start, resolution)
    end
end

