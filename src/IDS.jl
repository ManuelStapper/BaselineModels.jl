###########################
### Extrapolation Model ###
###########################

"""
    IDSModel(;p::Int = 3)

Increase-Decrease-Stable model for trend-aware forecasting.

Adaptive forecasting method that detects trend consistency in recent observations
and applies linear extrapolation only when a clear directional pattern exists.
Otherwise reverts to constant (mean) forecasting.

# Fields
- `p::Int`: Number of recent observations to analyse for trend detection (default: 3)

# Algorithm Logic
1. **Trend Detection**: Examine signs of differences in last p observations
2. **Consistency Check**: If all differences have the same sign, trend detected
3. **Forecasting Strategy**:
   - **Trend detected**: Fit linear trend and extrapolate
   - **No trend**: Use mean of recent observations (constant forecast)
"""
struct IDSModel <: AbstractBaselineModel
    p::Int
    function IDSModel(;p::Int = 3)
        (p > 0) || throw(ArgumentError("`p` must be positive"))
        new(p)
    end
end

"""
    IDSParameter(a::Real, b::Real)

Parameters for fitted IDS model.

# Fields
- `a::Float64`: Intercept parameter (level component)
- `b::Float64`: Slope parameter (trend component, 0 if no trend detected)

When no trend is detected, a = mean of recent observations and b = 0.
When trend is detected, (a,b) are OLS estimates from linear regression.
"""
struct IDSParameter <: AbstractModelParameters
    a::Float64
    b::Float64
    function IDSParameter(a, b)
        ((a isa Real) & (b isa Real)) || throw(ArgumentError("Parameters must be real numbers"))
        new(Float64(a), Float64(b))
    end
end

"""
    IDSEstimationSetting()

Estimation settings for IDS model.

Empty struct as IDS uses simple deterministic rules rather than
statistical estimation. Included for interface consistency.
"""
struct IDSEstimationSetting <: AbstractEstimationSetting
end

"""
    IDSFitted

Container for fitted IDS model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::IDSModel`: Model specification  
- `par::IDSParameter`: Estimated parameters
- `estimation_setting::IDSEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata
"""
struct IDSFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::IDSModel
    par::IDSParameter
    estimation_setting::IDSEstimationSetting
    temporal_info::TemporalInfo
    function IDSFitted(x, model::IDSModel, par::IDSParameter,
        estimation_setting::IDSEstimationSetting,
        temporal_info::TemporalInfo = TemporalInfo())
        #
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    fit_baseline(x::Vector{T}, model::IDSModel;
                setting::Union{IDSEstimationSetting, Nothing} = IDSEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> IDSFitted

Fit IDS model using trend detection and conditional estimation.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::IDSModel`: IDS model specification
- `setting`: Estimation settings (optional, no effect)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `IDSFitted`: Fitted model with conditional parameters

# Fitting Process
1. **Extract Recent Data**: Use last min(p, length(x)) observations
2. **Compute Differences**: Calculate period-to-period changes
3. **Check Trend Consistency**: Verify all differences have same sign
4. **Conditional Estimation**:
   - **Trend detected**: OLS regression on recent data
   - **No trend**: Simple mean calculation

# Implementation Notes
- Uses OLS implementation internally when trend detected
- Automatically handles cases where p > length(x)
"""
function fit_baseline(x::Vector{T},
        model::IDSModel;
        setting::Union{IDSEstimationSetting, Nothing} = IDSEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    
    p = minimum([model.p, length(x)])
    y = x[end-p + 1:end]
    s = sign.(diff(y))
    if length(unique(s)) == 1
        X = [ones(p) 1 - p:0]
        β = inv(X'X)*X'y
        par = IDSParameter(β[1], β[2])
    else
        par = IDSParameter(mean(y), 0)
    end

    if isnothing(setting)
        setting = IDSEstimationSetting()
    end
    IDSFitted(x, model, par, setting, temporal_info)
end

"""
    point_forecast(fitted::IDSFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts using conditional trend extrapolation.

# Arguments
- `fitted::IDSFitted`: Fitted IDS model
- `horizon`: Forecast horizons

# Returns
- `Vector{Float64}`: Point forecasts

# Forecasting Logic
Uses the fitted linear model parameters regardless of trend detection:
- X̂_{T+h} = a + b × h for all h > 0

When no trend was detected (b = 0), this reduces to constant forecasting.
When trend was detected (b ≠ 0), this extrapolates the linear pattern.

This continues the fitted linear relationship into the future, where:
- a captures the level component
- b captures the trend component (zero if no trend detected)

# Example
```julia
# Generate forecasts
forecasts = point_forecast(fitted, 1:5)

# For trending series: forecasts show linear growth
# For non-trending series: forecasts are constant
```
"""
function point_forecast(fitted::IDSFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))
    
    p = minimum([fitted.model.p, length(fitted.x)])
    y = fitted.x[end-p + 1:end]
    s = sign.(diff(y))

    # Use OLS model implementation for forecasts
    ols_model = OLSModel(p = p, d = 1)
    ols_par = OLSParameter([fitted.par.a, fitted.par.b])
    ols_fitted = OLSFitted(fitted.x, ols_model, ols_par, OLSEstimationSetting(),
        fitted.temporal_info)
    point_forecast(ols_fitted, horizon)
end

# Parametric forecast intervals
"""
    interval_forecast(fitted::IDSFitted, method::ParametricInterval, ...) -> Tuple

Generate parametric prediction intervals using OLS framework.

Converts IDS parameters to equivalent OLS model and applies standard
linear regression interval formulas. Provides analytical prediction
intervals when parametric approach is preferred.

Delegates to OLS interval computation after parameter transformation.
"""
function interval_forecast(fitted::IDSFitted,
    method::ParametricInterval,
    horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
    levels::Vector{Float64} = [0.95];
    alpha_precision::Int = 10,
    include_median::Bool = true)

    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    # Validate input:
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    p = minimum([fitted.model.p, length(fitted.x)])
    y = fitted.x[end-p + 1:end]
    s = sign.(diff(y))

    # Use OLS model implementation for forecasts
    ols_model = OLSModel(p = p, d = 1)
    ols_par = OLSParameter([fitted.par.a, fitted.par.b])
    ols_fitted = OLSFitted(fitted.x, ols_model, ols_par, OLSEstimationSetting(),
        fitted.temporal_info)
    interval_forecast(ols_fitted, method, horizon, levels,
        alpha_precision = alpha_precision, include_median = include_median)
end

"""
    interval_forecast(fitted::IDSFitted, method::ModelTrajectoryInterval, ...) -> Tuple

Generate trajectory-based prediction intervals using OLS simulation.

Converts IDS parameters to equivalent OLS model and applies trajectory
sampling methodology.

Delegates to OLS trajectory generation after parameter transformation.
"""
function interval_forecast(fitted::IDSFitted,
    method::ModelTrajectoryInterval,
    horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
    levels::Vector{Float64} = [0.95];
    alpha_precision::Int = 10,
    include_median::Bool = true)

    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    # Validate input:
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    p = minimum([fitted.model.p, length(fitted.x)])
    y = fitted.x[end-p + 1:end]
    s = sign.(diff(y))

    # Use OLS model implementation for forecasts
    ols_model = OLSModel(p = p, d = 1)
    ols_par = OLSParameter([fitted.par.a, fitted.par.b])
    ols_fitted = OLSFitted(fitted.x, ols_model, ols_par, OLSEstimationSetting(),
        fitted.temporal_info)
    interval_forecast(ols_fitted, method, horizon, levels,
        alpha_precision = alpha_precision, include_median = include_median)
end