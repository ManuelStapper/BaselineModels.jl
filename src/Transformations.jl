abstract type AbstractDataTransformation end

"""
    NoTransform()

Identity transformation (no transformation applied).

Passes data through unchanged. Used as default when no transformation is needed
or to explicitly specify that no transformation should be applied.

# Mathematical Definition
- Forward: f(x) = x  
- Inverse: f⁻¹(y) = y
"""
struct NoTransform <: AbstractDataTransformation end

"""
    LogPlusOneTransform(c = 1)

Logarithmic transformation with additive constant.

Applies log(x + c) transformation, useful for data that may contain zeros
or when a shift is needed before taking logarithms.

# Fields
- `constant::Float64`: Additive constant c (default: 1)

# Mathematical Definition
- Forward: f(x) = log(x + c)
- Inverse: f⁻¹(y) = exp(y) - c

# Requirements
- Forward: x > -c (i.e., x + c > 0)
- Handles zero values when c > 0
"""
struct LogPlusOneTransform <: AbstractDataTransformation
    constant::Float64
    function LogPlusOneTransform(c = 1)
        c isa Real || throw(ArgumentError("Constant must be a real number"))
        new(c)
    end
end

"""
    LogTransform()

Natural logarithm transformation.

Applies log(x) transformation for strictly positive data.

# Mathematical Definition
- Forward: f(x) = log(x)
- Inverse: f⁻¹(y) = exp(y)

# Requirements
- Forward: x > 0 (strictly positive)
- Undefined for zero or negative values
"""
struct LogTransform <: AbstractDataTransformation end

"""
    SquareRootTransform()

Square root transformation.

# Mathematical Definition  
- Forward: f(x) = √x
- Inverse: f⁻¹(y) = y²

# Requirements
- Forward: x ≥ 0 (non-negative)
- Suitable for count data and non-negative continuous data
"""
struct SquareRootTransform <: AbstractDataTransformation end

"""
    PowerTransform(lambda)

Box-Cox style power transformation.

# Fields
- `lambda::Float64`: Power parameter

# Mathematical Definition
- Forward: f(x) = x^λ (if λ ≠ 0), log(x) (if λ = 0)
- Inverse: f⁻¹(y) = y^(1/λ) (if λ ≠ 0), exp(y) (if λ = 0)

# Requirements
- Forward: x > 0 (strictly positive)

# Special Cases
- λ = 1: Identity transformation
- λ = 0.5: Square root transformation
- λ = 0: Log transformation
- λ = -1: Reciprocal transformation
"""
struct PowerTransform <: AbstractDataTransformation
    lambda::Float64
    function PowerTransform(lambda)
        lambda isa Real || throw(ArgumentError("lambda must be a real number"))
        new(lambda)
    end
end

"""
    PowerPlusOneTransform(lambda, c)

Power transformation with additive constant.

# Fields
- `lambda::Float64`: Power parameter
- `constant::Float64`: Additive constant

# Mathematical Definition
- Forward: f(x) = (x + c)^λ (if λ ≠ 0), log(x + c) (if λ = 0)
- Inverse: f⁻¹(y) = y^(1/λ) - c (if λ ≠ 0), exp(y) - c (if λ = 0)

# Requirements
- Forward: x > -c (i.e., x + c > 0)
"""
struct PowerPlusOneTransform <: AbstractDataTransformation
    lambda::Float64
    constant::Float64
    function PowerPlusOneTransform(lambda, c)
        lambda isa Real || throw(ArgumentError("lambda must be a real number"))
        c isa Real || throw(ArgumentError("constant must be a real number"))
        new(lambda, c)
    end
end

"""
    transform(x::Vector{T}, t::AbstractDataTransformation) -> Vector{Float64}

Apply forward transformation to data vector.

# Arguments
- `x::Vector{T}`: Input data where T <: Real
- `t::AbstractDataTransformation`: Transformation specification

# Returns
- `Vector{Float64}`: Transformed data
"""
function transform(x::Vector{T}, t::NoTransform) where {T <: Real}
    x
end

"""
    inverse_transform(y::Vector{T}, t::AbstractDataTransformation) -> Vector{Float64}

Apply inverse transformation to recover original scale.

# Arguments
- `y::Vector{T}`: Transformed data where T <: Real
- `t::AbstractDataTransformation`: Same transformation used for forward transform

# Returns
- `Vector{Float64}`: Data on original scale
"""
function inverse_transform(x::Vector{T}, t::NoTransform) where {T <: Real}
    x
end

function transform(x::Vector{T}, t::LogPlusOneTransform) where {T <: Real}
    (minimum(x) > -t.constant) || throw(ArgumentError("Log transform requires positive values"))
    return log.(x .+ t.constant)
end

function inverse_transform(y::Vector{T}, t::LogPlusOneTransform) where {T <: Real}
    return exp.(y) .- t.constant
end

function transform(x::Vector{T}, t::LogTransform) where {T <: Real}
    all(x .> 0) || throw(ArgumentError("Log transform requires positive values"))
    return log.(x)
end

function inverse_transform(y::Vector{Float64}, t::LogTransform)
    return exp.(y)
end

function transform(x::Vector{T}, t::PowerTransform) where {T <: Real}
    all(x .> 0) || throw(ArgumentError("Power transform requires positive values"))
    if t.lambda == 0
        return log.(x)
    else
        x.^(t.lambda)
    end
end

function inverse_transform(y::Vector{T}, t::PowerTransform) where {T <: Real}
    all(y .> 0) || throw(ArgumentError("Power transform requires positive values"))
    if t.lambda == 0
        return exp.(y)
    else
        y.^(1 ./ t.lambda)
    end
end

function transform(x::Vector{T}, t::PowerPlusOneTransform) where {T <: Real}
    all(x .> t.constant) || throw(ArgumentError("Power transform requires positive values"))
    if t.lambda == 0
        return log.(x .+ t.constant)
    else
        (x .+ t.constant).^(t.lambda)
    end
end

function inverse_transform(y::Vector{T}, t::PowerPlusOneTransform) where {T <: Real}
    minimum(y) .> -t.constant || throw(ArgumentError("Power transform requires positive values"))
    if t.lambda == 0
        return exp.(y) .- t.constant
    else
        y.^(1 ./ t.lambda) .- t.constant
    end
end

"""
    TransformedModel(model::AbstractBaselineModel, transformation::Union{AbstractDataTransformation, Nothing}, season_trend::Union{STTransform, Nothing})

Wrapper for applying data transformations to any forecasting model.

# Fields
- `model::AbstractBaselineModel`: Underlying forecasting model
- `transformation::AbstractDataTransformation`: Data transformation to apply
- `season_trend::Union{STTransform, Nothing}`: Optional seasonal-trend preprocessing

# Integration with Seasonal-Trend
When both transformation and seasonal-trend processing are specified:
1. **Pre-filter**: Remove seasonality/trend using STTransform
2. **Transform**: Apply data transformation
3. **Model**: Fit underlying model to processed data
4. **Forecast**: Generate predictions on transformed scale
5. **Inverse Transform**: Convert forecasts back to original scale
6. **Post-filter**: Add seasonality/trend back to forecasts
"""
struct TransformedModel
    model::AbstractBaselineModel
    transformation::Union{AbstractDataTransformation, Nothing}
    season_trend::Union{STTransform, Nothing}
end

"""
    TransformedParameter(modelParameter, seasParameter, xTransformed)

Parameters for fitted transformed model.

# Fields
- `modelParameter::AbstractModelParameters`: Parameters of underlying model
- `seasParameter::Union{STParameter, Nothing}`: Seasonal-trend parameters (if used)
- `xTransformed::Vector{Float64}`: Data after all transformations applied

Contains all information needed to reverse transformations and generate
forecasts on the original scale.
"""
struct TransformedParameter
    modelParameter::AbstractModelParameters
    seasParameter::Union{STParameter, Nothing}
    xTransformed::Vector{Float64}
    function TransformedParameter(modelParameter, seasParameter, xTransformed)
        xTransformed isa Vector{Float64} || throw(ArgumentError("xTransformed must be a real-valued Vector"))
        new(modelParameter, seasParameter, xTransformed)
    end
end

"""
    TransformedFitted

Container for fitted model with transformations.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::TransformedModel`: Model with transformation specifications
- `par::TransformedParameter`: All fitted parameters
- `fitted_baseline::AbstractFittedModel`: Fitted underlying model
- `estimation_setting`: Estimation settings used
- `temporal_info::TemporalInfo`: Temporal metadata

Stores complete transformation chain for proper forecast inversion.
"""
struct TransformedFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::TransformedModel
    par::TransformedParameter
    fitted_baseline::AbstractFittedModel
    estimationSetting::Union{AbstractEstimationSetting, Nothing}
    temporal_info::TemporalInfo
    function TransformedFitted(x, model::TransformedModel,
            par::TransformedParameter,
            fitted_baseline::AbstractFittedModel,
            estimation_setting::Union{AbstractEstimationSetting, Nothing} = nothing,
            temporal_info::TemporalInfo = TemporalInfo())
        if !(x isa Vector{Float64})
            Float64.(x)
        end
        new(x, model, par, fitted_baseline, estimation_setting, temporal_info)
    end
end

"""
    transform(model::AbstractBaselineModel; 
              transformation::Union{AbstractDataTransformation, Nothing} = nothing,
              season_trend::Union{STTransform, Nothing} = nothing) -> Union{AbstractBaselineModel, TransformedModel}

Add transformations to any forecasting model.

# Arguments
- `model::AbstractBaselineModel`: Base forecasting model
- `transformation`: Data transformation to apply (optional)
- `season_trend`: Seasonal-trend preprocessing (optional)

# Returns
- Original model if no transformations specified
- `TransformedModel` wrapper if transformations specified
"""
function transform(model::AbstractBaselineModel;
        transformation::Union{AbstractDataTransformation, Nothing} = nothing,
        season_trend::Union{STTransform, Nothing} = nothing)
    if isnothing(transformation) && isnothing(season_trend)
        return model
    end
    TransformedModel(model, transformation, season_trend)
end

"""
    fit_baseline(x::Vector{T}, model::TransformedModel; ...) -> TransformedFitted

Fit model with full transformation pipeline.

# Process
1. **Seasonal Filtering**: Apply STTransform if specified
2. **Data Transformation**: Apply data transformation if specified  
3. **Model Fitting**: Fit underlying model to processed data
4. **Parameter Storage**: Store all transformation parameters

# Returns
- `TransformedFitted`: Complete fitted model with transformation chain

All transformations are stored for proper forecast inversion.
"""
function fit_baseline(x::Vector{T},
        model::TransformedModel;
        setting::Union{AbstractEstimationSetting, Nothing} = nothing,
        temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    #
    if !isnothing(model.season_trend)
        xFiltered, seasPar = preFilter(x, model.season_trend)
    else
        xFiltered = x
        seasPar = nothing
    end

    if !isnothing(model.transformation)
        xTransformed = transform(xFiltered, model.transformation)
    else
        xTransformed = xFiltered
    end

    fitted_baseline = fit_baseline(xTransformed, model.model, setting = setting, temporal_info = temporal_info)
    tPars = TransformedParameter(fitted_baseline.par, seasPar, xTransformed)
    TransformedFitted(x, model, tPars, fitted_baseline, setting, temporal_info)
end

"""
    forecast(fitted::TransformedFitted; ...) -> Forecast

Generate forecasts with automatic transformation inversion.

# Process
1. **Base Forecasting**: Generate forecasts from underlying fitted model
2. **Inverse Transform**: Convert all forecast components back to original scale
   - Point forecasts
   - Median forecasts  
   - Prediction intervals (bounds transformed)
   - Sample trajectories (element-wise transformation)
3. **Seasonal Restoration**: Add seasonal-trend components back if used
"""
function forecast(fitted::TransformedFitted;
        interval_method::AbstractIntervalMethod = NoInterval(),
        horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
        levels::Vector{Float64} = [0.95],
        alpha_precision::Int = 10,
        include_median::Bool = true,
        truth::Union{Vector{Float64}, Nothing} = nothing,
        model_name::String = "")
    #
    fc_baseline = forecast(fitted.fitted_baseline,
        interval_method = interval_method,
        horizon = horizon,
        levels = levels,
        alpha_precision = alpha_precision,
        include_median = include_median,
        truth = truth,
        model_name = model_name)
    #
    fc_point = fc_baseline.mean
    fc_median = fc_baseline.median
    fc_intervals = fc_baseline.intervals
    fc_trajectories = fc_baseline.trajectories
    
    if !isnothing(fitted.model.transformation)
        t = fitted.model.transformation
        if !isnothing(fc_point)
            fc_point = inverse_transform(fc_point, t)
        end
        if !isnothing(fc_median)
            fc_median = inverse_transform(fc_median, t)
        end
        if !isnothing(fc_intervals)
            fc_intervals = (i -> ForecastInterval(inverse_transform(i.lower, t), inverse_transform(i.upper, t), i.levels)).(fc_intervals)
        end
        if !isnothing(fc_trajectories)
            for h = 1:size(fc_trajectories)[2]
                fc_trajectories[:, h] = inverse_transform(fc_trajectories[:, h], t)
            end            
        end
    end

    forecastOut = Forecast(horizon = fc_baseline.horizon,
        mean = fc_point,
        median = fc_median,
        intervals = fc_intervals,
        truth = fc_baseline.truth,
        trajectories = fc_trajectories,
        reference_date = fc_baseline.reference_date,
        target_date = fc_baseline.target_date,
        resolution = fc_baseline.resolution,
        model_name = fc_baseline.model_name)

    if !isnothing(fitted.model.season_trend)
        forecastOut = postFilter(fitted.x, forecastOut, fitted.model.season_trend, fitted.par.seasParameter)
    end
    forecastOut
end

"""
    transform(x::Nothing, t::AbstractDataTransformation) -> Nothing
    transform(x::Matrix{Float64}, t::AbstractDataTransformation) -> Matrix{Float64}
    transform(fc::Forecast, t::AbstractDataTransformation) -> Forecast

Extended transformation methods for different data types.

# Matrix Transformation
Applies transformation column-wise to matrix data (e.g., trajectory matrices).

# Forecast Transformation
Transforms all components of a Forecast object:
- Mean forecasts
- Median forecasts
- Interval bounds (lower and upper separately)
- Sample trajectories (element-wise)
"""
function transform(x::Nothing, t::AbstractDataTransformation)
    nothing
end

function transform(x::Matrix{Float64}, t::AbstractDataTransformation)
    out = zero(x)
    for i = 1:size(x)[2]
        out[:, i] = transform(x[:, i], t)
    end
    out
end

function transform(fc::Forecast, t::AbstractDataTransformation)
    if has_intervals(fc)
        intervalsOut = (ff -> ForecastInterval(transform(ff.lower, t), transform(ff.upper, t), ff.levels)).(fc.intervals)
    else
        intervalsOut = nothing
    end
    Forecast(horizon = fc.horizon,
        mean = transform(fc.mean, t),
        median = transform(fc.median, t),
        intervals = intervalsOut,
        truth = transform(fc.truth, t),
        trajectories = transform(fc.trajectories, t),
        reference_date = fc.reference_date,
        target_date = fc.target_date,
        resolution = fc.resolution)
end