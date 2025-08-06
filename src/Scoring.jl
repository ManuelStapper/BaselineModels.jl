# Supertype for different scoring rules
abstract type AbstractScoringRule end
abstract type AbstractPointScoringRule <: AbstractScoringRule end
abstract type AbstractIntervalScoringRule <: AbstractScoringRule end
abstract type AbstractTrajectoryScoringRule <: AbstractScoringRule end

"""
    PointScoringRule(;error::AbstractPointError = ForecastError(),
                     forecast_transformation::AbstractDataTransformation = NoTransform(),
                     target::Symbol = :mean,
                     missing_strategy::Symbol = :skip,
                     error_transformation::Function = identity,
                     forecast_aggregation_function::Function = mean,
                     result_transformation::Function = identity)

Flexible scoring rule for point forecasts with customisable components.

Provides a composable framework for creating point forecast scoring rules
by combining error types, transformations, and aggregation functions.

# Fields
- `error::AbstractPointError`: Type of forecast error to compute
- `forecast_transformation::AbstractDataTransformation`: Transform forecasts before scoring
- `target::Symbol`: Which forecast component to evaluate (:mean, :median, :qXX)
- `missing_strategy::Symbol`: How to handle missing values (:skip, :error, :propagate)
- `error_transformation::Function`: Transform errors before aggregation
- `forecast_aggregation_function::Function`: Aggregate errors across forecasts
- `result_transformation::Function`: Final transformation of aggregated result

# Target Specification
- `:mean`: Use point forecasts
- `:median`: Use median forecasts
- `:qXX`: Use quantile forecasts (e.g., `:q25`, `:q75`, `:q95`)
"""
struct PointScoringRule <: AbstractPointScoringRule
    error::AbstractPointError
    forecast_transformation::AbstractDataTransformation
    target::Symbol
    missing_strategy::Symbol
    error_transformation::Function
    forecast_aggregation_function::Function
    result_transformation::Function

    function PointScoringRule(;error::AbstractPointError = ForecastError(),
        forecast_transformation::AbstractDataTransformation = NoTransform(),
        target::Symbol = :mean,
        missing_strategy::Symbol = :skip,
        error_transformation::Function = identity,
        forecast_aggregation_function::Function = mean,
        result_transformation::Function = identity)
        if !(target in [:mean, :median]) && !((string(target)[1] == 'q') && all((n -> n in string.(0:9)).(split(string(target), "")[2:end])))
            target = :mean
        end
        if !(missing_strategy in [:error, :skip, :propagate])
            missing_strategy = :skip
        end
        new(error, forecast_transformation, target, missing_strategy,
            error_transformation, forecast_aggregation_function, result_transformation)
    end
end

"""
    MAE(;missing_strategy::Symbol = :skip) -> PointScoringRule

Mean Absolute Error scoring rule.
"""
function MAE(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = AbsoluteError(),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = identity)
end

"""
    MdAE(;missing_strategy::Symbol = :skip) -> PointScoringRule

Median Absolute Error scoring rule.
"""
function MdAE(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = AbsoluteError(),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = median,
        result_transformation = identity)
end

"""
    MAPE(;missing_strategy::Symbol = :skip) -> PointScoringRule

Mean Absolute Percentage Error scoring rule.
"""
function MAPE(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = AbsoluteError(relative = true),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = identity)
end

"""
    MSE(;missing_strategy::Symbol = :skip) -> PointScoringRule

Mean Squared Error scoring rule.
"""
function MSE(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = SquaredError(),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = identity)
end

"""
    MSPE(;missing_strategy::Symbol = :skip) -> PointScoringRule

Mean Squared Percentage Error scoring rule.
"""
function MSPE(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = SquaredError(relative = true),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = identity)
end

"""
    RMSE(;missing_strategy::Symbol = :skip) -> PointScoringRule  

Root Mean Squared Error scoring rule.
"""
function RMSE(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = SquaredError(),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = sqrt)
end

"""
    Bias(;missing_strategy::Symbol = :skip) -> PointScoringRule

Forecast bias scoring rule. Computes E[forecast - truth].
"""
function Bias(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = ForecastError(),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = identity)
end

"""
    RelativeBias(;missing_strategy::Symbol = :skip) -> PointScoringRule

Relative forecast bias scoring rule. Computes E[(forecast - truth)/truth].
"""
function RelativeBias(;missing_strategy::Symbol = :skip)
    PointScoringRule(error = ForecastError(relative = true),
        forecast_transformation = NoTransform(),
        target = :mean,
        missing_strategy = missing_strategy,
        error_transformation = identity,
        forecast_aggregation_function = mean,
        result_transformation = identity)
end

"""
    score(fc::Union{Vector{Forecast}, Forecast}, rule::PointScoringRule;
          horizon::Union{Int, Nothing, Vector{Int}, UnitRange{Int}} = nothing,
          transformation::AbstractDataTransformation = NoTransform(),
          by_horizon::Bool = false) -> Union{Float64, Vector{Float64}}

Evaluate point forecasts using specified scoring rule.

# Arguments
- `fc::Vector{Forecast}`: Collection of forecasts to evaluate
- `rule::PointScoringRule`: Scoring rule specification
- `horizon`: Specific horizon(s) to evaluate (default: all available)
- `transformation`: Additional transformation applied before scoring
- `by_horizon::Bool`: Return separate scores for each horizon

# Returns
- `Float64`: Overall score (if by_horizon=false)
- `Vector{Float64}`: Score for each horizon (if by_horizon=true)

# Horizon Specification
- `nothing`: Evaluate all available horizons
- `Int`: Single horizon (e.g., horizon=1)
- `Vector{Int}`: Multiple horizons (e.g., horizon=[1,3,6])
- `UnitRange{Int}`: Range of horizons (e.g., horizon=1:12)
"""
function score(fc::Union{Vector{Forecast}, Forecast},
        rule::PointScoringRule;
        horizon::Union{Int, Nothing, Vector{Int}, UnitRange{Int}} = nothing,
        transformation::T1 = NoTransform(),
        by_horizon::Bool = false) where {T1 <: AbstractDataTransformation}
    if fc isa Forecast
        fc = [fc]
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    if by_horizon
        if horizon isa Vector{Int}
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(horizon)
        elseif isnothing(horizon)
            all_horizons = sort(unique(vcat((f -> f.horizon).(fc)...)))
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(all_horizons)
        end
    end
    fcVec = remove_trajectories.(fc)

    if horizon isa Int
        fcVec = fcVec[(f -> any(f.horizon .== horizon)).(fcVec)]
        fcVec = (f -> filter_horizons(f, [horizon])).(fcVec)
    elseif horizon isa Vector{Int}        
        fcVec = fcVec[(f -> length(intersect(f.horizon, horizon)) > 0).(fcVec)]
        fcVec = (f -> filter_horizons(f, horizon)).(fcVec)
    end

    fcVec = (f -> transform(f, transformation)).(fcVec)
    errors = vcat(all_forecast_errors(fcVec, rule.error, missing_strategy = rule.missing_strategy)...)
    errors = rule.error_transformation.(errors)
    out = rule.forecast_aggregation_function(errors)
    rule.result_transformation(out)
end

"""
    WIS(;levels::Union{Vector{Float64}, Nothing} = nothing,
        level_weights::Union{Vector{Float64}, Nothing} = nothing,
        missing_strategy::Symbol = :skip)

Weighted Interval Score for prediction interval evaluation.

# Fields
- `levels::Vector{Float64}`: Specific confidence levels to evaluate (optional)
- `level_weights::Vector{Float64}`: Weights for each confidence level (optional)
- `missing_strategy::Symbol`: Missing value handling strategy

# Default Weighting
When no weights specified: w₀ = 1/2, wₖ = (1-αₖ)/2 for level αₖ
"""
struct WIS <: AbstractIntervalScoringRule
    levels::Union{Vector{Float64}, Nothing}
    level_weights::Union{Vector{Float64}, Nothing}
    missing_strategy::Symbol

    function WIS(;levels::Union{Vector{Float64}, Nothing} = nothing,
        level_weights::Union{Vector{Float64}, Nothing} = nothing,
        missing_strategy::Symbol = :skip)
        if !(missing_strategy in [:error, :skip, :propagate])
            missing_strategy = :skip
        end
        if !isnothing(levels) && !isnothing(level_weights)
            length(levels) == length(level_weights) || throw(ArgumentError("Level and weights must have the same length"))
        end
        new(levels, level_weights, missing_strategy)
    end
end

function score(fc::Union{Vector{Forecast}, Forecast},
        rule::WIS;
        horizon::Union{Int, Nothing, Vector{Int}, UnitRange{Int}} = nothing,
        transformation::T1 = NoTransform(),
        by_horizon::Bool = false) where {T1 <: AbstractDataTransformation}
    if fc isa Forecast
        fc = [fc]
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    if by_horizon
        if horizon isa Vector{Int}
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(horizon)
        elseif isnothing(horizon)
            all_horizons = sort(unique(vcat((f -> f.horizon).(fc)...)))
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(all_horizons)
        end
    end
    fcVec = remove_trajectories.(fc)
    if horizon isa Int
        fcVec = fcVec[(f -> any(f.horizon .== horizon)).(fcVec)]
        fcVec = (f -> filter_horizons(f, [horizon])).(fcVec)
    elseif horizon isa Vector{Int}        
        fcVec = fcVec[(f -> length(intersect(f.horizon, horizon)) > 0).(fcVec)]
        fcVec = (f -> filter_horizons(f, horizon)).(fcVec)
    end
    if !isnothing(rule.levels)
        fcVec = (f -> filter_levels(f, rule.levels)).(fcVec)
    end

    fcVec = (f -> transform(f, transformation)).(fcVec)

    all_levels = vcat((f -> (i -> i.levels).(f.intervals)).(fcVec)...)
    unique_levels = unique(vcat(all_levels))
    if length(unique_levels) > 1
        min_levels = unique_levels[1]
        for i = 2:length(unique_levels)
            min_levels = intersect(min_levels, unique_levels[i])
        end
        fcVec = (f -> filter_levels(f, min_levels)).(fcVec)
    else
        min_levels = unique_levels[1]
    end

    errors = vcat(all_forecast_errors(fcVec, IntervalScore(), missing_strategy = rule.missing_strategy)...)
    medians = vcat((f -> f.median).(fcVec)...)
    truths = vcat((f -> f.truth).(fcVec)...)

    # If no weights are given, use default approach
    if isnothing(rule.level_weights)
        w = [1; 1 .- min_levels]./2
    else
        length(rule.level_weights) == length(min_levels) || throw(ArgumentError("Number of weights must equal number of levels"))
        w = rule.level_weights
    end
    errors = w[1] .* abs.(medians .- truths) .+ (e -> sum(e .* w[2:end])).(errors)
    
    mean(errors)
end

"""
    CRPS(;levels::Union{Vector{Float64}, Nothing} = nothing,
         missing_strategy::Symbol = :skip)

Continuous Ranked Probability Score for probabilistic forecasts.

Proper scoring rule measuring distance between forecast and empirical CDFs.
Approximated using prediction intervals when full distribution unavailable.

# Fields
- `levels::Vector{Float64}`: Confidence levels for CRPS approximation
- `missing_strategy::Symbol`: Missing value handling

where F(x) is forecast CDF and 𝟙{y ≤ x} is empirical CDF.

# Approximation Method
Uses trapezoidal integration over quantile levels to approximate
the continuous integral using available prediction intervals.
"""
struct CRPS <: AbstractIntervalScoringRule
    levels::Union{Vector{Float64}, Nothing}
    missing_strategy::Symbol

    function CRPS(;levels::Union{Vector{Float64}, Nothing} = nothing,
        missing_strategy::Symbol = :skip)
        if !(missing_strategy in [:error, :skip, :propagate])
            missing_strategy = :skip
        end
        new(levels, missing_strategy)
    end
end

function score(fc::Union{Vector{Forecast}, Forecast},
        rule::CRPS;
        horizon::Union{Int, Nothing, Vector{Int}, UnitRange{Int}} = nothing,
        transformation::T1 = NoTransform(),
        by_horizon::Bool = false) where {T1 <: AbstractDataTransformation}
    if fc isa Forecast
        fc = [fc]
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    if by_horizon
        if horizon isa Vector{Int}
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(horizon)
        elseif isnothing(horizon)
            all_horizons = sort(unique(vcat((f -> f.horizon).(fc)...)))
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(all_horizons)
        end
    end
    fcVec = remove_trajectories.(fc)
    if horizon isa Int
        fcVec = fcVec[(f -> any(f.horizon .== horizon)).(fcVec)]
        fcVec = (f -> filter_horizons(f, [horizon])).(fcVec)
    elseif horizon isa Vector{Int}        
        fcVec = fcVec[(f -> length(intersect(f.horizon, horizon)) > 0).(fcVec)]
        fcVec = (f -> filter_horizons(f, horizon)).(fcVec)
    end
    if !isnothing(rule.levels)
        fcVec = (f -> filter_levels(f, rule.levels)).(fcVec)
    end

    fcVec = (f -> transform(f, transformation)).(fcVec)
    
    errors = vcat(all_forecast_errors(fcVec, CRPScore(), missing_strategy = rule.missing_strategy)...)
    
    mean(errors)
end

"""
    CRPS_trajectory(;levels::Union{Vector{Float64}, Nothing} = nothing,
                    missing_strategy::Symbol = :skip)

CRPS for trajectory-based probabilistic forecasts.

Computes exact CRPS using sample trajectories rather than interval approximation.

# Fields
- `levels::Vector{Float64}`: Not used (trajectories provide full distribution)
- `missing_strategy::Symbol`: Missing value handling

# Method
Uses empirical CDF from trajectory samples to compute exact CRPS:
1. **Empirical CDF**: F̂(x) = (1/n) ∑ᵢ 𝟙{trajectory_i ≤ x}
2. **Integration**: Numerical integration of squared CDF differences
3. **Exact Score**: No approximation error from interval-based methods
"""
struct CRPS_trajectory <: AbstractTrajectoryScoringRule
    levels::Union{Vector{Float64}, Nothing}
    missing_strategy::Symbol

    function CRPS_trajectory(;levels::Union{Vector{Float64}, Nothing} = nothing,
        missing_strategy::Symbol = :skip)
        if !(missing_strategy in [:error, :skip, :propagate])
            missing_strategy = :skip
        end
        new(levels, missing_strategy)
    end
end

"""
    score(fc::Union{Vector{Forecast}, Forecast}, rule::WIS; ...) -> Float64
    score(fc::Union{Vector{Forecast}, Forecast}, rule::CRPS; ...) -> Float64  
    score(fc::Union{Vector{Forecast}, Forecast}, rule::CRPS_trajectory; ...) -> Float64

Evaluate interval/trajectory forecasts using specified scoring rule.

# Arguments
Similar to point scoring with additional requirements:
- **WIS**: Requires forecasts with intervals and median
- **CRPS**: Requires forecasts with intervals (approximation)
- **CRPS_trajectory**: Requires forecasts with sample trajectories

# Level Filtering
When `levels` specified in scoring rule:
- Filters forecasts to include only specified confidence levels
- Ensures fair comparison across different interval methods
- Automatically handles missing levels (uses intersection of available levels)
"""
function score(fc::Union{Vector{Forecast}, Forecast},
        rule::CRPS_trajectory;
        horizon::Union{Int, Nothing, Vector{Int}, UnitRange{Int}} = nothing,
        transformation::T1 = NoTransform(),
        by_horizon::Bool = false) where {T1 <: AbstractDataTransformation}
    if fc isa Forecast
        fc = [fc]
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    if by_horizon
        if horizon isa Vector{Int}
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(horizon)
        elseif isnothing(horizon)
            all_horizons = sort(unique(vcat((f -> f.horizon).(fc)...)))
            return (h -> score(fc, rule, horizon = h, transformation = transformation)).(all_horizons)
        end
    end
    fcVec = fc[has_trajectories.(fc)]
    length(fcVec) > 0 || throw(ArgumentError("No forecast with trajectories"))
    if horizon isa Int
        fcVec = fcVec[(f -> any(f.horizon .== horizon)).(fcVec)]
        fcVec = (f -> filter_horizons(f, [horizon])).(fcVec)
    elseif horizon isa Vector{Int}        
        fcVec = fcVec[(f -> length(intersect(f.horizon, horizon)) > 0).(fcVec)]
        fcVec = (f -> filter_horizons(f, horizon)).(fcVec)
    end
    if !isnothing(rule.levels)
        fcVec = (f -> filter_levels(f, rule.levels)).(fcVec)
    end

    fcVec = (f -> transform(f, transformation)).(fcVec)
    
    errors = vcat(all_forecast_errors(fcVec, CRPScore_trajectory(), missing_strategy = rule.missing_strategy)...)
    
    mean(errors)
end
