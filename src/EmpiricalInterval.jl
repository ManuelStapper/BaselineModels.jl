"""
    make_step(d::ContinuousUnivariateDistribution, from::Float64, truncate::Bool, clip::Bool) -> Float64
    make_step(d::Vector{Float64}, from::Float64, truncate::Bool, clip::Bool) -> Float64

Generate next step in trajectory simulation with positivity corrections.

# Arguments
- `d`: Either a continuous distribution or vector of historical errors
- `from::Float64`: Current trajectory value
- `truncate::Bool`: If true, truncate error distribution at -from to ensure positivity
- `clip::Bool`: If true, set negative results to zero after sampling

# Returns
- `Float64`: Next trajectory value = from + sampled_error

# Positivity Handling
- **Truncate**: Sample from truncated distribution to prevent negative outcomes
- **Clip**: Allow negative sampling but clip final result to zero
- **Neither**: No positivity correction applied

Used internally by `interval_forecast` for trajectory-based prediction intervals.
"""
function make_step(d::ContinuousUnivariateDistribution, from::Float64, truncate::Bool, clip::Bool)
    if truncate
        return from + rand(truncated(d, -from, Inf))
    else
        out = from + rand(d)
        if clip && (out < 0)
            return 0.0
        else
            return out
        end
    end
end

function make_step(d::Vector{Float64}, from::Float64, truncate::Bool, clip::Bool)
    if truncate
        valid_errors = d[d .>= -from]
        if isempty(valid_errors)
            return from
        end
        return from + sample(d[d .>= -from])
    else
        if isempty(d)
            return from
        end
        out = from + sample(d)
        if clip && (out < 0)
            return 0.0
        else
            return out
        end
    end
end

"""
    interval_forecast(fitted::AbstractFittedModel, method::EmpiricalInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate prediction intervals using historical forecast errors.

# Arguments
- `fitted::AbstractFittedModel`: Any fitted forecasting model
- `method::EmpiricalInterval`: Empirical interval method specification
- `horizon`: Forecast horizons
- `levels::Vector{Float64}`: Confidence levels (e.g., [0.8, 0.95])
- `alpha_precision::Int = 10`: Decimal precision for quantile probabilities
- `include_median::Bool = true`: Whether to include median forecasts

# Returns
- `Tuple` containing:
  - Point forecasts from the fitted model
  - Median forecasts (from trajectory quantiles)
  - Prediction intervals (from trajectory quantiles)
  - Sample trajectories (if `method.return_trajectories = true`)

# Method Overview
1. **Historical Errors**: Compute past forecast errors using `historical_forecast_errors`
2. **Error Distribution**: Fit specified distribution or use raw errors
3. **Trajectory Simulation**: Generate future paths by sampling from error distribution
4. **Interval Construction**: Compute empirical quantiles across trajectories

# Error Sampling Strategies
- **Raw errors**: Direct bootstrap sampling from historical errors
- **Fitted distribution**: Sample from parametric distribution fitted to errors
- **Symmetry correction**: Include both positive and negative versions of errors
- **Stepwise vs. multi-step**: Use 1-step errors repeatedly for all horizons vs. h-step specific errors

# Positivity Correction Options
- `:none`: No correction
- `:truncate`: Truncate error distribution to prevent negative forecasts
- `:zero_floor`: Set negative trajectory values to zero during simulation
- `:post_clip`: Clip final trajectory values to non-negative

# Method Configuration
Key `EmpiricalInterval` parameters:
- `n_trajectories`: Number of simulation paths (default: 1000)
- `bootstrap_distribution`: Distribution to fit to errors (default: none)
- `symmetry_correction`: Include negative errors (default: false)
- `stepwise`: Use 1-step errors for all horizons (default: false)
- `min_observation`: Minimum observations for error computation

# Example
```julia
# Basic empirical intervals
method = EmpiricalInterval(n_trajectories=2000)
fc_point, fc_median, fc_intervals, _ = interval_forecast(fitted, method, 1:6)

# With fitted error distribution and positivity correction
method = EmpiricalInterval(
    bootstrap_distribution=Normal(),
    positivity_correction=:truncate,
    symmetry_correction=true
)
results = interval_forecast(fitted, method, 1:12, [0.8, 0.95])

# Access interval bounds
lower_95 = [interval.lower[end] for interval in results[3]]
upper_95 = [interval.upper[end] for interval in results[3]]
```

# Computational Notes
- Requires sufficient historical data for reliable error estimation
- Uses random seed from method for reproducibility
"""
function interval_forecast(fitted::AbstractFittedModel,
    method::EmpiricalInterval,
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
    
    # Setting seed if specified
    Random.seed!(method.seed)

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))

    # Compute all past forecasts here
    # In some settings we only need one horizon
    errors = historical_forecast_errors(fitted, ifelse(method.stepwise, [1], horizon), method.min_observation)
 
    # Account for symmetry_correction here
    if method.symmetry_correction
        for i = 1:length(errors)
            append!(errors[i], -errors[i])
        end
    end

    # Fit a bootstrap distribution here
    error_distribution = Vector{Union{Vector{Float64}, ContinuousUnivariateDistribution}}(errors)
    
    if method.bootstrap_distribution isa ContinuousUnivariateDistribution
        for i = 1:length(errors)
            error_distribution[i] = Distributions.fit(typeof(method.bootstrap_distribution), errors[i])
        end
    else
        isnothing(method.bootstrap_distribution) || @warn "Bootstrap distribution treated as `nothing`"
    end

    truncated = method.positivity_correction == :truncate
    clip = method.positivity_correction == :zero_floor

    # Generate trajectories here
    trajectories = zeros(method.n_trajectories, maximum(horizon))
    trajectories[:, 1] = [make_step(error_distribution[1], fc_point[1], truncated, clip) for _ in 1:method.n_trajectories]
    
    for i = 2:maximum(horizon)
        if method.stepwise
            trajectories[:, i] = [make_step(error_distribution[1], from, truncated, clip) for from = trajectories[:, i-1]]
        else
            trajectories[:, i] = [make_step(error_distribution[1], fc_point[i], truncated, clip) for _ in 1:method.n_trajectories]
        end
    end

    # Removing the seed
    Random.seed!(nothing)

    # Select only relevant horizons
    trajectories = trajectories[:, horizon]

    fc_intervals = Vector{ForecastInterval}(undef, length(horizon))
    fc_median = include_median ? zeros(length(horizon)) : nothing 

    if include_median
        i_median = findfirst(alpha .== 0.5)
    end

    for h = 1:length(horizon)
        all_quantiles = [quantile(trajectories[:, h], q) for q = alpha]
        ls = reverse(all_quantiles[alpha .< 0.5])
        us = all_quantiles[alpha .> 0.5]
        fc_intervals[h] = ForecastInterval(ls, us, levels[levels .> 0])
    
        if include_median
            fc_median[h] = all_quantiles[i_median]
        end
    end

    if !method.return_trajectories
        trajectories = nothing
    end

    return fc_point, fc_median, fc_intervals, trajectories
end