#########################
### Marginal Forecast ###
#########################

"""
    MarginalModel(;p::Int = Int(1e10))

Marginal distribution model for stationary forecasting.

Estimates the marginal distribution of the time series and uses it for
forecasting. Assumes observations are independently drawn from a stationary
distribution, ignoring temporal dependence. Provides distributional forecasts
based on historical empirical distribution.

# Fields
- `p::Int`: Maximum number of recent observations to use (default: 10^10, effectively all data)

When p < length(data), uses only the most recent p observations for estimation.
This allows adaptation to potential distributional changes over time.

# Mathematical Foundation
Assumes X_t ~ F for all t, where F is an unknown stationary distribution.
Estimates F using empirical distribution of historical observations.
"""
struct MarginalModel <: AbstractBaselineModel
    p::Int
    function MarginalModel(;p::Int = Int(1e10))
        p > 0 || throw(ArgumentError("Order 'p' must be positive"))
        new(p)
    end
end

"""
    MarginalParameter(μ::Real)

Parameter for fitted marginal model.

# Fields
- `μ::Float64`: Estimated marginal mean

This single parameter represents the center of the marginal distribution.
All point forecasts equal this value since no temporal structure is modeled.
"""
struct MarginalParameter <: AbstractModelParameters
    μ::Float64
    function MarginalParameter(μ)
        μ isa Real || throw(ArgumentError("μ must be a real-valued scalar"))
        new(Float64(μ))
    end
end

"""
    MarginalEstimationSetting(;estimation_function::Function = mean)

Settings for marginal distribution estimation.

# Fields
- `estimation_function::Function`: Function to compute location parameter (default: mean)

# Function Requirements
Must accept a vector and return a scalar. Common choices:
- `mean`: Sample mean (default, optimal for symmetric distributions)
- `median`: Robust location estimate (better for skewed/outlier-prone data)
- `mode`: Most frequent value approximation
- Custom quantiles: `x -> quantile(x, 0.3)` for specific quantiles
"""
struct MarginalEstimationSetting <: AbstractEstimationSetting
    estimation_function::Function

    function MarginalEstimationSetting(;estimation_function::Function = mean)
        estimation_function(collect(1:10)) isa Real || throw(ArgumentError("estimation_function must have real-valued scalar output"))
        new(estimation_function)
    end
end

"""
    MarginalFitted

Container for fitted marginal model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::MarginalModel`: Model specification
- `par::MarginalParameter`: Estimated parameter (location)
- `estimation_setting::MarginalEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata

Contains complete information for distributional forecasting based on
the empirical marginal distribution.
"""
struct MarginalFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::MarginalModel
    par::MarginalParameter
    estimation_setting::MarginalEstimationSetting
    temporal_info::TemporalInfo

    function MarginalFitted(x,
            model::MarginalModel,
            par::MarginalParameter,
            estimation_setting::MarginalEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo())
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    fit_baseline(x::Vector{T}, model::MarginalModel;
                setting::Union{MarginalEstimationSetting, Nothing} = MarginalEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> MarginalFitted

Fit marginal model by estimating distribution parameters.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::MarginalModel`: Model specification
- `setting`: Estimation settings (optional)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `MarginalFitted`: Fitted model with estimated marginal mean

# Fitting Process
1. **Select Data**: Use most recent min(p, length(x)) observations
2. **Estimate Location**: Apply estimation_function to selected data
3. **Store Parameters**: Save location parameter and full data for intervals
"""
function fit_baseline(x::Vector{T},
        model::MarginalModel;
        setting::Union{MarginalEstimationSetting, Nothing} = MarginalEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    pp = minimum([model.p, length(x)])
    if isnothing(setting)
        setting = MarginalEstimationSetting()
    end
    est = setting.estimation_function(x[end-pp + 1:end])
    MarginalFitted(x, model, MarginalParameter(est), setting, temporal_info)
end

"""
    point_forecast(fitted::MarginalFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts using marginal mean.

# Arguments
- `fitted::MarginalFitted`: Fitted marginal model
- `horizon`: Forecast horizons (all return same value)

# Returns
- `Vector{Float64}`: Point forecasts (all equal to estimated marginal mean)

# Method
All forecasts equal the fitted parameter: X̂_{T+h} = μ̂ for all h > 0

Since no temporal dependence is modeled, forecast horizon is irrelevant.
"""
function point_forecast(fitted::MarginalFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    fill(fitted.par.μ, length(horizon))
end

"""
    interval_forecast(fitted::MarginalFitted, method::ParametricInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Nothing)

Generate parametric prediction intervals using empirical quantiles.

# Arguments
- Standard interval forecast arguments

# Returns
- Standard interval forecast tuple (no trajectories)

# Method
1. **Extract Data**: Use same data subset as in fitting (last p observations)
2. **Compute Quantiles**: Calculate empirical quantiles at specified levels
3. **Replicate Intervals**: Same intervals for all horizons
4. **Apply Corrections**: Handle positivity constraints if specified
"""
function interval_forecast(fitted::MarginalFitted,
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
    pp = minimum([fitted.model.p, length(fitted.x)])

    # Validate input:
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))
    all_quantiles = quantile(fitted.x[end-pp+1:end], alpha)
    
    if method.positivity_correction == :post_clip
        all_quantiles[all_quantiles .< 0] .= 0
    end

    ls = reverse(all_quantiles[alpha .< 0.5])
    us = all_quantiles[alpha .> 0.5]
    fc_intervals = fill(ForecastInterval(ls, us, levels[levels .> 0]), length(horizon))

    if include_median
        fc_median = fill(all_quantiles[findfirst(alpha .== 0.5)], length(horizon))
    end

    return fc_point, fc_median, fc_intervals, nothing
end

# And model trajectory intervals
"""
    interval_forecast(fitted::MarginalFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate trajectory-based prediction intervals using bootstrap sampling.

# Arguments
- Standard interval forecast arguments with trajectory method

# Returns
- Standard interval forecast tuple with optional trajectories

# Simulation Method
1. **Data Pool**: Use same data subset as in fitting
2. **Bootstrap Sampling**: Sample with replacement for each trajectory
3. **Generate Matrix**: Create independent samples for each horizon
4. **Compute Quantiles**: Empirical quantiles from trajectory ensemble
5. **Apply Corrections**: Handle positivity constraints if specified

# Positivity Correction
- `:truncate`: Remove negative values from sampling pool
- `:zero_floor` / `:post_clip`: Set negative samples to zero
"""
function interval_forecast(fitted::MarginalFitted,
    method::ModelTrajectoryInterval,
    horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
    levels::Vector{Float64} = [0.95];
    alpha_precision::Int = 10,
    include_median::Bool = true)

    pp = minimum([fitted.model.p, length(fitted.x)])
    xx = fitted.x[end-pp+1:end]
    Random.seed!(method.seed)

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

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))

    if method.positivity_correction == :truncate
        any(xx .>= 0) || error("Time series must have positive elements for positivity correction by truncation")
        trajectories = sample(xx[xx .>= 0], (method.n_trajectories, length(horizon)))
    else
        trajectories = sample(xx, (method.n_trajectories, length(horizon)))
    end

    if method.positivity_correction in [:post_clip, :zero_floor]
        trajectories[trajectories .< 0] .= 0.0
    end

    # Removing the seed
    Random.seed!(nothing)

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