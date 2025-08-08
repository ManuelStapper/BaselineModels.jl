##########################
### Last Similar Dates ###
##########################

"""
    LSDModel(;s::Int = 1, w::Int = 0)

Last Similar Dates model for periodic forecasting.

Forecasts future values using historical observations from the same periodic
position, optionally including nearby dates within a window. Ideal for
time series with strong periodic patterns but limited trend.

# Fields
- `s::Int`: Periodicity (season length, default: 1)
- `w::Int`: Window size around similar dates (default: 0)

# Algorithm
For forecasting at time T+h:
1. **Identify Similar Dates**: Find all historical times t where t ≡ (T+h) mod s
2. **Apply Window**: Include dates t±w within the window
3. **Aggregate**: Use estimation function (default: mean) on similar observations

# Examples
```julia
# Weekly pattern (s=7), exact day matching
model = LSDModel(s=7, w=0)

# Monthly pattern with ±2 day window
model = LSDModel(s=30, w=2)  

# Quarterly pattern with ±1 period window
model = LSDModel(s=4, w=1)

# No periodicity (uses all historical data)
model = LSDModel(s=1, w=0)
```
"""
struct LSDModel <: AbstractBaselineModel
    s::Int
    w::Int
    function LSDModel(;s::Int = 1, w::Int = 0)
        ((w >= 0) & (w < s)) || throw(ArgumentError("Invalid window size"))
        (s >= 1) || throw(ArgumentError("Invalid periodicity"))

        new(s, w)
    end
end

"""
    LSDParameter(μ::Vector{Float64})

Parameters for fitted LSD model.

# Fields  
- `μ::Vector{Float64}`: Mean values for each periodic position (length s)

Each element μ[i] represents the average value for periodic position i,
computed from all historical observations at that position (±window).
"""
struct LSDParameter <: AbstractModelParrameters
    μ::Vector{Float64}
    function LSDParameter(μ)
        new(Float64.(μ))
    end
end

"""
    LSDEstimationSetting(;estimation_function::Function = mean)

Settings for LSD estimation.

# Fields
- `estimation_function::Function`: Aggregation function for similar dates (default: mean)

# Function Requirements
Must accept a vector and return a scalar. Common choices:
- `mean`: Average of similar dates (default)
- `median`: Robust central tendency
- `maximum`/`minimum`: Extreme value forecasting
- Custom functions for specialized aggregation

# Example
```julia
# Robust estimation using median
setting = LSDEstimationSetting(estimation_function = median)

# Conservative forecasting using 75th percentile
setting = LSDEstimationSetting(estimation_function = x -> quantile(x, 0.75))
```
"""
struct LSDEstimationSetting <: AbstractEstimationSetting
    estimation_function::Function
    function LSDEstimationSetting(;estimation_function::Function = mean)
        estimation_function(collect(1:10)) isa Real || throw(ArgumentError("estimation_function must have real-valued scalar output"))
        new(estimation_function)
    end
end

"""
    LSDFitted

Container for fitted LSD model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::LSDModel`: Model specification
- `par::LSDParameter`: Estimated periodic means
- `estimation_setting::LSDEstimationSetting`: Settings used  
- `temporal_info::TemporalInfo`: Temporal metadata

Parameter validation ensures μ vector length matches periodicity s.
"""
struct LSDFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::LSDModel
    par::LSDParameter
    estimation_setting::LSDEstimationSetting
    temporal_info::TemporalInfo
    function LSDFitted(x,
            model::LSDModel,
            par::LSDParameter,
            estimation_setting::LSDEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo())
        #
        (length(par.μ) == model.s) || throw(ArgumentError("Length of means must equal periodicity"))
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    fit_baseline(x::Vector{T1}, model::LSDModel;
                setting::Union{LSDEstimationSetting, Nothing} = LSDEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> LSDFitted

Fit LSD model by computing periodic averages.

# Arguments
- `x::Vector{T1}`: Time series data where T1 <: Real
- `model::LSDModel`: Model specification
- `setting`: Estimation settings (optional)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `LSDFitted`: Fitted model with periodic means

# Fitting Process
1. **Create Index Grid**: Generate indices for each periodic position
2. **Apply Windows**: Expand indices by ±w around each position
3. **Extract Observations**: Collect data for each windowed position
4. **Aggregate**: Apply estimation function to get μ[i] for each position i
```
"""
function fit_baseline(x::Vector{T1},
        model::LSDModel;
        setting::Union{LSDEstimationSetting, Nothing} = LSDEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo()) where {T1 <: Real}
    w = model.w
    s = model.s
    T = length(x)
    if isnothing(setting)
        setting = LSDEstimationSetting()
    end

    # Create a set of indices for s = 1 and then shift along the time series
    # Should first contain negative indices (up to -S+1)
    cen = collect(1:s:T+w)
    ind = vcat((cc -> cc-w:cc+w).(cen)...)
    μ = zeros(s)
    for ss = 1:s
        indS = ind .+ ss .- 1
        indS = indS[0 .< indS .<= T]
        μ[ss] = setting.estimation_function(x[indS])
    end

    LSDFitted(x, model, LSDParameter(μ), setting, temporal_info)
end

"""
    point_forecast(fitted::LSDFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts using periodic lookup.

# Arguments
- `fitted::LSDFitted`: Fitted LSD model
- `horizon`: Forecast horizons

# Returns
- `Vector{Float64}`: Point forecasts based on periodic means

# Forecasting Method
For each horizon h:
1. **Compute Position**: pos = (T + h) mod s, adjust for 1-indexing
2. **Lookup Mean**: forecast = μ[pos]
3. **Return Value**: Use stored periodic average

This cyclically repeats the fitted periodic pattern into the future.
"""
function point_forecast(fitted::LSDFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    tNew = length(fitted.x) .+ horizon
    ind = mod.(tNew, fitted.model.s)
    ind[ind .== 0] .= fitted.model.s
    fitted.par.μ[ind]
end

"""
    interval_forecast(fitted::LSDFitted, method::ParametricInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Nothing)

Generate parametric prediction intervals using historical quantiles.

# Arguments
- Standard interval forecast arguments

# Returns
- Standard interval forecast tuple (no trajectories)

# Method
1. **Group by Position**: Collect historical data for each periodic position
2. **Compute Quantiles**: Calculate empirical quantiles for each position  
3. **Forecast Lookup**: Use position-specific quantiles for prediction intervals
4. **Apply Corrections**: Handle positivity constraints if specified
"""
function interval_forecast(fitted::LSDFitted,
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

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))
    
    # Compute quantiles for S subsets, i.e. data that has been used for forecasting
    T = length(fitted.x)
    s = fitted.model.s
    w = fitted.model.w
    
    cen = collect(1:s:T+w)
    ind = vcat((cc -> cc-w:cc+w).(cen)...)

    all_quantiles = fill(Float64[], s)

    for ss = 1:s
        indS = ind .+ ss .- 1
        indS = indS[0 .< indS .<= T]
        all_quantiles[ss] = quantile(fitted.x[indS], alpha)
    end
    
    if method.positivity_correction == :post_clip
        for ss = 1:s
            all_quantiles[ss][all_quantiles[ss] .< 0] .= 0.0
        end
    end

    all_intervals = fill(ForecastInterval([1.0], [1.0], [0.5]), s)

    for ss = 1:s
        ls = reverse(all_quantiles[ss][alpha .< 0.5])
        us = all_quantiles[s][alpha .> 0.5]
        all_intervals[ss] = ForecastInterval(ls, us, levels[levels .> 0])
    end

    tNew = length(fitted.x) .+ horizon
    ind = mod.(tNew, s)
    ind[ind .== 0] .= s
    fc_intervals = all_intervals[ind]

    if include_median
        all_medians = (qq -> qq[findfirst(alpha .== 0.5)]).(all_quantiles)
        fc_median = all_medians[ind]
    end

    return fc_point, fc_median, fc_intervals, nothing
end

# ModelTrajectoryInterval by sampling from the relevant observations
"""
    interval_forecast(fitted::LSDFitted, method::ModelTrajectoryInterval,
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
1. **Group by Position**: Organise historical data by periodic position
2. **Bootstrap Sampling**: For each horizon, sample from corresponding position's data
3. **Generate Trajectories**: Create multiple independent sample paths
4. **Compute Quantiles**: Empirical quantiles from trajectory ensemble
5. **Apply Corrections**: Handle positivity constraints if specified

# Bootstrap Details
- Each periodic position maintains its own historical data pool
- Sampling is with replacement from position-specific observations
- No temporal correlation between horizons (independent sampling)
- Trajectories reflect empirical distribution at each position

# Positivity Correction
- `:truncate`: Remove negative observations from sampling pools
- `:zero_floor` / `:post_clip`: Set negative samples to zero
"""
function interval_forecast(fitted::LSDFitted,
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

    Random.seed!(method.seed)

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
    
    # Compute quantiles for S subsets, i.e. data that has been used for forecasting
    T = length(fitted.x)
    s = fitted.model.s
    w = fitted.model.w
    
    cen = collect(1:s:T+w)
    ind = vcat((cc -> cc-w:cc+w).(cen)...)

    all_observations = fill(Float64[], s)

    for ss = 1:s
        indS = ind .+ ss .- 1
        indS = indS[0 .< indS .<= T]
        all_observations[ss] = fitted.x[indS]
    end

    if method.positivity_correction == :truncate
        all_observations = (xx -> xx[xx .>= 0]).(all_observations)
    end

    if method.positivity_correction in [:zero_floor, :post_clip]
        for ss = 1:length(all_observations)
            all_observations[ss][all_observations[ss] .< 0] .= 0.0
        end
    end

    tNew = length(fitted.x) .+ horizon
    ind = mod.(tNew, s)
    ind[ind .== 0] .= s
    
    trajectories = zeros(method.n_trajectories, length(horizon))
    for i = 1:length(ind)
        trajectories[:, i] = sample(all_observations[ind[i]], method.n_trajectories)
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