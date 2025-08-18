#################
### OLS Model ###
#################

"""
    OLSModel(;p::Int = 3, d::Int = 1)

Ordinary Least Squares model for polynomial trend forecasting.

Fits polynomial trends to recent observations using linear regression.

# Fields
- `p::Int`: Number of recent observations to use for fitting (default: 3)
- `d::Int`: Polynomial degree (1=linear, 2=quadratic, 3=cubic, etc., default: 1)

# Mathematical Model
For the most recent p observations, fits:
y_t = β₀ + β₁t + β₂t² + ... + βₐtᵈ + ε_t

where t represents relative time positions within the fitting window.
"""
struct OLSModel <: AbstractBaselineModel
    p::Int # Temporal lag
    d::Int # Dimension (1 = linear, 2 = square, ...)
    function OLSModel(;p::Int = 3, d::Int = 1)
        p >= 1 + d || throw(ArgumentError("Insufficient temporal lag"))
        new(p, d)
    end
end

"""
    OLSParameter(β::Vector{Float64})

Parameters for fitted OLS model.

# Fields
- `β::Vector{Float64}`: Polynomial coefficients [β₀, β₁, β₂, ..., βₐ]
"""
struct OLSParameter <: AbstractModelParameters
    β::Vector{Float64}
    function OLSParameter(β)
        new(Float64.(β))
    end
end

"""
    OLSEstimationSetting()

Estimation settings for OLS model.

Empty struct as OLS uses standard least squares estimation with
no additional configuration options.
"""
struct OLSEstimationSetting <: AbstractEstimationSetting
end

"""
    OLSFitted

Container for fitted OLS model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::OLSModel`: Model specification
- `par::OLSParameter`: Estimated polynomial coefficients
- `estimation_setting::OLSEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata
"""
struct OLSFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::OLSModel
    par::OLSParameter
    estimation_setting::OLSEstimationSetting
    temporal_info::TemporalInfo

    function OLSFitted(x, model::OLSModel,
            par::OLSParameter,
            estimation_setting::OLSEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo())
        #
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

# Notation: Last observation has time index 0 for regression
"""
    fit_baseline(x::Vector{T}, model::OLSModel;
                setting::Union{OLSEstimationSetting, Nothing} = OLSEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> OLSFitted

Fit OLS model using least squares on recent observations.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::OLSModel`: Model specification
- `setting`: Estimation settings (optional, no effect)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `OLSFitted`: Fitted model with estimated polynomial coefficients

# Fitting Process
1. **Extract Data**: Use last min(p, length(x)) observations
2. **Construct Design Matrix**: Create polynomial time terms
3. **Least Squares**: Solve normal equations β̂ = (X'X)⁻¹X'y
4. **Store Results**: Save coefficients and data
"""
function fit_baseline(x::Vector{T},
             model::OLSModel;
             setting::Union{OLSEstimationSetting, Nothing} = OLSEstimationSetting(),
             temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    #
    if isnothing(setting)
        setting = OLSEstimationSetting()
    end
    pp = minimum([model.p, length(x)])
    d = model.d

    y = x[end-pp + 1:end]
    X = zeros(pp, 1 + d)
    X[:, 1] .= 1.0
    tSeq = 1 - pp:0
    for i = 1:d
        X[:, 1 + i] = tSeq .^ i
    end
    β = inv(X'X)*X'y
    par = OLSParameter(β)
    OLSFitted(x, model, par, setting, temporal_info)
end

"""
    point_forecast(fitted::OLSFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts using polynomial extrapolation.

# Arguments
- `fitted::OLSFitted`: Fitted OLS model
- `horizon`: Forecast horizons

# Returns
- `Vector{Float64}`: Point forecasts from polynomial trend
"""
function point_forecast(fitted::OLSFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))
    
    d = fitted.model.d
    X = zeros(length(horizon), 1 + d)
    X[:, 1] .= 1.0
    for i = 1:d
        X[:, 1 + i] = horizon .^ i
    end

    X*fitted.par.β
end

# Parametric intervals
"""
    interval_forecast(fitted::OLSFitted, method::ParametricInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Nothing)

Generate parametric prediction intervals using t-distribution.

# Arguments
- Standard interval forecast arguments

# Returns
- Standard interval forecast tuple (no trajectories)

# Method
Uses standard linear regression prediction intervals:
1. **Residual Variance**: σ̂² = RSS/(p-d-1) from fitted model
2. **Prediction Variance**: Var(X̂_{T+h}) = σ̂² × (x_h'(X'X)⁻¹x_h)
3. **t-Distribution**: Uses t_{p-d-1} distribution for finite sample correction
4. **Intervals**: X̂_{T+h} ± t_{α/2,p-d-1} × √Var(X̂_{T+h})
"""
function interval_forecast(fitted::OLSFitted,
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

    β = fitted.par.β
    p = minimum([fitted.model.p, length(fitted.x)])
    d = fitted.model.d

    X = zeros(p, 1 + d)
    X[:, 1] .= 1.0
    tSeq = 1 - p:0
    for i = 1:d
        X[:, 1 + i] = tSeq .^ i
    end

    y = fitted.x[end-p + 1:end]

    Xnew = zeros(length(horizon), 1 + d)
    Xnew[:, 1] .= 1.0
    for i = 1:d
        Xnew[:, 1 + i] = horizon .^ i
    end

    σ = sqrt(sum((y .- X*β).^2)/(p - d - 1))
    tqVec = quantile(TDist(p - d - 1), alpha)
    XXinv = inv(X'X)

    all_quantiles = fill(Float64[], length(horizon))
    for i = 1:length(horizon)
        all_quantiles[i] = fc_point[i] .+ tqVec .* (σ * sqrt((Xnew[i, :]'*XXinv*Xnew[i, :])[1]))
    end
    
    if method.positivity_correction == :post_clip
        for i = 1:length(horizon)
            all_quantiles[i][all_quantiles[i] .< 0] .= 0
        end
    end
    fc_intervals = fill(ForecastInterval([1.0], [1.0], [0.5]), length(horizon))
    for i = 1:length(horizon)
        ls = reverse(all_quantiles[i][alpha .< 0.5])
        us = all_quantiles[i][alpha .> 0.5]
        fc_intervals[i] = ForecastInterval(ls, us, levels[levels .> 0])
    end

    if include_median
        fc_median = (qq -> qq[findfirst(alpha .== 0.5)]).(all_quantiles)
    end

    return fc_point, fc_median, fc_intervals, nothing
end

"""
    makeStep(from::Float64, distStep::Normal{Float64}, truncate::Bool = false) -> Float64

Generate next step in trajectory simulation with optional truncation.

Helper function for trajectory-based interval forecasting. Adds Gaussian
innovation to current value with optional positivity constraint.

# Arguments
- `from::Float64`: Current trajectory value
- `distStep::Normal{Float64}`: Innovation distribution N(0,σ̂)
- `truncate::Bool`: If true, truncate at -from to ensure non-negative result

# Returns
- `Float64`: Next trajectory value = from + innovation
"""
function makeStep(from::Float64, distStep::Normal{Float64}, truncate::Bool = false)
    if truncate
        dist2 = truncated(distStep, -from, Inf)
    else
        dist2 = distStep
    end
    from + rand(dist2)
end

"""
    interval_forecast(fitted::OLSFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate trajectory-based prediction intervals using simulation.

# Arguments
- Standard interval forecast arguments with trajectory method

# Returns
- Standard interval forecast tuple with optional trajectories

# Simulation Method
1. **Initialise**: Start from polynomial trend forecasts
2. **Innovation Distribution**: N(0, σ̂²) from residual variance
3. **Path Generation**: Add random innovations to trend paths
4. **Positivity Handling**: Apply correction methods if specified
5. **Quantiles**: Compute empirical quantiles from trajectory ensemble

# Positivity Correction
- `:none`: No correction (innovations can produce negative values)
- `:truncate`: Truncate innovation distribution to prevent negative outcomes
- `:zero_floor`: Set negative trajectory values to zero
- `:post_clip`: Clip final trajectory results to non-negative
"""
function interval_forecast(fitted::OLSFitted,
    method::ModelTrajectoryInterval,
    horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
    levels::Vector{Float64} = [0.95];
    alpha_precision::Int = 10,
    include_median::Bool = true)

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
    fc_point = point_forecast(fitted, maximum(horizon))
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))

    β = fitted.par.β
    p = minimum([fitted.model.p, length(fitted.x)])
    d = fitted.model.d

    X = zeros(p, 1 + d)
    X[:, 1] .= 1.0
    tSeq = 1 - p:0
    for i = 1:d
        X[:, 1 + i] = tSeq .^ i
    end

    y = fitted.x[end-p + 1:end]

    σ = sqrt(sum((y .- X*β).^2)/(p - d - 1))
    # Add fallback if σ is 0
    if σ < 0.00001
        σ = 0.00001
    end
    d_error = Normal(0, σ)

    fc_point_pos = copy(fc_point)
    if method.positivity_correction == :zero_floor
        fc_point_pos[fc_point_pos .< 0] .= 0.0
    end

    trajectories = zeros(method.n_trajectories, maximum(horizon))
    
    first_from = maximum([fc_point[1], 0.0])
    trajectories[:, 1] = makeStep.(fill(first_from, method.n_trajectories), d_error, method.positivity_correction == :truncate)
    if method.positivity_correction == :zero_clip
        trajectories[trajectories[:, 1] .< 0, 1] .= 0.0
    end

    for h = 2:maximum(horizon)
        trajectories[:, h] = makeStep.(trajectories[:, h-1], d_error, method.positivity_correction == :truncate)
        if method.positivity_correction == :zero_clip
            trajectories[trajectories[:, h] .< 0, h] .= 0.0
        end
    end

    if method.positivity_correction in [:zero_floor, :post_clip]
        trajectories[trajectories .< 0] .= 0.0
    end

    trajectories = trajectories[:, horizon]
    fc_point = fc_point[horizon]

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