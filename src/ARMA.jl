##################
### ARMA(p, q) ###
##################

"""
    ARMAModel(p::Int, q::Int, μ::Function, μDim::Int)
    ARMAModel(;p::Int, q::Int, s::Int = 0, trend::Bool = false)

Autoregressive Moving Average model of order (p, q).

The ARMA(p, q) model follows the specification:
X_t - μ_t = ϵ_t + ∑_{i=1}^p α_i (X_{t-i} - μ_{t-i}) + ∑_{i=1}^q β_i ϵ_{t-i}

where μ_t = μ(t, θ) is a deterministic trend/seasonal function.

# Arguments
- `p::Int`: Autoregressive order (number of lagged observations)
- `q::Int`: Moving average order (number of lagged forecast errors)
- `μ::Function`: Mean function μ(θ, t) where θ are parameters and t is time
- `μDim::Int`: Number of parameters in the mean function

# Convenience Constructor Arguments
- `p::Int`: Autoregressive order
- `q::Int`: Moving average order  
- `s::Int = 0`: Seasonal period (0 for no seasonality)
- `trend::Bool = false`: Include linear trend

# Examples
```julia
# ARMA(2,1) with constant mean
model = ARMAModel(p=2, q=1)

# ARMA(1,1) with trend
model = ARMAModel(p=1, q=1, trend=true)

# ARMA(2,1) with seasonal period 12
model = ARMAModel(p=2, q=1, s=12)

# Custom mean function
μ_func = (θ, t) -> θ[1] + θ[2]*cos(2π*t/365)
model = ARMAModel(1, 1, μ_func, 2)
```

# Mean Function Specifications
- **Constant**: μ(θ, t) = θ[1]
- **Trend**: μ(θ, t) = θ[1] + θ[2]*t  
- **Seasonal**: μ(θ, t) = θ[1] + θ[2]*sin(2π*t/s) + θ[3]*cos(2π*t/s)
- **Seasonal + Trend**: μ(θ, t) = θ[1] + θ[2]*sin(2π*t/s) + θ[3]*cos(2π*t/s) + θ[4]*t
"""
struct ARMAModel <: AbstractBaselineModel
    p::Int
    q::Int
    μ::Function
    μDim::Int
    function ARMAModel(p::Int, q::Int, μ::T1, μDim::Int) where {T1 <: Function}
        ((p >= 0) && (q >= 0)) || throw(ArgumentError("Model orders must be non-negative"))
        (μDim >= 0) || throw(ArgumentError("Number of trend parameters must be non-negative"))
        new(p, q, μ, μDim)
    end
end

function ARMAModel(;p::Int, q::Int, s::Int = 0, trend::Bool = false)
    if s == 0
        if trend
            μ = (θ, t) -> θ[1] + θ[2]*t
            μDim = 2
        else
            μ = (θ, t) -> θ[1]
            μDim = 1
        end
    else
        if trend
            μ = (θ, t) -> θ[1] + θ[2] * sin(t/s*2*π) + θ[3] * cos(t/s*2*π) + θ[4] * t
            μDim = 4
        else
            μ = (θ, t) -> θ[1] + θ[2] * sin(t/s*2*π) + θ[3] * cos(t/s*2*π)
            μDim = 3
        end
    end
    return ARMAModel(p, q, μ, μDim)
end

"""
    ARMAParameter(α, β, μ, σ²)

Parameters for fitted ARMA model.

# Fields
- `α::Vector{Float64}`: Autoregressive coefficients [α₁, α₂, ..., αₚ]
- `β::Vector{Float64}`: Moving average coefficients [β₁, β₂, ..., βₑ]  
- `μ::Vector{Float64}`: Mean function parameters
- `σ²::Float64`: Innovation variance (must be non-negative)
"""
struct ARMAParameter <: AbstractModelParameters
    α::Vector{Float64}
    β::Vector{Float64}
    μ::Vector{Float64}
    σ²::Float64
    function ARMAParameter(α, β, μ, σ²)
        (σ² >= 0) || throw(ArgumentError("Variance cannot be negative"))
        new(Float64.(α), Float64.(β), Float64.(μ), Float64(σ²))
    end
end

"""
    ARMAEstimationSetting(;ensure_stability::Bool = true)

Settings for ARMA model estimation.

# Fields
- `ensure_stability::Bool`: If true, constrains parameters to ensure stationarity and invertibility

When `ensure_stability = true`:
- AR parameters are constrained so all characteristic equation roots lie outside unit circle
- MA parameters are constrained so all characteristic equation roots lie outside unit circle
- This ensures the model is both stationary and invertible

When `ensure_stability = false`:
- No parameter constraints are imposed during estimation
- Model may be non-stationary or non-invertible
- Useful for exploratory analysis or when theory suggests non-stationarity
"""
struct ARMAEstimationSetting <: AbstractEstimationSetting
    ensure_stability::Bool
    function ARMAEstimationSetting(;ensure_stability::Bool = true)
        new(ensure_stability)
    end
end

"""
    ARMAFitted

Container for a fitted ARMA model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::ARMAModel`: Model specification
- `par::ARMAParameter`: Estimated parameters
- `estimation_setting::ARMAEstimationSetting`: Estimation settings used
- `temporal_info::TemporalInfo`: Temporal metadata

This struct contains all information needed for forecasting and analysis.
"""
struct ARMAFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::ARMAModel
    par::ARMAParameter
    estimation_setting::ARMAEstimationSetting
    temporal_info::TemporalInfo
    function ARMAFitted(x, model::ARMAModel, par::ARMAParameter,
            estimation_setting::ARMAEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo())
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    ARMAtoMA(α::Vector{Float64}, β::Vector{Float64}, iMax::Int = 100) -> Vector{Float64}

Convert ARMA(p,q) representation to MA(∞) representation.

# Arguments
- `α::Vector{Float64}`: AR coefficients [α₁, α₂, ..., αₚ]
- `β::Vector{Float64}`: MA coefficients [β₁, β₂, ..., βₑ]
- `iMax::Int = 100`: Maximum number of MA coefficients to compute

# Returns
- `Vector{Float64}`: MA(∞) coefficients [ψ₁, ψ₂, ..., ψ_{iMax}]

# Example
```julia
α = [0.5, -0.3]  # AR(2) coefficients
β = [0.2]        # MA(1) coefficient  
ψ = ARMAtoMA(α, β, 10)  # First 10 MA coefficients
```
"""
function ARMAtoMA(α::Vector{Float64},
                  β::Vector{Float64},
                  iMax::Int = 100)
    ψ = zeros(iMax)
    p = length(α)
    q = length(β)
    M = maximum([p, q+1])
    # Make α and β the same length
    a = [α; zeros(M - p)]
    b = [β; zeros(M - q)]
    for j = 1:M-1
        ψ[j] = b[j] + sum(a[1:j-1] .* ψ[j-1:-1:1]) + a[j]
    end
    for j = M:iMax
        if j == p
            ψ[j] =  sum(a[1:p-1] .* ψ[j-1:-1:1]) + a[p]
        else
            ψ[j] = sum(a[1:p] .* ψ[j-1:-1:j-p])
        end
    end
    return ψ
end

function ARMAθ2par(θ::Vector{Float64}, model::ARMAModel)
    p = model.p
    q = model.q
    ARMAParameter(θ[1:p], θ[p+1:p+q], θ[p+q+1:end-1], θ[end])
end

function ARMApar2θ(par::ARMAParameter, model::ARMAModel)
    [par.α; par.β; par.μ; par.σ²]
end

"""
    is_stationary(α::Vector{Float64}; tolerance::Float64 = 1e-10) -> Bool

Check if AR polynomial corresponds to a stationary process.

Tests whether all roots of the characteristic equation 1 - α₁z - α₂z² - ... - αₚzᵖ = 0
lie outside the unit circle.

# Arguments
- `α::Vector{Float64}`: AR coefficients
- `tolerance::Float64 = 1e-10`: Numerical tolerance for root magnitude checking

# Returns
- `Bool`: true if stationary, false otherwise
```
"""
function is_stationary(α::Vector{Float64}; tolerance::Float64 = 1e-10)
    p = length(α)
    p == 0 && return true
    if p == 1
        return abs(α[1]) < 1 - tolerance
    end

    coeffs = [1.0; -α]
    roots = Polynomials.roots(Polynomials.Polynomial(coeffs))
    return all(abs.(roots) .> 1 + tolerance)
end

"""
    is_invertible(β::Vector{Float64}; tolerance::Float64 = 1e-10) -> Bool

Check if MA polynomial corresponds to an invertible process.

Tests whether all roots of the characteristic equation 1 + β₁z + β₂z² + ... + βₑzᵍ = 0
lie outside the unit circle.

# Arguments
- `β::Vector{Float64}`: MA coefficients
- `tolerance::Float64 = 1e-10`: Numerical tolerance for root magnitude checking

# Returns
- `Bool`: true if invertible, false otherwise

# Example
```julia
β₁ = [0.3, -0.2]     # Typically invertible  
β₂ = [1.5, 0.1]      # Non-invertible
is_invertible(β₁)    # true
is_invertible(β₂)    # false
```
"""
function is_invertible(β::Vector{Float64}; tolerance::Float64 = 1e-10)
    q = length(β)
    q == 0 && return true
    if q == 1
        return abs(β[1]) < 1 - tolerance
    end
    coeffs = [1.0; β]
    roots = Polynomials.roots(Polynomials.Polynomial(coeffs))
    
    return all(abs.(roots) .> 1 + tolerance)
end

function check_stability(α::Vector{Float64}, β::Vector{Float64}; 
                              tolerance::Float64 = 1e-10)
    
    stationary = is_stationary(α; tolerance=tolerance)
    invertible = is_invertible(β; tolerance=tolerance)
    stationary && invertible
end

"""
    fit_baseline(x::Vector{T}, model::ARMAModel; 
                setting::Union{ARMAEstimationSetting, Nothing} = ARMAEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> ARMAFitted

Fit ARMA model to time series data using maximum likelihood estimation.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::ARMAModel`: ARMA model specification
- `setting::ARMAEstimationSetting`: Estimation settings (optional)
- `temporal_info::TemporalInfo`: Temporal metadata (optional)

# Returns
- `ARMAFitted`: Fitted model object containing estimated parameters and data

# Estimation Method
Uses conditional maximum likelihood estimation:
1. Computes residuals ϵ_t = X_t - μ_t - ∑αᵢ(X_{t-i} - μ_{t-i}) - ∑βⱼϵ_{t-j}
2. Maximises Gaussian likelihood L(θ) = ∏ᵢ φ(ϵᵢ; 0, σ²)
3. Optimisation via `Optim.jl` with starting values from method of moments

# Parameter Constraints
When `setting.ensure_stability = true`:
- AR parameters constrained for stationarity
- MA parameters constrained for invertibility
- Innovation variance σ² > 0

# Notes
- Uses conditional likelihood (assumes initial values are known)
- For short series, may be sensitive to starting values
"""
function fit_baseline(x::Vector{T1},
             model::ARMAModel;
             setting::Union{ARMAEstimationSetting, Nothing} = ARMAEstimationSetting(),
             temporal_info::TemporalInfo = TemporalInfo()) where {T1 <: Real}
    #
    if isnothing(setting)
        setting = ARMAEstimationSetting()
    end
    # (Negative) Log-likelihood function for optimisation
    function tf(θ, x, model, ensure_stability)
        # Only check for positive variance, other restrictions not imposed
        if θ[end] < 0
            return Inf
        end

        par = ARMAθ2par(θ, model)        
        if ensure_stability
            if !check_stability(par.α, par.β)
                return Inf
            end
        end

        p = model.p
        q = model.q
        M = maximum([p, q])
        T = length(x)

        m = (t -> model.μ(par.μ, t)).(1:T)
        y = x .- m
        
        yHat = zeros(T)
        yHat[1:M] = zeros(M)
        for t = M+1:T
            yHat[t] = sum(par.α .* y[t-1:-1:t-p]) .+ sum(par.β .* (y[t-1:-1:t-q] .- yHat[t-1:-1:t-q]))
        end
        d = Normal.(yHat, sqrt(par.σ²))
        -sum(logpdf.(d, y))
    end

    # No dependency for initialisation
    inits = [zeros(model.p + model.q); zeros(model.μDim); var(x)]
    est = optimize(vars -> tf(vars, x, model, setting.ensure_stability), inits).minimizer
    estPar = ARMAθ2par(est, model)

    ARMAFitted(x, model, estPar, setting, temporal_info)
end

"""
    point_forecast(fitted::ARMAFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts from fitted ARMA model.

# Arguments
- `fitted::ARMAFitted`: Fitted ARMA model
- `horizon::Union{Vector{Int}, Int, UnitRange{Int}}`: Forecast horizons

# Returns
- `Vector{Float64}`: Point forecasts for specified horizons

# Forecasting Method
Uses optimal linear predictor:
X̂_{T+h|T} = μ_{T+h} + ∑_{i=1}^p α_i (X̂_{T+h-i|T} - μ_{T+h-i}) + ∑_{j=1}^q β_j ϵ̂_{T+h-j}

where:
- Past observations used for h ≤ i  
- Past forecasts used for h > i
- Forecast errors ϵ̂_t available for h ≤ j, zero for h > j
"""
function point_forecast(fitted::ARMAFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    #
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    α = fitted.par.α
    β = fitted.par.β

    p = length(α)
    q = length(β)
    M = maximum([p, q])
    
    μ = fitted.model.μ
    σ² = fitted.par.σ²

    T = length(fitted.x)
    h = maximum(horizon)
    tSeq = 1:T+h

    m = (t -> μ(fitted.par.μ, t)).(tSeq)
    y = [fitted.x .- m[1:T]; zeros(h)]    
    yHat = zeros(T + h)

    for t = M+1:T+h
        yHat[t] = sum(α .* y[t-1:-1:t-p]) .+ sum(β .* (y[t-1:-1:t-q] .- yHat[t-1:-1:t-q]))
        if t > T
            y[t] = yHat[t]
        end
    end

    yHat[T .+ horizon] .+ m[T .+ horizon]
end

# Parametric forecast intervals using analytical ARMA prediction variance
"""
    interval_forecast(fitted::ARMAFitted, method::ParametricInterval, 
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Nothing)

Generate parametric prediction intervals for ARMA model using analytical formulas.

# Arguments
- `fitted::ARMAFitted`: Fitted ARMA model
- `method::ParametricInterval`: Parametric interval method specification
- `horizon::Union{Vector{Int}, Int, UnitRange{Int}}`: Forecast horizons  
- `levels::Vector{Float64}`: Confidence levels (e.g., [0.8, 0.95])
- `alpha_precision::Int = 10`: Decimal precision for quantile probabilities
- `include_median::Bool = true`: Whether to include median forecasts

# Returns
- `Tuple` containing:
  - Point forecasts
  - Median forecasts (if requested)
  - Prediction intervals  
  - Trajectories (always `nothing` for parametric method)

# Mathematical Details
Uses the analytical prediction variance formula for ARMA models:

Var(X_{T+h} - X̂_{T+h|T}) = σ² × (1 + ψ₁² + ψ₂² + ... + ψ_{h-1}²)

where ψⱼ are the MA(∞) representation coefficients obtained from `ARMAtoMA()`.

The prediction intervals are:
X̂_{T+h|T} ± z_{α/2} × √Var(X_{T+h} - X̂_{T+h|T})

where z_{α/2} is the standard normal quantile.

# Positivity Correction
When `method.positivity_correction = :post_clip`:
- Negative interval bounds are set to zero after computation
"""
function interval_forecast(fitted::ARMAFitted,
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
    all(0.0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))
    
    # Convert ARMA to MA(∞) representation for prediction variance
    ψ = ARMAtoMA(fitted.par.α, fitted.par.β, maximum(horizon) + 10)
    
    # Compute prediction variance for each horizon
    σ² = fitted.par.σ²
    pred_vars = zeros(length(horizon))
    
    for (i, h) in enumerate(horizon)
        # Prediction variance: σ² * (1 + ψ₁² + ψ₂² + ... + ψ_{h-1}²)
        if h == 1
            pred_vars[i] = σ²
        else
            pred_vars[i] = σ² * (1 + sum(ψ[1:h-1].^2))
        end
    end
    
    # Create intervals for each horizon
    fc_intervals = Vector{ForecastInterval}(undef, length(horizon))
    fc_median = include_median ? zeros(length(horizon)) : nothing
    
    if include_median
        i_median = findfirst(alpha .== 0.5)
    end
    
    for (i, h) in enumerate(horizon)
        # Normal quantiles scaled by prediction standard deviation
        pred_sd = sqrt(pred_vars[i])
        all_quantiles = fc_point[i] .+ quantile(Normal(0, 1), alpha) .* pred_sd
        
        if method.positivity_correction == :post_clip
            all_quantiles[all_quantiles .< 0] .= 0.0
        end
        
        ls = reverse(all_quantiles[alpha .< 0.5])
        us = all_quantiles[alpha .> 0.5]
        fc_intervals[i] = ForecastInterval(ls, us, levels[levels .> 0])
        
        if include_median
            fc_median[i] = all_quantiles[i_median]
        end
    end

    return fc_point, fc_median, fc_intervals, nothing
end

# Model trajectory intervals by sampling from ARMA process
"""
    interval_forecast(fitted::ARMAFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1], 
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate prediction intervals by simulating ARMA model trajectories.

# Arguments
- `fitted::ARMAFitted`: Fitted ARMA model
- `method::ModelTrajectoryInterval`: Trajectory simulation method
- `horizon::Union{Vector{Int}, Int, UnitRange{Int}}`: Forecast horizons
- `levels::Vector{Float64}`: Confidence levels  
- `alpha_precision::Int = 10`: Decimal precision for quantile probabilities
- `include_median::Bool = true`: Whether to include median forecasts

# Returns
- `Tuple` containing:
  - Point forecasts (conditional expectations)
  - Median forecasts (from trajectory quantiles)
  - Prediction intervals (from trajectory quantiles)
  - Sample trajectories (if `method.return_trajectories = true`)

# Simulation Method
1. **Initialise**: Start from final model state with historical data
2. **Generate innovations**: Sample ϵ_t ~ N(0, σ²) for each future period  
3. **Propagate dynamics**: Apply ARMA recursion with random innovations
4. **Repeat**: Generate `method.n_trajectories` independent sample paths
5. **Quantiles**: Compute empirical quantiles across trajectories

# Positivity Correction Options
- `:none`: No correction applied
- `:truncate`: Truncate innovation distribution at boundary  
- `:zero_floor`: Set negative values to zero during simulation
- `:post_clip`: Clip final trajectory values to non-negative
"""
function interval_forecast(fitted::ARMAFitted,
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
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))

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
    
    # Extract model parameters
    α = fitted.par.α
    β = fitted.par.β
    σ² = fitted.par.σ²
    μ_func = fitted.model.μ
    μ_pars = fitted.par.μ
    
    p = length(α)
    q = length(β)
    M = maximum([p, q])
    T = length(fitted.x)
    hMax = maximum(horizon)

    m_full = (t -> μ_func(μ_pars, t)).(1:T+hMax)
    y_full = [fitted.x .- m_full[1:T]; zeros(hMax)]
    yHat_full = zeros(T + hMax)
    d_error = Normal(0, sqrt(σ²))

    # Generate trajectories
    trajectories = zeros(method.n_trajectories, hMax)
    
    for i in 1:method.n_trajectories
        for t = M+1:T+hMax
            # Compute conditional expectation
            yHat_full[t] = sum(α .* y_full[t-1:-1:t-p]) .+ sum(β .* (y_full[t-1:-1:t-q] .- yHat_full[t-1:-1:t-q]))
            if t > T
                # Add innovation for future periods
                y_full[t] = makeStep(yHat_full[t] + m_full[t], d_error, method.positivity_correction == :truncate) - m_full[t]
                if method.positivity_correction == :zero_floor
                    if y_full[t] + m_full[t] < 0
                        y_full[t] = -m_full[t]
                    end
                end
                trajectories[i, t - T] = y_full[t] + m_full[t]
            end
        end
    end

    if method.positivity_correction == :post_clip
        trajectories[trajectories .< 0] .= 0.0
    end

    trajectories = trajectories[:, horizon]
    
    # Removing the seed
    Random.seed!(nothing)

    # Compute intervals from trajectories
    fc_intervals = Vector{ForecastInterval}(undef, length(horizon))
    fc_median = include_median ? zeros(length(horizon)) : nothing
    
    if include_median
        i_median = findfirst(alpha .== 0.5)
    end

    for i = 1:length(horizon)
        all_quantiles = [quantile(trajectories[:, i], q) for q = alpha]
        ls = reverse(all_quantiles[alpha .< 0.5])
        us = all_quantiles[alpha .> 0.5]
        fc_intervals[i] = ForecastInterval(ls, us, levels[levels .> 0])
    
        if include_median
            fc_median[i] = all_quantiles[i_median]
        end
    end

    if !method.return_trajectories
        trajectories = nothing
    end

    return fc_point, fc_median, fc_intervals, trajectories
end