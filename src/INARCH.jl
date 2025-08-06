"""
    INARCHModel(;p = 1, s = 0, k = 1, nb = false)

Integer-valued Autoregressive Conditional Heteroskedasticity model for count data.

Specialized model for non-negative integer-valued time series such as counts,
frequencies, or occurrence data. Captures autoregressive dynamics in the
conditional mean with optional seasonality and overdispersion.

# Fields
- `p::Int`: Autoregressive order (number of lagged counts)
- `s::Int`: Seasonal period (0 for no seasonality, default: 0)
- `k::Int`: Number of harmonic components for seasonality (default: 1)
- `nb::Bool`: Use negative binomial distribution instead of Poisson (default: false)

# Mathematical Model
λ_t = β₀ + ∑_{i=1}^p α_i X_{t-i} + seasonal_component_t

where:
- X_t ~ Poisson(λ_t) or NegativeBinomial(λ_t, ϕ) if nb=true
- λ_t is the conditional mean (intensity)
- Seasonal component uses harmonic functions when s > 0

# Seasonality Specification
When s > 0, seasonal component is:
∑_{j=1}^k [γ_{2j-1} sin(2πjt/s) + γ_{2j} cos(2πjt/s)]
"""
struct INARCHModel <: AbstractBaselineModel
    p::Int
    s::Int
    k::Int
    nb::Bool
    function INARCHModel(;p = 1, s = 0, k = 1, nb = false)
        (s >= 0) || throw(ArgumentError("Seasonality must be a non-negative integer"))
        if s == 0
            k = 0
        end
        (k >= 0) || throw(ArgumentError("Number of harmonic waves must be a positive integer"))
        if k == 0
            s = 0
        end
        new(p, s, k, nb)
    end
end

"""
    INARCHParameter(β₀, α, ϕ, γ)

Parameters for fitted INARCH model.

# Fields
- `β₀::Float64`: Intercept
- `α::Vector{Float64}`: Autoregressive coefficients for lagged counts
- `ϕ::Float64`: Overdispersion parameter (for negative binomial, Inf if Poisson)
- `γ::Vector{Float64}`: Seasonal harmonic coefficients (empty if no seasonality)

# Parameter Constraints
- β₀ > 0: Positive baseline intensity
- α_i ≥ 0: Non-negative autoregressive effects
- ∑α_i < 1: Stationarity condition
- ϕ > 0: Positive overdispersion (when nb=true)
"""
struct INARCHParameter <: AbstractModelParameters
    β0::Float64
    α::Vector{Float64}
    ϕ::Float64
    γ::Vector{Float64}
    function INARCHParameter(β0, α, ϕ, γ)
        if !(typeof(α) <: Vector) & (length(α) == 1)
            α = [α]
        end
        if length(α) == 0
            α = Float64[]
        end
        if length(γ) == 0
            γ = Float64[]
        end

        new(Float64(β0), Float64.(α), Float64.(ϕ), Float64.(γ))   
    end
end

"""
    INARCHEstimationSetting()

Estimation settings for INARCH model.

Empty struct as INARCH uses standard maximum likelihood estimation
with automatic constraint handling. Included for interface consistency.
"""
struct INARCHEstimationSetting <: AbstractEstimationSetting
end

"""
    INARCHFitted

Container for fitted INARCH model.

# Fields
- `x::Vector{Int}`: Original integer time series data
- `model::INARCHModel`: Model specification
- `par::INARCHParameter`: Estimated parameters  
- `estimation_setting::INARCHEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata

Note: Data must be integer-valued for INARCH modeling.
"""
struct INARCHFitted <: AbstractFittedModel
    x::Vector{Int}
    model::INARCHModel
    par::INARCHParameter
    estimation_setting::INARCHEstimationSetting
    temporal_info::TemporalInfo
    function INARCHFitted(x,
            model::INARCHModel,
            par::INARCHParameter,
            estimation_setting::INARCHEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo())
        #
        if !(x isa Vector{Int})
            x = Int.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

function inarchTF(θ, x::Vector{Int}, model::INARCHModel, returnPars::Bool = false)
    T = length(x)
    p = model.p
    s = model.s
    k = model.k
    nb = model.nb

    nPar = 1 + p + 2*k + nb

    β0 = θ[1]
    α = θ[2:1+p]
    ϕ =  Inf
    if nb
        ϕ = θ[2 + p]
    end
    γ = θ[2 + p + nb:end]
    if returnPars
        return INARCHParameter(β0, α, ϕ, γ)
    end
    if (β0 <= 0) | any(α .< 0) | any(α .> 1) | (sum(α) > 1) | (ϕ <= 0)
        return Inf
    end

    λ = fill(β0, T - p)
    seas = zeros(T)

    if s > 0
        tSeq = (1:T) ./ s .* (2π)
        @inbounds for kk in 1:k
            ωk = kk * tSeq
            seas .+= γ[kk] .* sin.(ωk) .+ γ[k + kk] .* cos.(ωk)
        end
    end
    seas = exp.(seas)
    xScl = x ./ seas

    @inbounds for i in 1:p
        λ .+= α[i] .* @view xScl[(p+1:T) .- i]
    end

    λ = λ .* seas[p+1:end]

    if nb
        pp = ϕ ./ (ϕ .+ λ)
        if any(.!(0 .< pp .< 1))
            return Inf
        end
        out = -sum(logpdf.(NegativeBinomial.(ϕ, pp), @view x[p+1:end]))
    else
        out = -sum(logpdf.(Poisson.(λ), @view x[p+1:end]))
    end

    out
end

"""
    fit_baseline(x::Vector{Int}, model::INARCHModel;
                setting::Union{INARCHEstimationSetting, Nothing} = INARCHEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> INARCHFitted
    fit_baseline(x::Vector{Real}, model::INARCHModel; ...) -> INARCHFitted

Fit INARCH model using maximum likelihood estimation.

# Arguments
- `x::Vector{Int}` or `Vector{Real}`: Integer time series data
- `model::INARCHModel`: Model specification
- `setting`: Estimation settings (optional)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `INARCHFitted`: Fitted model with estimated parameters

# Estimation Method
1. **Likelihood Construction**
2. **Parameter Constraints**: Automatic handling of positivity and stationarity
3. **Optimisation**: Uses constrained optimisation with penalty methods
4. **Initialisation**: Method-of-moments starting values
"""
function fit_baseline(x::Vector{Int},
        model::INARCHModel;
        setting::Union{INARCHEstimationSetting, Nothing} = INARCHEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo())
    if isnothing(setting)
        setting = INARCHEstimationSetting()
    end
    inits = [mean(x)/2; fill(0.5 ./ model.p, model.p); ifelse(model.nb, 3.0, zeros(0)); zeros(model.k*2)]
    optRaw = optimize(vars -> inarchTF(vars, x, model), inits)
    par = inarchTF(optRaw.minimizer, x, model, true)

    INARCHFitted(x, model, par, setting, temporal_info)
end

function fit_baseline(x::Vector{Real},
    model::INARCHModel;
    setting::Union{INARCHEstimationSetting, Nothing} = INARCHEstimationSetting(),
    temporal_info::TemporalInfo = TemporalInfo())
    if isnothing(setting)
        setting = INARCHEstimationSetting()
    end
    if all(isinteger.(x))
        return fit_baseline(Int64.(x), model, setting = setting, temporal_info = temporal_info)
    else
        throw(ArgumentError("Time series must be integer."))
    end
    inits = [mean(x)/2; fill(0.5 ./ model.p, model.p); ifelse(model.nb, 3.0, zeros(0)); zeros(model.k*2)]
    optRaw = optimize(vars -> inarchTF(vars, x, model), inits)
    par = inarchTF(optRaw.minimizer, x, model, true)

    INARCHFitted(x, model, par, setting, temporal_info)
end

"""
    point_forecast(fitted::INARCHFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts from fitted INARCH model.

# Arguments
- `fitted::INARCHFitted`: Fitted INARCH model
- `horizon`: Forecast horizons

# Returns
- `Vector{Float64}`: Point forecasts (conditional expectations)

# Forecasting Method
Computes conditional expectations recursively:
λ_{T+h} = β₀ + ∑_{i=1}^p α_i X̂_{T+h-i} + seasonal_component_{T+h}

where:
- Use historical data X_t for h ≤ i
- Use previous forecasts X̂_{T+h-i} for h > i
- Seasonal component computed from fitted harmonics
"""
function point_forecast(fitted::INARCHFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))
    
    x = fitted.x
    T = length(x)
    model = fitted.model
    p = model.p
    s = model.s
    k = model.k
    nb = model.nb

    β0 = fitted.par.β0
    α = fitted.par.α
    ϕ = fitted.par.ϕ
    γ = fitted.par.γ

    hMax = maximum(horizon)

    tSeq = (T+1-p:T+hMax) ./ s .* (2*π)
    seas = zeros(hMax + p)
    if s > 0
        for kk = 1:k
            seas .+= γ[kk] .* sin.(tSeq)
            seas .+= γ[k + kk] .* cos.(tSeq)
        end
    end
    
    seas = exp.(seas)
    xOld = x[end-p + 1:end] ./ seas[1:p]
    seas = seas[p+1:end]
    
    fc_point = zeros(hMax)

    for hh in 1:hMax
        fc_point[hh] = β0
        for i = 1:p
            if hh - i <= 0
                fc_point[hh] += xOld[p + hh - i] * α[i]
            else
                fc_point[hh] += fc_point[hh - i] .* α[i]
            end
        end

        fc_point[hh] *= seas[hh]
    end
    fc_point[horizon]
end

"""
    interval_forecast(fitted::INARCHFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Int64}, Nothing})

Generate prediction intervals using INARCH trajectory simulation.

# Arguments
- `fitted::INARCHFitted`: Fitted INARCH model
- `method::ModelTrajectoryInterval`: Trajectory simulation method
- Standard interval forecast arguments

# Returns
- Standard interval forecast tuple with integer trajectories

# Simulation Method
1. **Initialise**: Start from historical data and fitted parameters
2. **Recursive Simulation**: For each future period t:
   - Compute λ_t from autoregressive and seasonal components
   - Sample X_t ~ Poisson(λ_t) or NB(ϕ, p_t)
   - Use sampled value for future λ computations
3. **Repeat**: Generate multiple trajectory samples
4. **Quantiles**: Compute empirical quantiles across trajectories

# Note
No ParametricInterval implementation available.
"""
function interval_forecast(fitted::INARCHFitted,
        method::ModelTrajectoryInterval,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
        levels::Vector{Float64} = [0.95];
        alpha_precision::Int = 10,
        include_median::Bool = true)
    #
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

    # New below this line
    x = fitted.x
    T = length(x)
    model = fitted.model
    p = model.p
    s = model.s
    k = model.k
    nb = model.nb

    β0 = fitted.par.β0
    α = fitted.par.α
    ϕ = fitted.par.ϕ
    γ = fitted.par.γ

    hMax = maximum(horizon)

    trajectories = zeros(Int64, method.n_trajectories, hMax)
    tSeq = (T+1-p:T+hMax) ./ s .* (2*π)
    seas = zeros(hMax + p)
    if s > 0
        for kk = 1:k
            seas .+= γ[kk] .* sin.(tSeq)
            seas .+= γ[k + kk] .* cos.(tSeq)
        end
    end
    
    seas = exp.(seas)
    xOld = x[end-p + 1:end] ./ seas[1:p]
    seas = seas[p+1:end]

    for hh in 1:hMax
        λ = fill(β0, method.n_trajectories)
        for i = 1:p
            if hh - i <= 0
                λ .+= xOld[p + hh - i] * α[i]
            else
                λ .+= trajectories[:, hh - i] .* α[i]
            end
        end

        λ .*= seas[hh]

        if nb
            pp = ϕ ./ (ϕ .+ λ)
            trajectories[:, hh] = rand.(NegativeBinomial.(ϕ, pp))
        else
            trajectories[:, hh] = rand.(Poisson.(λ))
        end
    end

    trajectories = trajectories[:, horizon]

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