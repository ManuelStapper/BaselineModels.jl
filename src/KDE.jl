######################
### Kernel Density ###
######################

"""
    KDEModel()

Kernel Density Estimation model for non-parametric forecasting.

Estimates the empirical distribution of the time series and uses it for
forecasting. Assumes observations are drawn independently from an unknown
distribution.
"""
struct KDEModel <: AbstractBaselineModel end

"""
    KDEParameter(x_seq::StepRangeLen, density::Vector{Float64})

Parameters for fitted KDE model.

# Fields
- `x_seq::StepRangeLen`: Grid points for density evaluation
- `density::Vector{Float64}`: Estimated density values at grid points

Together these define the estimated probability density function as a
discrete approximation on the specified grid.
"""
struct KDEParameter <: AbstractModelParameters
    x_seq::StepRangeLen
    density::Vector{Float64}
    function KDEParameter(x_seq::StepRangeLen, density::Vector{Float64})
        length(x_seq) == length(density) || throw(ArgumentError("x sequence and density must have the same length"))
        new(x_seq, density)
    end
end

"""
    KDEEstimationSetting(;bandwidth_selection = KernelDensity.default_bandwidth,
                         kernel = Normal,
                         npoints = 2048,
                         boundary = KernelDensity.kde_boundary,
                         weights = KernelDensity.default_weights)

Settings for KDE estimation using KernelDensity.jl backend.

# Fields
- `bandwidth_selection`: Bandwidth selection method or fixed value
- `kernel`: Kernel distribution type (default: Normal)  
- `npoints::Int`: Number of grid points for density evaluation (default: 2048)
- `boundary`: Boundary handling method or fixed boundaries
- `weights`: Observation weights method or fixed weights

# Bandwidth Selection
- Function: Automatic selection (e.g., `default_bandwidth`, Silverman's rule)
- Scalar: Fixed bandwidth value

# Boundary Handling
- Function: Automatic boundary detection
- Tuple: Fixed (lower, upper) boundaries
- Important for bounded data (e.g., positive values)
"""
struct KDEEstimationSetting <: AbstractEstimationSetting
    bandwidth_selection::Function
    kernel::Type{<:Distribution}
    npoints::Int
    boundary::Function
    weights::Function
    
    function KDEEstimationSetting(;bandwidth_selection = KernelDensity.default_bandwidth,
        kernel = Normal,
        npoints = 2048,
        boundary = KernelDensity.kde_boundary,
        weights = KernelDensity.default_weights)
        
        # Convert scalar bandwidth to function
        if bandwidth_selection isa Real
            bw_value = bandwidth_selection
            bandwidth_selection = data -> bw_value
        end
        
        # Convert tuple boundary to function
        if boundary isa Tuple{Real, Real}
            boundary_tuple = boundary
            boundary = (data, bandwidth) -> boundary_tuple
        end
        
        # Convert vector weights to function
        if weights isa Vector{<:Real}
            weights_vec = weights
            weights = data -> length(weights_vec) == length(data) ? weights_vec : error("Weights length mismatch")
        end
        
        new(bandwidth_selection, kernel, npoints, boundary, weights)
    end
end

"""
    KDEFitted

Container for fitted KDE model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::KDEModel`: Model specification
- `par::KDEParameter`: Estimated density parameters
- `estimation_setting::KDEEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata
"""
struct KDEFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::KDEModel
    par::KDEParameter
    estimation_setting::KDEEstimationSetting
    temporal_info::TemporalInfo
    function KDEFitted(x,
            model::KDEModel,
            par::KDEParameter,
            estimation_setting::KDEEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo())
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    fit_baseline(x::Vector{T}, model::KDEModel;
                setting::Union{KDEEstimationSetting, Nothing} = KDEEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> KDEFitted

Fit KDE model by estimating empirical density.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::KDEModel`: Model specification
- `setting`: KDE estimation settings (optional)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `KDEFitted`: Fitted model with estimated density

# Estimation Process
1. **Configure KDE**: Apply bandwidth selection and boundary handling
2. **Compute Density**: Use KernelDensity.jl with specified kernel
3. **Store Results**: Extract density values and evaluation grid
4. **Normalise**: Ensure density integrates to 1

# Example
```julia
# Basic KDE fit
fitted = fit_baseline(data, KDEModel())

# Custom settings for positive data
setting = KDEEstimationSetting(boundary = (0.0, Inf))
fitted = fit_baseline(positive_data, KDEModel(), setting = setting)
```
"""
function fit_baseline(x::Vector{T},
        model::KDEModel;
        setting::Union{KDEEstimationSetting, Nothing} = KDEEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    if isnothing(setting)
        setting = KDEEstimationSetting()
    end
    dens = kde(x,
        bandwidth = setting.bandwidth_selection(x),
        kernel = setting.kernel,
        npoints = setting.npoints,
        boundary = setting.boundary(x, setting.bandwidth_selection(x)),
        weights = setting.weights(x))
    par = KDEParameter(dens.x, dens.density)
    KDEFitted(x, model, par, setting, temporal_info)
end

"""
    point_forecast(fitted::KDEFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts using density mean.

# Arguments
- `fitted::KDEFitted`: Fitted KDE model
- `horizon`: Forecast horizons (all return same value)

# Returns
- `Vector{Float64}`: Point forecasts (all equal to density mean)

# Method
Computes mean of estimated density: E[X] = ∫ x f̂(x) dx

Uses numerical integration over the density grid.
Since KDE assumes independence, all forecasts equal the distributional mean.

# Example
```julia
forecasts = point_forecast(fitted, 1:10)
# All values equal to estimated mean
```
"""
function point_forecast(fitted::KDEFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    integrand = fitted.par.x_seq .* fitted.par.density
    
    mean_est = 0.0
    for i in 2:length(fitted.par.x_seq)
        dx = fitted.par.x_seq[i] - fitted.par.x_seq[i-1]
        mean_est += 0.5 * dx * (integrand[i-1] + integrand[i])
    end
    fill(mean_est, length(horizon))
end

"""
    quantile_from_kde(par::KDEParameter, q::Union{Float64, Vector{Float64}}) -> Union{Float64, Vector{Float64}}

Extract quantiles from fitted KDE.

# Arguments
- `par::KDEParameter`: KDE parameters with density estimates
- `q`: Quantile level(s) in (0,1)

# Returns
- Quantile value(s) corresponding to specified level(s)

Used internally for parametric interval construction.
"""
function quantile_from_kde(par::KDEParameter, q::Union{Float64, Vector{Float64}})
    # Remove zero density
    keep = par.density .> 0
    xx = par.x_seq[keep]
    yy = par.density[keep]
    # Remove small differences in x
    keep = [1; findall(diff(xx) .> 1e-10) .+ 1]
    xx = xx[keep]
    yy = yy[keep]
    # Add padding at lower end and scale density
    xx = [par.x_seq[1] - (par.x_seq[2] - par.x_seq[1]); par.x_seq]
    yy = cumsum([0.0; yy])./sum(yy)
    # Linear interpolation
    if q isa Float64
        return quantile_from_kde(par, [q])
    end
    nq = length(q)
    out = zeros(nq)
    ix = 1
    iq = 1
    while true
        if q[iq] <= yy[ix + 1]
            out[iq] = xx[ix] + (yy[ix + 1] - q[iq])/(yy[ix + 1] - yy[ix])*(xx[ix + 1] - xx[ix])
            iq += 1
            if iq > nq
                return out
            end
        else
            ix += 1
        end
    end
end

"""
    interval_forecast(fitted::KDEFitted, method::ParametricInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Nothing)

Generate parametric prediction intervals from KDE.

# Arguments
- Standard interval forecast arguments

# Returns
- Standard interval forecast tuple (no trajectories)

# Method
1. **Compute Quantiles**: Extract quantiles from estimated density
2. **Replicate Intervals**: Same intervals for all horizons
3. **Apply Corrections**: Handle positivity if specified

Since KDE assumes independence, prediction intervals are identical
across all forecast horizons and equal to distributional quantiles.

# Example
```julia
# 80% and 95% prediction intervals
fc_point, fc_median, fc_intervals, _ = interval_forecast(
    fitted, ParametricInterval(), 1:6, [0.8, 0.95]
)
```
"""
function interval_forecast(fitted::KDEFitted,
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
    all_quantiles = quantile_from_kde(fitted.par, alpha)
    
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

"""
    rand_from_kde(par::KDEParameter, n::Union{Int, Tuple{Int, Int}}, lower_bound::Float64 = -Inf) -> Union{Vector{Float64}, Matrix{Float64}}
    rand_from_kde(par::KDEParameter, lower_bound::Float64 = -Inf) -> Float64

Sample from fitted KDE distribution.

# Arguments
- `par::KDEParameter`: KDE parameters
- `n`: Number of samples or (nrows, ncols) for matrix
- `lower_bound`: Optional lower bound for truncation

# Returns
- Random sample(s) from estimated distribution

# Method
1. **Weighted Sampling**: Sample grid points using density as weights
2. **Local Perturbation**: Add uniform noise within grid intervals
3. **Boundary Handling**: Apply truncation if lower bound specified

Used internally for trajectory-based interval forecasting.
"""
function rand_from_kde(par::KDEParameter, n::Union{Int, Tuple{Int, Int}}, lower_bound::Float64 = -Inf)
    delta = par.x_seq[2] - par.x_seq[1]
    weights = copy(par.density)
    weights[par.x_seq .< lower_bound + delta/2] .= 0.0
    wsample(par.x_seq, weights, n) .+ rand(Uniform(), n) .* delta .- delta/2
end

function rand_from_kde(par::KDEParameter, lower_bound::Float64 = -Inf)
    rand_from_kde(par, 1, lower_bound)[1]
end

"""
    interval_forecast(fitted::KDEFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate trajectory-based prediction intervals from KDE.

# Arguments
- Standard interval forecast arguments with trajectory method

# Returns
- Standard interval forecast tuple with optional trajectories

# Simulation Method
1. **Sample Trajectories**: Draw independent samples from estimated density
2. **Replicate Across Horizons**: Same distribution for all forecast periods
3. **Compute Quantiles**: Empirical quantiles from trajectory samples
4. **Apply Corrections**: Handle positivity constraints if specified

# Example
```julia
method = ModelTrajectoryInterval(n_trajectories=5000, return_trajectories=true)
fc_point, fc_median, fc_intervals, trajectories = interval_forecast(
    fitted, method, 1:12, [0.9, 0.95]
)
```
"""
function interval_forecast(fitted::KDEFitted,
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
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))

    trajectories = rand_from_kde(fitted.par,
        (method.n_trajectories, length(horizon)),
        ifelse(method.positivity_correction == :truncate, 0.0, -Inf))

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