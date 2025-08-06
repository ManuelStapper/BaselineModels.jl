"""
    STLModel(;s::Int = 1)

Seasonal and Trend decomposition using Loess model.

Implements STL (Seasonal and Trend decomposition using Loess) for 
non-parametric decomposition of time series into seasonal, trend, 
and remainder components.

# Fields
- `s::Int`: Seasonal period (must be ≥ 1, default: 1 for non-seasonal)

# Mathematical Framework
Decomposes time series as: X_t = S_t + T_t + R_t
where:
- S_t: Seasonal component (periodic with period s)
- T_t: Trend component (smooth, slowly varying)
- R_t: Remainder component (irregular fluctuations)

# Algorithm Overview
STL uses iterative application of LOESS smoothing:
1. **Seasonal Extraction**: Smooth cycle-subseries to extract seasonal pattern
2. **Trend Estimation**: Smooth seasonally adjusted series for trend
3. **Iteration**: Repeat with robust weights for outlier handling
"""
struct STLModel <: AbstractBaselineModel
    s::Int
    
    function STLModel(;s::Int = 1)
        s >= 0 || throw(ArgumentError("Periodicity must be positive"))
        new(s)
    end
end

"""
    STLParameter(S, T, R)

Parameters for fitted STL decomposition.

# Fields  
- `S::Vector{Float64}`: Seasonal component for each observation
- `T::Vector{Float64}`: Trend component for each observation
- `R::Vector{Float64}`: Remainder component for each observation
"""
struct STLParameter <: AbstractModelParameters
    S::Vector{Float64}
    T::Vector{Float64}
    R::Vector{Float64}
    
    function STLParameter(S, T, R)
        length(S) == length(T) == length(R) || throw(ArgumentError("All components must have same length"))
        new(Float64.(S), Float64.(T), Float64.(R))
    end
end

"""
    STLEstimationSetting(;ni::Int = 2, no::Int = 5, ns::Int = 13,
                         nt::Int = -1, nl::Int = -1, s::Int = 1)

Settings for STL decomposition algorithm.

# Fields
- `ni::Int`: Number of inner iterations (default: 2)
- `no::Int`: Number of outer iterations for robustness (default: 5)  
- `ns::Int`: Smoothing parameter for seasonal extraction (default: 13)
- `nt::Int`: Smoothing parameter for trend estimation (auto if -1)
- `nl::Int`: Smoothing parameter for low-pass filter (auto if -1)
- `s::Int`: Seasonal period (used for automatic parameter calculation)

# Automatic Parameter Selection
When nt = -1: nt = max(5, ceil(1.5*s/(1-3/(2*ns)))) (odd)
When nl = -1: nl = s + iseven(s) (next odd number ≥ s)

# Parameter Interpretation
- **ni**: More inner iterations → better convergence (2-3 usually sufficient)
- **no**: More outer iterations → better outlier robustness (5-10 typical)
- **ns**: Larger ns → smoother seasonal component (7-15 typical range)
- **nt**: Larger nt → smoother trend component (auto-selection usually good)
- **nl**: Controls low-pass filtering in seasonal extraction
"""
struct STLEstimationSetting <: AbstractEstimationSetting
    ni::Int 
    no::Int
    ns::Int
    nt::Int
    nl::Int
    function STLEstimationSetting(;ni::Int = 2,
            no::Int = 5,
            ns::Int = 13,
            nt::Int = -1,
            nl::Int = -1,
            s::Int = 1)
        ni > 0 || throw(ArgumentError("Inner loops must be positive"))
        no > 0 || throw(ArgumentError("Outer loops must be positive"))
        ns > 0 || throw(ArgumentError("Seasonality smoothing coefficient must be positive"))
        
        if nl < 0
            nl = s + iseven(s)
        end
        if nt < 0
            nt = ceil(Int, 1.5 * s / (1 - 3/(2*ns)))
            nt = nt + iseven(nt)
            nt = max(nt, 5)
        end
        new(ni, no, ns, nt, nl)
    end
end

"""
    STLFitted

Container for fitted STL decomposition.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::STLModel`: Model specification
- `par::STLParameter`: Decomposed components (S, T, R)
- `estimation_setting::STLEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata
"""
struct STLFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::STLModel
    par::STLParameter
    estimation_setting::STLEstimationSetting
    temporal_info::TemporalInfo
    
    function STLFitted(x, model::STLModel, par::STLParameter,
                      estimation_setting::STLEstimationSetting,
                      temporal_info::TemporalInfo = TemporalInfo())
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        length(x) == length(par.S) || throw(ArgumentError("Data and parameters must have same length"))
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    loess_smooth(y::Vector{Float64}, q::Int, w::Vector{Float64}) -> Vector{Float64}

Apply LOESS smoothing with specified bandwidth and weights.

# Arguments
- `y::Vector{Float64}`: Values to smooth
- `q::Int`: Bandwidth (number of points in local regression)
- `w::Vector{Float64}`: Weights for robust fitting

# Returns  
- `Vector{Float64}`: Smoothed values (length = length(y) + 2 for boundary handling)

# Method
1. **Local Regression**: For each point, fit weighted linear regression to nearest q points
2. **Boundary Handling**: Use asymmetric neighbourhoods at series endpoints
3. **Robust Weights**: Incorporate weights for outlier resistance
4. **Buffer Points**: Add boundary points to handle edge effects

Used internally for seasonal and trend smoothing in STL algorithm.
"""
function loess_smooth(y::Vector{Float64}, q::Int, w::Vector{Float64})
    T = length(y)
    out = zeros(T + 2)
    m = div(q + 1, 2)

    # Start (asymmetric)
    for s = 0:m
        t_seq = 1:q
        yy = y[t_seq]
        λ = abs.(t_seq .- s)
        u = λ ./ maximum(λ)
        ww = (1 .- u.^3).^3
        ww = ww .* w[t_seq]
        ww = ww ./ sum(ww)

        t_bar = sum(ww .* t_seq)
        y_bar = sum(ww .* yy)
        b_hat = sum(ww .* (t_seq .- t_bar) .* (yy .- y_bar)) / sum(ww .* (t_seq .- t_bar).^2)
        a_hat = y_bar - b_hat * t_bar
        out[s + 1] = a_hat + b_hat * s
    end
    
    # Middle (symmetric)
    for s = (m + 1):(T - m)
        t_seq = (s - m):(s + m)
        yy = y[t_seq]
        λ = abs.(t_seq .- s)
        u = λ ./ maximum(λ)
        ww = (1 .- u.^3).^3
        ww = ww .* w[t_seq]
        ww = ww ./ sum(ww)
        out[s + 1] = sum(yy .* ww)
    end
    
    # End (asymmetric)
    for s = (T - m + 1):(T + 1)
        t_seq = (T - q + 1):T
        yy = y[t_seq]
        λ = abs.(t_seq .- s)
        u = λ ./ maximum(λ)
        ww = (1 .- u.^3).^3
        ww = ww .* w[t_seq]
        ww = ww ./ sum(ww)
        
        t_bar = sum(ww .* t_seq)
        y_bar = sum(ww .* yy)
        b_hat = sum(ww .* (t_seq .- t_bar) .* (yy .- y_bar)) / sum(ww .* (t_seq .- t_bar).^2)
        a_hat = y_bar - b_hat * t_bar
        out[s + 1] = a_hat + b_hat * s
    end
    
    return out
end

"""
    moving_average(y::Vector{Float64}, q::Int) -> Vector{Float64}

Compute simple moving average with specified window size.

# Arguments
- `y::Vector{Float64}`: Input series
- `q::Int`: Window size for averaging

# Returns
- `Vector{Float64}`: Moving averages (length = length(y) - q + 1)
"""
function moving_average(y::Vector{Float64}, q::Int)
    return [mean(y[(1:q) .+ i]) for i = 0:(length(y) - q)]
end

"""
    fit_baseline(x::Vector{T}, model::STLModel;
                setting::Union{STLEstimationSetting, Nothing} = nothing,
                temporal_info::TemporalInfo = TemporalInfo()) -> STLFitted

Fit STL decomposition to time series data.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::STLModel`: STL model specification
- `setting`: STL algorithm settings (optional, auto-generated if nothing)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `STLFitted`: Fitted decomposition with seasonal, trend, and remainder components

# Algorithm Steps
1. **Initialise**: Set up component vectors and weights
2. **Outer Loop**: Iterate for robustness (no iterations)
   - **Inner Loop**: Iterate for convergence (ni iterations)
     - **Seasonal**: Extract seasonal component using cycle-subseries smoothing
     - **Deseasonalise**: Remove seasonal component
     - **Trend**: Smooth deseasonalised series
     - **Remainder**: Compute residuals after trend removal
   - **Robustness**: Update weights based on remainder size
3. **Final Components**: Return S, T, R vectors

# Data Requirements
- Minimum length: At least 2 complete seasonal cycles (2s observations)
- Seasonal period: s must be reasonable relative to series length
"""
function fit_baseline(x::Vector{T}, model::STLModel;
                     setting::Union{STLEstimationSetting, Nothing} = nothing,
                     temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    
    if isnothing(setting)
        setting = STLEstimationSetting(s = model.s)
    end

    setting.ns < length(x)/model.s || throw(ArgumentError("Time series needs at least `ns` seasonal cycles"))
    
    nObs = length(x)
    model.s < nObs || throw(ArgumentError("Periodicity must be less than data length"))
    
    T_comp = zeros(nObs)
    S_comp = zeros(nObs)
    R_comp = zeros(nObs)
    L_comp = zeros(nObs)
    w = ones(nObs)
    
    xx = Float64.(x)
    
    ni = setting.ni
    no = setting.no
    nl = setting.nl
    nt = setting.nt
    ns = setting.ns

    s = model.s
    
    # Outer loop for robust fitting
    for i_outer = 1:no
        # Inner loop for convergence
        for i_inner = 1:ni
            # Step 1: De-trending
            dt = xx .- T_comp
            
            # Step 2: Cycle-subseries smoothing
            C = zeros(nObs + 2 * s)
            
            if s >= 2
                for ss = 1:s
                    yy = dt[ss:s:end]
                    if length(yy) > 1
                        ind = collect(ss:s:nObs) .+ s
                        smoothed = loess_smooth(yy, ns, w[ss:s:end])
                        C[ind] = smoothed[2:end-1]  # Remove buffer points
                    end
                end
                
                # Step 3: Low pass filter
                if length(C) >= s
                    L_filtered = moving_average(C, s)
                    if length(L_filtered) >= s
                        L_filtered = moving_average(L_filtered, s)
                    end
                    if length(L_filtered) >= 3
                        L_filtered = moving_average(L_filtered, 3)
                    end
                    if length(L_filtered) > 0
                        L_smooth = loess_smooth(L_filtered, nl, ones(length(L_filtered)))
                        L_comp = L_smooth[2:min(end-1, nObs+1)]
                        if length(L_comp) > nObs
                            L_comp = L_comp[1:nObs]
                        elseif length(L_comp) < nObs
                            L_comp = [L_comp; zeros(nObs - length(L_comp))]
                        end
                    end
                end
            end
            
            # Step 4: De-trending smoothed cycle-subseries
            S_comp = C[(s + 1):(s + nObs)] .- L_comp
            
            # Step 5: De-seasonalizing
            ds = xx .- S_comp
            
            # Step 6: Trend smoothing
            T_smooth = loess_smooth(ds, nt, w)
            T_comp = T_smooth[2:(end-1)]
        end
        
        # Update weights for next outer iteration
        R_comp = xx .- T_comp .- S_comp
        if no > 1 && i_outer < no
            h = 6 * median(abs.(R_comp))
            if h > 0
                u = abs.(R_comp) ./ h
                w = (1 .- u.^2).^2
                w[u .> 1] .= 0.0
                min_w = minimum(w[w .> 0])
                w[w .== 0] .= min_w > 0 ? min_w : 1e-6
            end
        end
    end
    
    par = STLParameter(S_comp, T_comp, R_comp)
    return STLFitted(x, model, par, setting, temporal_info)
end

"""
    point_forecast(fitted::STLFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts from STL decomposition.

# Arguments
- `fitted::STLFitted`: Fitted STL decomposition
- `horizon`: Forecast horizons

# Returns
- `Vector{Float64}`: Point forecasts combining trend and seasonal components

# Forecasting Method
1. **Trend Extrapolation**: Linear extrapolation of trend component
   - Estimate trend increment from recent trend values
   - Project trend linearly into future
2. **Seasonal Extension**: Cyclically repeat seasonal pattern
   - Extract average seasonal pattern across all cycles
   - Repeat pattern for forecast horizons
3. **Combine Components**: Add trend and seasonal forecasts
"""
function point_forecast(fitted::STLFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))
    
    nObs = length(fitted.x)
    h_max = maximum(horizon)
    
    # Forecast trend using average trend increment
    if length(fitted.par.T) > 1
        trend_increment = mean(diff(fitted.par.T))
        last_trend = fitted.par.T[end]
        trend_forecast = last_trend .+ trend_increment .* (1:h_max)
    else
        trend_forecast = fill(fitted.par.T[end], h_max)
    end
    
    # Forecast seasonality by cycling through seasonal pattern
    seasonal_pattern = zeros(fitted.model.s)
    for ss = 1:fitted.model.s
        seasonal_indices = ss:fitted.model.s:nObs
        if !isempty(seasonal_indices)
            seasonal_pattern[ss] = mean(fitted.par.S[seasonal_indices])
        end
    end
    
    # Extend seasonal pattern for forecast horizon
    seasonal_forecast = repeat(seasonal_pattern, ceil(Int, h_max / fitted.model.s))[1:h_max]
    
    # Combine trend and seasonal components
    point_forecasts = trend_forecast .+ seasonal_forecast
    
    return point_forecasts[horizon]
end

"""
    interval_forecast(fitted::STLFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate prediction intervals using STL remainder bootstrap.

# Arguments
- Standard interval forecast arguments with trajectory method

# Returns
- Standard interval forecast tuple with optional trajectories

# Simulation Method
1. **Deterministic Components**: Compute trend and seasonal forecasts as in point_forecast
2. **Remainder Sampling**: Bootstrap sample from historical remainder component
3. **Trajectory Generation**:
   - Start with deterministic forecast (trend + seasonal)
   - Add bootstrapped remainder innovations
   - Apply positivity corrections if specified
4. **Quantile Computation**: Empirical quantiles from trajectory ensemble

# Bootstrap Details
- **Remainder Pool**: Uses fitted remainder component R_t
- **Symmetry**: Includes both positive and negative remainder values
- **Independence**: Assumes remainder innovations are independent across time
- **Stationarity**: Assumes remainder distribution constant over time

# Positivity Correction
- `:truncate`: Remove negative remainder values from bootstrap pool
- `:zero_floor` / `:post_clip`: Set negative trajectory values to zero

# Note
STL only supports ModelTrajectoryInterval and EmpiricalInterval method.
No ParametricInterval available due to non-parametric nature of the decomposition.
"""
function interval_forecast(fitted::STLFitted,
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

    # Validate input
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    # Setting seed if specified
    Random.seed!(method.seed)

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    
    # Structure levels and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))
    
    remainder = fitted.par.R
    # Add symmetry for bootstrap
    remainder_pool = [remainder; -remainder]
    
    h_max = maximum(horizon)
    trajectories = zeros(method.n_trajectories, h_max)
    
    # Generate trajectories
    truncated = method.positivity_correction == :truncate
    clip = method.positivity_correction == :zero_floor

    trajectories[:, 1] = [make_step(remainder_pool, fc_point[1], truncated, clip) for _ in 1:method.n_trajectories]
    for hh = 2:h_max
        trajectories[:, hh] = [make_step(remainder_pool, from, truncated, clip) for from = trajectories[:, hh-1]]
    end

    if method.positivity_correction == :post_clip
        trajectories[trajectories .< 0] .= 0.0
    end

    trajectories = trajectories[:, horizon]
    
    # Remove seed
    Random.seed!(nothing)

    # Compute intervals from trajectories
    fc_intervals = Vector{ForecastInterval}(undef, length(horizon))
    fc_median = include_median ? zeros(length(horizon)) : nothing
    
    if include_median
        i_median = findfirst(alpha .== 0.5)
    end

    for i = 1:length(horizon)
        all_quantiles = [quantile(trajectories[:, i], q) for q in alpha]
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