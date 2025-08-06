"""
    historical_forecast_errors(fitted::AbstractFittedModel,
                               horizon::Union{Vector{Int}, Int, UnitRange{Int}},
                               min_observation::Int = 1) -> Vector{Vector{Float64}}

Compute historical out-of-sample forecast errors for empirical interval construction.

Performs pseudo-real-time evaluation by refitting the model at each time point
and computing forecast errors. Used primarily by `EmpiricalInterval` method
to construct prediction intervals from historical forecast performance.

# Arguments
- `fitted::AbstractFittedModel`: Fitted model (used for model specification and data)
- `horizon::Union{Vector{Int}, Int, UnitRange{Int}}`: Forecast horizons to evaluate
- `min_observation::Int`: Minimum observations required for model fitting (default: 1)

# Returns
- `Vector{Vector{Float64}}`: Nested vector structure
  - Outer vector: One element per horizon (length = max(horizon))
  - Inner vectors: Historical errors for that horizon across all time points

# Algorithm
1. **Time Loop**: For each possible forecast origin t from T-1 down to min_observation
2. **Data Split**: Use observations 1:t for fitting, t+1:t+h for evaluation  
3. **Refit Model**: Fit model to truncated data using same specification
4. **Generate Forecasts**: Compute point forecasts for required horizons
5. **Compute Errors**: Calculate forecast errors (observed - predicted)
6. **Store Results**: Accumulate errors by horizon in result vectors

The function returns {e_{t,h} : t = min_observation, ..., T-h} for each h.

# Error Handling
- **Fitting failures**: Skipped silently (model may be unstable with limited data)
- **Forecasting failures**: Skipped silently (numerical issues in prediction)
- **Missing data**: NaN values filtered from results
- **Insufficient data**: Returns empty vectors for affected horizons

Primarily used internally by `EmpiricalInterval` method:
```julia
# EmpiricalInterval calls this automatically
method = EmpiricalInterval(min_observation=10)
intervals = interval_forecast(fitted, method, 1:12)
```
"""
function historical_forecast_errors(fitted::AbstractFittedModel,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}},
        min_observation::Int = 1)
    if horizon isa Int
        horizon > 0 || throw(ArgumentError("Horizon must be positive"))
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    all(horizon .>= 0) || throw(ArgumentError("All horizons must be positive"))
    min_observation > 0 || throw(ArgumentError("min_observation must be positive"))
    min_observation < length(fitted.x) || throw(ArgumentError("min_observation must be less than data length"))

    hMax = maximum(horizon)
    forecast_errors = [Float64[] for _ in 1:hMax]

    for tEnd = length(fitted.x) - 1:-1:min_observation
        xx = fitted.x[1:tEnd]
        if tEnd + hMax >= length(fitted.x)
            xxNew = fitted.x[tEnd+1:end]
            hKeep = tEnd .+ (1:hMax) .<= length(fitted.x)
        else
            xxNew = fitted.x[tEnd+1:tEnd+hMax]
            hKeep = fill(true, hMax)
        end
        
        newFit = nothing
        try
            newFit = fit_baseline(xx, fitted.model, setting = fitted.estimation_setting)
        catch e
            # @warn "Estimation failed at tEnd = $tEnd: $e"
            continue
        end

        pred = nothing
        try
            pred = point_forecast(newFit, (1:hMax)[hKeep])
        catch e
            # @warn "Forecast generation failed at tEnd=$tEnd: $e"
            continue  # Skip this time point
        end
        
        obs = fitted.x[tEnd .+ (1:hMax)[hKeep]]
        errors = obs - pred
        for i in 1:sum(hKeep)
            push!(forecast_errors[i], errors[i])
        end
    end
    forecast_errors = (e -> e[.!isnan.(e)]).(forecast_errors)

    return forecast_errors
end