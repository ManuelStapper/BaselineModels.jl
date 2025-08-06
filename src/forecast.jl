"""
    forecast(fitted::AbstractFittedModel;
             interval_method::AbstractIntervalMethod = NoInterval(),
             horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
             levels::Vector{Float64} = [0.95],
             alpha_precision::Int = 10,
             include_median::Bool = true,
             truth::Union{Vector{Float64}, Nothing} = nothing,
             model_name::String = "") -> Forecast

Generate comprehensive forecasts from any fitted model.

Main interface for creating complete `Forecast` objects with point predictions,
prediction intervals, and optional sample trajectories from any fitted model
in the forecasting framework.

# Arguments
- `fitted::AbstractFittedModel`: Any fitted forecasting model
- `interval_method::AbstractIntervalMethod`: Method for computing prediction intervals
  - `NoInterval()`: Point forecasts only
  - `EmpiricalInterval()`: Bootstrap from historical errors
  - `ParametricInterval()`: Model-based analytical intervals
  - `ModelTrajectoryInterval()`: Simulation-based intervals
- `horizon::Union{Vector{Int}, Int, UnitRange{Int}}`: Forecast horizons (default: [1])
- `levels::Vector{Float64}`: Confidence levels for intervals (default: [0.95])
- `alpha_precision::Int`: Decimal precision for quantile computation (default: 10)
- `include_median::Bool`: Whether to compute median forecasts (default: true)
- `truth::Union{Vector{Float64}, Nothing}`: Observed values for evaluation (optional)
- `model_name::String`: Descriptive name for the forecast (default: "")

# Returns
- `Forecast`: Complete forecast object containing:
  - Point forecasts from the fitted model
  - Median forecasts (if requested)
  - Prediction intervals at specified levels
  - Truth values (if provided)
  - Sample trajectories (if interval method generates them)
  - Temporal metadata (reference date, target dates, resolution)

# Workflow
1. **Validate inputs**: Check horizon specification and confidence levels
2. **Generate intervals**: Call appropriate `interval_forecast` method
3. **Extract components**: Separate point forecasts, intervals, trajectories
4. **Create metadata**: Generate temporal information from fitted model
5. **Construct object**: Build complete `Forecast` with all components

# Example Usage
```julia
# Basic point forecasts
fc = forecast(fitted_model, horizon=1:12)

# With empirical prediction intervals
fc = forecast(fitted_model, 
              interval_method=EmpiricalInterval(n_trajectories=2000),
              horizon=1:24,
              levels=[0.8, 0.95],
              model_name="ARMA(2,1)")

# With parametric intervals and truth values
fc = forecast(fitted_model,
              interval_method=ParametricInterval(),
              horizon=1:6,
              truth=observed_values,
              model_name="Exponential Smoothing")

# Full trajectory-based analysis
fc = forecast(fitted_model,
              interval_method=ModelTrajectoryInterval(
                  n_trajectories=5000,
                  return_trajectories=true,
                  positivity_correction=:truncate
              ),
              horizon=1:12,
              levels=[0.5, 0.8, 0.95])
```

# Temporal Information
Automatically extracts temporal metadata from the fitted model:
- `reference_date`: Date of last observation
- `target_date`: Dates being forecast
- `resolution`: Time step between observations
"""
function forecast(fitted::AbstractFittedModel;
                  interval_method::AbstractIntervalMethod = NoInterval(),
                  horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                  levels::Vector{Float64} = [0.95],
                  alpha_precision::Int = 10,
                  include_median::Bool = true,
                  truth::Union{Vector{Float64}, Nothing} = nothing,
                  model_name::String = "")
    #
    # Validate input
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(0.0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    alpha_precision > 0 || throw(ArgumentError("Alpha precision must be positive"))

    fc_point, fc_median, fc_interval, fc_trajectory = interval_forecast(fitted, interval_method, horizon, levels, alpha_precision = alpha_precision, include_median = include_median)

    ti = fitted.temporal_info
    rd = ti.start + ti.resolution * (length(fitted.x) - 1)

    Forecast(horizon = horizon,
        mean = fc_point,
        median = fc_median,
        intervals = fc_interval,
        truth = truth,
        trajectories = fc_trajectory,
        reference_date = rd,
        target_date = rd .+ horizon .* ti.resolution,
        resolution = ti.resolution,
        model_name = model_name)
end