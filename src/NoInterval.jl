"""
    interval_forecast(fitted::AbstractFittedModel, method::NoInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Nothing, Nothing)

Generate point forecasts without prediction intervals.

# Arguments
- `fitted::AbstractFittedModel`: Any fitted forecasting model
- `method::NoInterval`: Interval method specification (no parameters)
- `horizon::Union{Vector{Int}, Int, UnitRange{Int}}`: Forecast horizons (default: [1])
- `levels::Vector{Float64}`: Ignored (no intervals computed)
- `alpha_precision::Int`: Ignored (no quantiles computed)
- `include_median::Bool`: Whether to return median forecasts (default: true)

# Returns
- `Tuple` containing:
  - `Vector{Float64}`: Point forecasts from fitted model
  - `Union{Vector{Float64}, Nothing}`: Median forecasts (same as point forecasts) or nothing
  - `Nothing`: No prediction intervals
  - `Nothing`: No sample trajectories
"""
function interval_forecast(fitted::AbstractFittedModel,
    method::NoInterval,
    horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
    levels::Vector{Float64} = [0.95];
    alpha_precision::Int = 10,
    include_median::Bool = true)
    fc_point = point_forecast(fitted, horizon)
    if include_median
        fc_median = fc_point
    else
        fc_median = nothing
    end
    
    return fc_point, fc_median, nothing, nothing
end