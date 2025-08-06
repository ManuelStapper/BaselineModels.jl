####################################
### Additional Model "Modelname" ###
####################################

# Struct for the new model
struct ModelnameModel <: AbstractBaselineModel
    # Architectual parameters as entries here
end

struct ModelnameParameter <: AbstractModelParameters
    # List all parameters of the model to be estimated here
end

struct ModelnameEstimationSetting <: AbstractEstimationSetting
    # Include estimation settings in needed.
    # This includes every argument beyond fit_baseline(x, model)
    # Can be empty if no further settings needed
end

struct ModelnameFitted <: AbstractFittedModel
    # Struct for a fitted model
    # Must always contain time series x, model, parameters and estimation settings
    x::Vector{Float64}
    model::ModelnameModel
    par::ModelnameParameter
    estimation_setting::ModelnameEstimationSetting
    temporal_info::TemporalInfo
end

function fit_baseline(x::Vector{T},
        model::ModelnameModel;
        setting::ModelnameEstimationSetting = ModelnameEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    # Function to fit model to time series
    # Ideally takes time series model and estimation setting as input
    # That way, the EmpiricalInterval method for prediction intervals can be applied
end

function point_forecast(fitted::ModelnameFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horisons must be provided."))
    
    # Function to compute point forecasts.
    # Horizon checks are left here for convenience
end


##################################################
### If forecast intervals shall be computed by ###
### a method other than EmpiricalInterval      ###
##################################################

# For ParametricInterval or ModelTrajectoryInterval

function interval_forecast(fitted::ModelnameFitted,
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
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horisons must be provided."))

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))

    # Positivity corrections can only be :none or :post_clip
    # For :post_clip make sure to clip all intervals at zero

    # Function must always return point forecast, median forecast (if specified)
    # intervals and trajectories are ommitted for this method

    return fc_point, fc_median, fc_intervals, nothing
end


function interval_forecast(fitted::ModelnameFitted,
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
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horisons must be provided."))

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)
    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l*0.5).(levels)...), digits = alpha_precision)))

    # Positivity corrections can be :none or :post_clip, :truncate and :zero_floor
    # For :post_clip make sure to clip all intervals at zero
    # Should be included for completeness, no correction as fallback

    # Function must always return point forecast, median forecast (if specified)
    # intervals and trajectories (if specified)

    return fc_point, fc_median, fc_intervals, trajectories
end

