# Code file for forecast error types and computations
# Prep for a scoring rule wrapper function
# Split into point errors, interval errors and trajectory errors

abstract type AbstractError end

####################
### Point errors ###
####################

abstract type AbstractPointError <: AbstractError end

"""
    ForecastError(;relative::Bool = false)

Standard forecast error: target - truth.

# Fields
- `relative::Bool`: If true, compute relative error (target - truth)/truth

Used for bias measurement and signed error analysis.
"""
struct ForecastError <: AbstractPointError
    relative::Bool
    function ForecastError(;relative::Bool = false)
        new(relative)
    end
end

"""
    AbsoluteError(;relative::Bool = false, inner::Bool = true)

Absolute forecast error for accuracy measurement.

# Fields  
- `relative::Bool`: If true, compute relative version
- `inner::Bool`: For relative errors, whether to apply abs() inside (true) or outside (false) division

Computes |target - truth| or |target - truth|/truth depending on configuration.
"""
struct AbsoluteError <: AbstractPointError
    relative::Bool
    inner::Bool
    function AbsoluteError(;relative::Bool = false, inner::Bool = true)
        new(relative, inner)
    end
end

"""
    SquaredError(;relative::Bool = false, inner::Bool = true)

Squared forecast error for variance-based metrics.

# Fields
- `relative::Bool`: If true, compute relative version  
- `inner::Bool`: For relative errors, whether to square inside (true) or outside (false) division

Used in MSE, RMSE calculations and emphasizes larger errors.
"""
struct SquaredError <: AbstractPointError
    relative::Bool
    inner::Bool
    function SquaredError(;relative::Bool = false, inner::Bool = true)
        new(relative, inner)
    end
end

"""
    SignError()

Sign of forecast error for directional accuracy.

Returns sign(target - truth): +1, 0, or -1.
Used to assess whether forecasts correctly predict direction of change.
"""
struct SignError <: AbstractPointError end

"""
    PinballError(alpha = nothing)

Asymmetric loss function for quantile forecasting.

# Fields
- `alpha::Union{Float64, Nothing}`: Quantile level (0 < alpha < 1)

Computes: 2 × [α × (truth - target) if truth < target, else (1-α) × (target - truth)]

If `alpha = nothing`, automatically inferred from target quantile specification.
"""
struct PinballError <: AbstractPointError
    alpha::Union{Float64, Nothing}
    function PinballError(alpha = nothing)
        (isnothing(alpha) || (0 < alpha < 1)) || throw(ArgumentError("`alpha` must be between 0 and 1"))
        new(alpha)
    end
end

# Helper function that gets the target from a forecast object
# - fc::Forecast
# - target::Symbol One of :mean, :median, :qXXX where for example :q50 yields the median, :q25 the lower quartile etc.
"""
    get_target_from_forecast(fc::Forecast, target::Symbol) -> Vector{Float64}

Extract target values from forecast object.

# Arguments
- `fc::Forecast`: Forecast object
- `target::Symbol`: Target specification
  - `:mean`: Point forecasts
  - `:median`: Median forecasts  
  - `:qXX`: Quantile forecasts (e.g., `:q25`, `:q75`, `:q95`)

# Returns
- `Vector{Float64}`: Extracted target values for all horizons

# Quantile Specification
- `:q50` equivalent to `:median`
- `:q25` extracts 25th percentile from prediction intervals
- Automatically finds appropriate confidence level in intervals

Fallback to `:mean` if requested target unavailable.
"""
function get_target_from_forecast(fc::Forecast, target::Symbol)
    out = nothing
    if target == :median
        has_median(fc) || throw(ArgumentError("Median forecast must be provided"))
        out = fc.median
    elseif (string(target)[1] == 'q') & all((n -> n in string.(0:9)).(split(string(target), "")[2:end]))
        prob_string = string(target)[2:end]
        alpha = parse(Int, prob_string)/(10^length(prob_string))
        if alpha == 0.5
            return fc.median
        end
        has_intervals(fc) || throw(ArgumentError("Forecast object does not have quantile forecast"))

        level = abs(1 - 2*alpha)
        out = zeros(Union{Missing, Float64}, length(fc.horizon))

        for h = 1:length(fc.horizon)
            if any(fc.intervals[h].levels .≈ level)
                l_ind = findfirst(fc.intervals[h].levels .≈ level)
                if alpha > 0.5
                    out[h] = fc.intervals[h].upper[l_ind]
                else
                    out[h] = fc.intervals[h].lower[l_ind]
                end
            else
                out[h] = missing
            end
        end

        if !any(ismissing.(out)) 
            out = Vector{Float64}(out)
        end
    end

    # Fallback
    if isnothing(out)
        out = fc.mean
    end
    
    return out
end

# Given values for target and truth, compute errors
"""
    compute_error(target::Real, truth::Real, error::AbstractPointError) -> Float64

Compute single point forecast error.

# Arguments
- `target::Real`: Forecast value
- `truth::Real`: Observed value  
- `error::AbstractPointError`: Error type specification

# Returns
- `Float64`: Computed error value

Core error computation for all point error types with automatic dispatch.
"""
function compute_error(target::Real, truth::Real, error::ForecastError)
    if error.relative
        (truth != 0) || throw(ArgumentError("Truth cannot be zero for relative errors"))
        return (target - truth)/truth
    end
    target - truth
end

function compute_error(target::Real, truth::Real, error::AbsoluteError)
    if error.relative
        (truth != 0) || throw(ArgumentError("Truth cannot be zero for relative errors"))
        if error.inner
            return abs(target - truth)/truth
        else
            return abs((target - truth)/truth)
        end
    end
    abs(target - truth)
end

function compute_error(target::Real, truth::Real, error::SquaredError)
    if error.relative
        (truth != 0) || throw(ArgumentError("Truth cannot be zero for relative errors"))
        if error.inner
            return (target - truth)^2/truth
        else
            return ((target - truth)/truth)^2
        end
    end
    (target - truth)^2
end

function compute_error(target::Real, truth::Real, error::SignError)
    sign(target - truth)
end

function compute_error(target::Real, truth::Real, error::PinballError)
    2*ifelse(truth < target, (1 - error.alpha) * (target - truth), error.alpha*(truth - target))
end

function compute_error(target::Union{Missing, T1},
        truth::Union{Missing, T2},
        error::T3,
        missing_strategy::Symbol = :skip) where {T1 <: Real, T2 <: Real, T3 <: AbstractPointError}
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end
    if ismissing(target) || ismissing(truth)
        if missing_strategy == :error
            throw(ArgumentError("Missing value in truth or target"))
        end
        if missing_strategy == :skip
            return nothing
        end
        if missing_strategy == :propagate
            return missing
        end
    else
        return compute_error(target, truth, error)
    end
end

"""
    compute_error(target::Vector, truth::Vector, error::AbstractPointError, missing_strategy::Symbol = :skip) -> Vector{Float64}

Compute vector of point forecast errors with missing value handling.

# Arguments
- `target::Vector`: Forecast values (may contain missing)
- `truth::Vector`: Observed values (may contain missing)
- `error::AbstractPointError`: Error type
- `missing_strategy::Symbol`: How to handle missing values
  - `:skip`: Remove missing observations
  - `:error`: Throw error if missing found
  - `:propagate`: Keep missing in results

# Returns
- `Vector{Float64}`: Error values (length depends on missing strategy)

Vectorized error computation with flexible missing value handling.
"""
function compute_error(target::Union{Vector{Missing}, Vector{Union{Missing, T1}}, Vector{T1}},
        truth::Union{Vector{Missing}, Vector{Union{Missing, T2}}, Vector{T2}},
        error::T3,
        missing_strategy::Symbol = :skip) where {T1, T2 <: Real, T3 <: AbstractPointError}
    length(target) == length(truth) || throw(ArgumentError("Target and truth must be the same length"))
    errors = [compute_error(target[i], truth[i], error, :propagate) for i in 1:length(truth)]

    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    if missing_strategy == :error
        if any(ismissing.(errors))
            n_missing = sum(ismissing.(errors))
            missing_idx = findall(ismissing.(errors))
            throw(ArgumentError("Found $n_missing missing values at indices: $missing_idx"))
        end
        return errors
    elseif missing_strategy == :skip
        if any(.!ismissing.(errors))
            return collect(skipmissing(errors))
        else
            return Float64[]
        end
    elseif missing_strategy == :propagate
        return errors
    end
end

"""
    forecast_error(fc::Forecast, error::AbstractPointError;
                   horizon::Int = fc.horizon[1],
                   truth::Union{Vector, Real, Nothing} = fc.truth,  
                   target::Symbol = :mean,
                   missing_strategy::Symbol = :skip) -> Union{Float64, Nothing, Missing}

Compute forecast error for single horizon.

# Arguments
- `fc::Forecast`: Forecast object
- `error::AbstractPointError`: Error type specification
- `horizon::Int`: Specific horizon to evaluate
- `truth`: Truth values (defaults to fc.truth)
- `target::Symbol`: Which forecast component to use
- `missing_strategy::Symbol`: Missing value handling

# Returns
- Error value, `nothing` (if skipped), or `missing` (if propagated)

Evaluates forecast accuracy for a specific horizon and target combination.
"""
function forecast_error(fc::Forecast,
        error::T1;
        horizon::Int = fc.horizon[1],
        truth::Union{Vector{T2}, T2, Nothing} = fc.truth,
        target::Symbol = :mean,
        missing_strategy::Symbol = :skip) where {T1 <: AbstractPointError, T2 <: Real}
    # Check validity of horizons
    (horizon > 0) || throw(ArgumentError("Horizon must be positive"))
    (horizon in fc.horizon) || throw(ArgumentError("Selected horizon must be in forecast"))

    # Check missing strategy
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end
    
    # Check if truth is available
    if isnothing(truth)
        truth = fill(missing, length(fc.horizon))
    end

    # If truth provided as argument, check validity
    h_ind = findfirst(fc.horizon .== horizon)
    if truth isa Real
        truth_val = truth
    else
        (length(truth) == length(fc.horizon)) || throw(ArgumentError("Length of truth must match horizons"))
        truth_val = truth[h_ind]
    end

    # Get target from forecast object
    target_val = get_target_from_forecast(fc, target)

    # Check if alpha is given for pinball loss and replace if not
    if error isa PinballError
        if isnothing(error.alpha)
            if target in [:mean, :median]
                error = PinballError(0.5)
            else
                ((string(target)[1] == 'q') && all((n -> n in string.(0:9)).(split(string(target), "")[2:end]))) || throw(ArgumentError("Target is not a valid quantile"))
                prob_string = string(target)[2:end]
                alpha = parse(Int, prob_string)/(10^length(prob_string))
                error = PinballError(alpha)
            end
        end
    end

    compute_error(target_val[h_ind], truth_val, error, missing_strategy)
end

"""
    all_forecast_errors(fc::Forecast, error::AbstractPointError;
                        truth::Union{Vector, Real, Nothing} = fc.truth,
                        target::Symbol = :mean, 
                        missing_strategy::Symbol = :skip) -> Vector{Float64}

Compute forecast errors for all horizons.

# Arguments  
- `fc::Forecast`: Forecast object
- `error::AbstractPointError`: Error type specification
- `truth`: Truth values (defaults to fc.truth)
- `target::Symbol`: Which forecast component to use
- `missing_strategy::Symbol`: Missing value handling

# Returns
- `Vector{Float64}`: Errors for all horizons (length depends on missing strategy)

Batch computation of forecast errors across all available horizons.
"""
function all_forecast_errors(fc::Forecast,
        error::T1;
        truth::Union{Vector{T2}, T2, Nothing} = fc.truth,
        target::Symbol = :mean,
        missing_strategy::Symbol = :skip) where {T1 <: AbstractPointError, T2 <: Real}
    if isnothing(truth)
        truth = fill(missing, length(fc.horizon))
    end

    length(truth) == length(fc.horizon) || throw(ArgumentError("Truth must match length of horizon"))

    target_val = get_target_from_forecast(fc, target)
    
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    if error isa PinballError
        if isnothing(error.alpha)
            if target in [:mean, :median]
                error = PinballError(0.5)
            else
                ((string(target)[1] == 'q') && all((n -> n in string.(0:9)).(split(string(target), "")[2:end]))) || throw(ArgumentError("Target is not a valid quantile"))
                prob_string = string(target)[2:end]
                alpha = parse(Int, prob_string)/(10^length(prob_string))
                error = PinballError(alpha)
            end
        end
    end

    compute_error(target_val, truth, error, missing_strategy)
end

#######################
### Interval Errors ###
#######################

# Define error types for a target and truth value, where
# the target is a forecast interval
abstract type AbstractIntervalError <: AbstractError end

"""
    CoverageError(;correct_level::Bool = true)

Coverage indicator for prediction intervals.

# Fields
- `correct_level::Bool`: If true, subtract nominal coverage from empirical coverage

Returns whether observation falls within prediction interval(s).
Used for interval calibration assessment.
"""
struct CoverageError <:AbstractIntervalError
    correct_level::Bool
    function CoverageError(;correct_level::Bool = true)
        new(correct_level)
    end
end

"""
    IntervalWidth()

Width of prediction intervals.

Computes upper - lower bounds for each confidence level.
Used to assess forecast uncertainty and interval efficiency.
"""
struct IntervalWidth <: AbstractIntervalError end

"""
    IntervalScore()

Interval score combining coverage and width.

Computes: width + 2/α × undercoverage_penalty + 2/α × overcoverage_penalty
where α = 1 - confidence_level.

Proper scoring rule that rewards both coverage and sharpness.
"""
struct IntervalScore <: AbstractIntervalError end

"""
    CRPScore()

Continuous Ranked Probability Score for interval forecasts.

Approximates CRPS using prediction intervals and median forecast.
Integrates squared differences between forecast and empirical CDFs.

Proper scoring rule for probabilistic forecasts represented as quantiles.
"""
struct CRPScore <: AbstractIntervalError end

"""
    compute_error(target::ForecastInterval, truth::Real, error::AbstractIntervalError) -> Vector{Float64}

Compute interval forecast errors.

Returns vector of errors (one per confidence level) for interval-type errors,
or scalar for summary errors like CRPS.
"""
function compute_error(target::ForecastInterval, truth::Real, error::CoverageError)
    if error.correct_level
        return .!(target.lower .<= truth .<= target.upper) .- (1 .- target.levels)
    else
        return .!(target.lower .<= truth .<= target.upper)
    end
end

function compute_error(target::ForecastInterval, truth::Real, error::IntervalWidth)
    target.upper .- target.lower
end

function compute_error(target::ForecastInterval, truth::Real, error::IntervalScore)
    out = target.upper .- target.lower
    alpha = 1 .- target.levels
    out .+= 2 ./ alpha .* (target.lower .- truth) .* (truth .< target.lower)
    out .+= 2 ./ alpha .* (truth .- target.upper) .* (truth .> target.upper)
    out
end

function compute_error(target::ForecastInterval,
        target_median::Union{Real, Nothing},
        truth::Real,
        error::CRPScore)
    med = !isnothing(target_median)
    y_seq = [reverse(target.lower); ifelse(med, target_median, []); target.upper]
    alpha = (1 .- target.levels)./2
    alpha = [reverse(alpha); ifelse(med, 0.5, []); 1 .- alpha]

    out = 0.0
    for i = 1:length(y_seq) - 1
        if y_seq[i+1] < truth
            out += alpha[i]*(y_seq[i+1] - y_seq[i])
        elseif y_seq[i] > truth 
            out += (1 - alpha[i])*(y_seq[i+1] - y_seq[i])
        else
            out += alpha[i]*(truth - y_seq[i])
            out += (1 - alpha[i])*(y_seq[i+1] - truth)
        end
    end
    out
end


function compute_error(target::Union{Missing, ForecastInterval},
        truth::Union{Missing, T1},
        error::T2,
        missing_strategy::Symbol = :skip;
        target_median::Union{Nothing, T3} = nothing) where {T1 <: Real, T2 <: AbstractIntervalError, T3 <: Real}
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end
    if !ismissing(target)
        n_levels = length(target.levels)
    else
        n_levels = 1
    end

    if error isa CRPScore
        returnScalar = true
    else
        returnScalar = false
    end

    if ismissing(target) || ismissing(truth)
        if missing_strategy == :error
            throw(ArgumentError("Missing value in truth or target"))
        end
        if missing_strategy == :skip
            return ifelse(returnScalar, nothing, fill(nothing, n_levels))
        end
        if missing_strategy == :propagate
            return ifelse(returnScalar, missing, fill(missing, n_levels))
        end
    else
        # might not be needed, as there is a fallback?
        if error isa CRPScore
            return compute_error(target, target_median, truth, error)
        else
            return compute_error(target, truth, error)
        end
    end
end

function compute_error(target::Union{Vector{ForecastInterval}, Vector{Union{Missing, ForecastInterval}}, Vector{Missing}},
        truth::Union{Vector{Missing}, Vector{Union{Missing, T1}}, Vector{T1}},
        error::T2,
        missing_strategy::Symbol = :skip;
        target_median::Union{Vector{T3}, Vector{Nothing}, Vector{Union{T3, Nothing}}, Nothing} = nothing) where {T1 <: Real, T2 <: AbstractIntervalError, T3 <: Real}
    length(target) == length(truth) || throw(ArgumentError("Target and truth must be the same length"))
    if isnothing(target_median)
        target_median = fill(nothing, length(target))
    end
    if error isa CRPScore
        errors = [compute_error(target[i], truth[i], error, :propagate, target_median = target_median[i]) for i in 1:length(truth)]
    else
        errors = [compute_error(target[i], truth[i], error, :propagate) for i in 1:length(truth)]
    end

    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    if missing_strategy == :error
        if any(ismissing.(errors))
            n_missing = sum(ismissing.(errors))
            missing_idx = findall(ismissing.(errors))
            throw(ArgumentError("Found $n_missing missing values at indices: $missing_idx"))
        end
        return errors
    elseif missing_strategy == :skip
        if any(.!ismissing.(errors))
            return collect(skipmissing(errors))
        else
            if error isa CRPScore
                returnScalar = true
            else
                returnScalar = false
            end
            return ifelse(returnScalar, Float64[], fill(Float64[], length(target.levels)))
        end
    elseif missing_strategy == :propagate
        return errors
    end
end


function forecast_error(fc::Forecast,
        error::T1;
        horizon::Int = fc.horizon[1],
        truth::Union{Vector{T2}, T2, Nothing} = fc.truth,
        missing_strategy::Symbol = :skip) where {T1 <: AbstractIntervalError, T2 <: Real}
    # Check validity of horizons
    (horizon > 0) || throw(ArgumentError("Horizon must be positive"))
    (horizon in fc.horizon) || throw(ArgumentError("Selected horizon must be in forecast"))

    # Check missing strategy
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end
    
    # Check if truth is available
    if isnothing(truth)
        truth = fill(missing, length(truth.horizon))
    end

    # If truth provided as argument, check validity
    h_ind = findfirst(fc.horizon .== horizon)
    if truth isa Real
        truth_val = truth
    else
        (length(truth) == length(fc.horizon)) || throw(ArgumentError("Length of truth must match horizons"))
        truth_val = truth[h_ind]
    end

    if error isa CRPScore
        return compute_error(fc.intervals[h_ind], truth_val, error, missing_strategy, target_median = fc.median)
    else
        return compute_error(fc.intervals[h_ind], truth_val, error, missing_strategy)
    end
end

function all_forecast_errors(fc::Forecast,
        error::T1;
        truth::Union{Vector{T2}, T2, Nothing} = fc.truth,
        missing_strategy::Symbol = :skip) where {T1 <: AbstractIntervalError, T2 <: Real}
    if isnothing(truth)
        truth = fill(missing, length(fc.horizon))
    end
    length(truth) == length(fc.horizon) || throw(ArgumentError("Truth must match length of horizon"))
    
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    if error isa CRPScore
        return compute_error(fc.intervals, truth, error, missing_strategy, target_median = fc.median)
    else
        return compute_error(fc.intervals, truth, error, missing_strategy)
    end
end

#########################
### Trajectory Errors ###
#########################

# Define error types for a target and truth value, where
# the target is a forecast trajectory
abstract type AbstractTrajectoryError <: AbstractError end

"""
    LogLoss(;offset::Float64 = 0.01, score::Bool = false)

Logarithmic loss for trajectory forecasts.

# Fields
- `offset::Float64`: Small constant to prevent log(0)
- `score::Bool`: Currently unused

Computes -log(P(truth)) where P(truth) is empirical probability from trajectories.
Proper scoring rule for probabilistic forecasts.
"""
struct LogLoss <: AbstractTrajectoryError
    offset::Float64
    score::Bool
    # For continuous data might add kernel here?
    function LogLoss(;offset::T1 = 0.01, score::Bool = false) where T1 <: Real
        new(Float64(offset), score)
    end
end

"""
    CRPScore_trajectory(;n_points::Int = 0, alpha_seq = Float64[])

CRPS for trajectory-based forecasts.

# Fields
- `n_points::Int`: Number of evaluation points (if > 1)
- `alpha_seq::Vector{Float64}`: Specific quantile levels to use

Computes exact CRPS using full trajectory distribution.
More accurate than interval-based approximation when trajectories available.
"""
struct CRPScore_trajectory <: AbstractTrajectoryError
    n_points::Int
    alpha_seq::Vector{Float64}
    function CRPScore_trajectory(;n_points::Int = 0, alpha_seq = Float64[])
        if alpha_seq isa StepRangeLen
            alpha_seq = collect(alpha_seq)
        end
        if n_points <= 1
            n_points = 0
        end
        (alpha_seq isa Vector) || throw(ArgumentError("Sequence of probabilities `alpha_seq` must be a vector."))
        all(0 .< alpha_seq .< 1) || throw(ArgumentError("All probabilities in `alpha_seq` must be between 0 and 1"))
        ((n_points == 0) || (length(alpha_seq) == 0)) || throw(ArgumentError("Specify only one, the number of points `n_points` or the sequence of probabilities `alpha_seq`"))
        new(n_points, alpha_seq)
    end
end

"""
    compute_error(target::Vector{Float64}, truth::Real, error::AbstractTrajectoryError) -> Float64

Compute trajectory forecast errors.

# Arguments
- `target::Vector{Float64}`: Sample trajectories for single horizon
- `truth::Real`: Observed value
- `error::AbstractTrajectoryError`: Error type

Evaluates probabilistic forecast accuracy using full trajectory information.
"""
function compute_error(target::Vector{Float64}, truth::Real, error::LogLoss)
    n_trajectories = length(target)
    -log(mean(target .== truth) + error.offset)
end

function compute_error(target::Vector{Float64}, truth::Real, error::CRPScore_trajectory)
    n_trajectories = length(target)

    if error.n_points > 1
        y_min = minimum(target)
        y_max = maximum(target)
        stepsize = (y_max - y_min)/(error.n_points - 1)
        y_seq = y_min:stepsize:y_max
    elseif length(error.alpha_seq) > 0
        alpha = error.alpha_seq
        y_seq = quantile(target, alpha)
    else
        y_seq = sort(unique(target))
    end
    
    p = (yy -> mean(target .== yy)).(y_seq)
    alpha = cumsum(p)

    out = 0.0
    for i = 1:length(y_seq) - 1
        if y_seq[i+1] < truth
            out += alpha[i]*(y_seq[i+1] - y_seq[i])
        elseif y_seq[i] > truth 
            out += (1 - alpha[i])*(y_seq[i+1] - y_seq[i])
        else
            out += alpha[i]*(truth - y_seq[i])
            out += (1 - alpha[i])*(y_seq[i+1] - truth)
        end
    end
    out
end

function compute_error(target::Union{Missing, Vector{T1}},
        truth::Union{Missing, T2},
        error::T3,
        missing_strategy::Symbol = :skip) where {T1 <: Real, T2 <: Real, T3 <: AbstractTrajectoryError}
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    if ismissing(target) || ismissing(truth)
        if missing_strategy == :error
            throw(ArgumentError("Missing value in truth or target"))
        end
        if missing_strategy == :skip
            return nothing
        end
        if missing_strategy == :propagate
            return missing
        end
    else
        # might not be needed, as there is a fallback?
        return compute_error(target, truth, error)
    end
end

function compute_error(target::Union{Matrix{T1}, Matrix{Union{Missing, T1}}, Matrix{Missing}},
        truth::Union{Vector{Missing}, Vector{Union{Missing, T2}}, Vector{T2}},
        error::T3,
        missing_strategy::Symbol = :skip) where {T1 <: Real, T2 <: Real, T3 <: AbstractTrajectoryError}
    size(target)[2] == length(truth) || throw(ArgumentError("Number of columns in target must equal the length of truth"))
    errors = [compute_error(target[:, i], truth[i], error, :propagate) for i in 1:length(truth)]

    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    if missing_strategy == :error
        if any(ismissing.(errors))
            n_missing = sum(ismissing.(errors))
            missing_idx = findall(ismissing.(errors))
            throw(ArgumentError("Found $n_missing missing values at indices: $missing_idx"))
        end
        return errors
    elseif missing_strategy == :skip
        if any(.!ismissing.(errors))
            return collect(skipmissing(errors))
        else
            return Float64[]
        end
    elseif missing_strategy == :propagate
        return errors
    end
end

function forecast_error(fc::Forecast,
        error::T1;
        horizon::Int = fc.horizon[1],
        truth::Union{Vector{T2}, T2, Nothing} = fc.truth,
        missing_strategy::Symbol = :skip) where {T1 <: AbstractTrajectoryError, T2 <: Real}
    # Check validity of horizons
    (horizon > 0) || throw(ArgumentError("Horizon must be positive"))
    (horizon in fc.horizon) || throw(ArgumentError("Selected horizon must be in forecast"))

    # Check missing strategy
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end
    
    # Check if truth is available
    if isnothing(truth)
        truth = fill(missing, length(truth.horizon))
    end

    # If truth provided as argument, check validity
    h_ind = findfirst(fc.horizon .== horizon)
    if truth isa Real
        truth_val = truth
    else
        (length(truth) == length(fc.horizon)) || throw(ArgumentError("Length of truth must match horizons"))
        truth_val = truth[h_ind]
    end

    if has_trajectories(fc)
        return compute_error(fc.trajectories[:, h_ind], truth_val, error, missing_strategy)
    else
        return compute_error(missing, truth_val, error, missing_strategy)
    end
end

function all_forecast_errors(fc::Forecast,
        error::T1;
        truth::Union{Vector{T2}, T2, Nothing} = fc.truth,
        missing_strategy::Symbol = :skip) where {T1 <: AbstractTrajectoryError, T2 <: Real}
    if isnothing(truth)
        truth = fill(missing, length(fc.horizon))
    end
    length(truth) == length(fc.horizon) || throw(ArgumentError("Truth must match length of horizon"))
    
    if !(missing_strategy in [:error, :skip, :propagate])
        missing_strategy = :skip
    end

    compute_error(fc.trajectories, truth, error, missing_strategy)
end

# General vector evaluation of forecast_error
# Truth must be in the forecast object already here
"""
    forecast_error(fc::Vector{Forecast}, error::AbstractError; 
                   horizon::Int = fc[1].horizon[1],
                   missing_strategy::Symbol = :skip) -> Vector

Compute forecast errors across multiple forecast objects.

Batch processing for forecast evaluation across different models, time periods,
or forecast origins. Returns vector of error values matching input forecasts.
"""
function forecast_error(fc::Vector{Forecast}, error::T1;
    horizon::Int = fc[1].horizon[1],
    missing_strategy::Symbol = :skip) where {T1 <: AbstractError}
    [forecast_error(fc[i], error, horizon = horizon, missing_strategy = missing_strategy) for i = 1:length(fc)]
end

"""
    all_forecast_errors(fc::Vector{Forecast}, error::AbstractError;
                        missing_strategy::Symbol = :skip) -> Vector{Vector}

Compute all forecast errors across multiple forecast objects.

Returns nested vector structure: outer vector for forecasts, 
inner vectors for horizons within each forecast.
"""
function all_forecast_errors(fc::Vector{Forecast}, error::T1;
    missing_strategy::Symbol = :skip) where {T1 <: AbstractError}
    [all_forecast_errors(fc[i], error, missing_strategy = missing_strategy) for i = 1:length(fc)]
end
