"""
    NoInterval

Method for no computation of forecast intervals.
"""
struct NoInterval <: AbstractIntervalMethod end

"""
    EmpiricalInterval

Method to compute prediction intervals.
Historic prediction errors are used either directly or to fit error distribution.

# Fields
- `n_trajectories::Int`: Number of trajectory samples to generate
- `min_observation::Int`: Minimum number of observations needed in each historic fit
- `bootstrap_distribution::Union{ContinuousUnivariateDistribution, Symbol}`: Distribution to fit to historic prediction errors
        or `nothing` if histtoric forecast errors should be used directly. If a distribution is provided, it must have
        `fit(d, x)` implemented.
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
- `positivity_correction::Symbol`: Shall lower bounds be truncated at zero? (:none, :post_clip, :truncate or :zero_floor)
    * `:none`: No correction
    * `:post_clip`: Setting negative interval boundaries to zero
    * `:truncate`: Truncates the sampling distribution at zero during sampling of trajectories
    * `:zero_floor`: Sets negative samples to zero in trajectories
- `symmetry_correction::Bool`: Use forecast errors of both signs?
- `stepwise::Bool`: Use only one-step-ahead of corresponding h-step-ahead forecast errors?
- `return_trajectories::Bool`: Returns `nothing` as trajectory if set to `true`
"""
struct EmpiricalInterval <: AbstractIntervalMethod
    n_trajectories::Int
    min_observation::Int
    bootstrap_distribution::Union{ContinuousUnivariateDistribution, Nothing}
    seed::Union{Int, Nothing}
    positivity_correction::Symbol
    symmetry_correction::Bool
    stepwise::Bool
    return_trajectories::Bool
    function EmpiricalInterval(;n_trajectories = 1000, min_observation = 1, bootstrap_distribution = nothing, seed = nothing, positivity_correction = :none, symmetry_correction = false, stepwise = false, return_trajectories = false)
        n_trajectories > 0 || throw(ArgumentError("Number of trajectories must be positive"))
        new(n_trajectories, min_observation, bootstrap_distribution, seed, positivity_correction, symmetry_correction, stepwise, return_trajectories)
    end
end

# Below two methods are model specific

"""
    ParametricInterval

Method to compute prediction intervals.
The fitted model is used to compute prediction intervals directly.

# Fields
- `positivity_correction::Symbol`: Shall lower bounds be truncated at zero? (:none, :post_clip)
    * `:none`: No correction
    * `:post_clip`: Setting negative interval boundaries to zero
"""
struct ParametricInterval <: AbstractIntervalMethod
    positivity_correction::Symbol
    function ParametricInterval(;positivity_correction = :none)
        if !(positivity_correction in [:none, :post_clip])
            @warn "positivity correction invalid, set to `:none`"
            positivity_correction = :none
        end
        new(positivity_correction)
    end
end

"""
    ModelTrajectoryInterval

Method to compute prediction intervals.
The fitted model is used to generate trajectories.

# Fields
- `n_trajectories::Int`: Number of trajectory samples to generate
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
- `positivity_correction::Symbol`: Shall lower bounds be truncated at zero? (:none, :post_clip, :truncate or :zero_floor)
    * `:none`: No correction
    * `:post_clip`: Setting negative interval boundaries to zero
    * `:truncate`: Truncates the sampling distribution at zero during sampling of trajectories
    * `:zero_floor`: Sets negative samples to zero in trajectories
- `return_trajectories::Bool`: Returns `nothing` as trajectory if set to `false`
"""
struct ModelTrajectoryInterval <: AbstractIntervalMethod
    n_trajectories::Int
    seed::Union{Int, Nothing}
    positivity_correction::Symbol
    return_trajectories::Bool
    function ModelTrajectoryInterval(;n_trajectories = 1000, seed = nothing, positivity_correction = :none, return_trajectories = false)
        n_trajectories > 0 || throw(ArgumentError("Number of trajectories must be positive"))
        new(n_trajectories, seed, positivity_correction, return_trajectories)
    end
end
