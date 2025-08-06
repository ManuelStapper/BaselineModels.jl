# Functions for calibration
"""
    OneStepFunction(l::Float64, u::Float64)

Step function for probability integral transform (PIT) analysis.

Represents a step function that transitions from 0 to 1 between bounds `l` and `u`.
Used to construct empirical cumulative distribution functions from forecast intervals.

# Fields
- `l::Float64`: Lower bound (automatically rounded to 3 decimal places)
- `u::Float64`: Upper bound (automatically rounded to 3 decimal places)

# Constructor Behavior
- Automatically swaps bounds if `l > u`
- Validates that both bounds are in [0,1]
- Rounds to 3 decimal places for numerical stability
"""
struct OneStepFunction
    l::Float64
    u::Float64
    function OneStepFunction(l::T1, u::T2) where {T1, T2 <: Real}
        l = round(l, digits = 3)
        u = round(u, digits = 3)
        if !(0 <= l <= 1) || !(0 <= u <= 1)
            error("Invalid steps")
        end
        if l > u
            new(float(u), float(l))
        else
            new(float(l), float(u))
        end
    end
end

"""
    evaluate_step(u::Float64, step_function::OneStepFunction) -> Float64
    evaluate_step(u::Float64, step_functions::Vector{OneStepFunction}) -> Float64

Evaluate step function(s) at point `u`.

For single step function:
- Returns 0 if `u ≤ l`
- Returns 1 if `u ≥ u`  
- Returns linear interpolation `(u - l)/(u - l)` between bounds

For multiple step functions:
- Returns average evaluation across all functions
- Used to construct aggregate PIT functions from multiple forecasts

# Arguments
- `u::Float64`: Evaluation point in [0,1]
- `step_function(s)`: OneStepFunction or vector thereof
"""
function evaluate_step(u::T, step_function::OneStepFunction) where {T <: Real}
    if u <= step_function.l return 0.0 end
    if u >= step_function.u return 1.0 end
    return (u - step_function.l)/(step_function.u - step_function.l)
end

function evaluate_step(u::T, step_functions::Vector{OneStepFunction}) where {T <: Real}
    out = 0.0
    ns = length(step_functions)
    for i = 1:ns
        out += evaluate_step(u, step_functions[i])
    end
    out/ns
end

"""
    create_step(fc::Forecast; horizon::Union{Int, Vector{Int}, UnitRange{Int}, Nothing} = nothing) -> Union{OneStepFunction, Vector{OneStepFunction}, Missing}

Create step function from forecast interval for PIT analysis.

Converts prediction intervals into step functions representing the empirical CDF.
The step function bounds correspond to the probability levels where the observed 
value falls within the prediction intervals.

# Arguments
- `fc::Forecast`: Forecast object with intervals and truth values
- `horizon`: Specific horizon(s) to analyse (default: all horizons)

# Returns
- `OneStepFunction`: For single horizon
- `Vector{OneStepFunction}`: For multiple horizons  
- `Missing`: If truth or intervals unavailable for horizon

# Example
```julia
# Single horizon PIT step
step = create_step(forecast, horizon=1)

# Multiple horizons
steps = create_step(forecast, horizon=1:5)
```
"""
function create_step(fc::Forecast; horizon::Union{Int, Vector{Int}, UnitRange{Int}, Nothing} = nothing)
    has_truth(fc) ||  throw(ArgumentError("Truth must be provided"))
    if isnothing(horizon)
        horizon = fc.horizon
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end
    if horizon isa Vector
        return (hh -> create_step(fc, horizon = hh)).(horizon)
    end
    indH = fc.horizon .== horizon
    if sum(indH) == 0
        return missing
    end
    ind = findfirst(indH)
    truth = fc.truth[ind]
    α = reverse(round.((1 .- fc.intervals[ind].levels) ./ 2, digits = 10))

    if has_median(fc)
        α = [α; 0.5; reverse(1 .- α)]
        q = [fc.intervals[ind].lower; fc.median[ind]; reverse(fc.intervals[ind].upper)]
    else
        α = [α; reverse(1 .- α)]
        q = [fc.intervals[ind].lower; reverse(fc.intervals[ind].upper)]
    end
    if truth <= q[1]
        return OneStepFunction(0.0, α[1])
    end
    if truth > q[end]
        return OneStepFunction(α[end], 1.0)
    end

    iBin = sum(q .< truth)
    OneStepFunction(α[iBin], α[iBin + 1])
end

"""
    PIT_function(fc::Vector{Forecast}; horizon::Union{Int, Vector{Int}, UnitRange{Int}, Nothing} = nothing) -> Function
    PIT_function(steps::Vector{OneStepFunction}) -> Function

Construct probability integral transform (PIT) function from forecasts.

Creates empirical CDF by averaging step functions across multiple forecasts.
The resulting function maps [0,1] → [0,1] and should approximate the identity
function for well-calibrated forecasts.

# Arguments
- `fc::Vector{Forecast}`: Collection of forecasts with truth values
- `horizon`: Horizon(s) to analyse
- `steps::Vector{OneStepFunction}`: Pre-computed step functions

# Returns
- `Function`: PIT function `u -> F̂(u)` where F̂ is empirical CDF

# Example
```julia
# Create PIT function from forecasts
pit_func = PIT_function(forecasts, horizon=1)

# Evaluate at specific points
pit_values = pit_func.(0:0.1:1)
```
"""
function PIT_function(fc::Vector{Forecast}; horizon::Union{Int, Vector{Int}, UnitRange{Int}, Nothing} = nothing)
    steps = vcat((f -> create_step(f, horizon = horizon)).(fc)...)
    steps = Vector{OneStepFunction}(steps[.!ismissing.(steps)])
    u -> evaluate_step(u, steps)
end

function PIT_function(steps::Vector{OneStepFunction})
    u -> evaluate_step(u, steps)
end

"""
    CvM_divergence(fc::Vector{Forecast}; horizon::Union{Int, Vector{Int}, UnitRange{Int}, Nothing} = nothing) -> Float64
    CvM_divergence(steps::Vector{OneStepFunction}) -> Float64

Compute Cramér-von Mises divergence for forecast calibration assessment.

Measures the squared distance between the empirical PIT function and the 
uniform distribution (identity function). Provides a single statistic
summarising forecast calibration quality.

# Arguments
- `fc::Vector{Forecast}`: Collection of forecasts with truth values
- `horizon`: Horizon(s) to analyse
- `steps::Vector{OneStepFunction}`: Pre-computed step functions

# Returns
- `Float64`: CvM divergence statistic (≥ 0, closer to 0 indicates better calibration)

# Mathematical Details
Computes: n × ∫ [F̂(u) - u]² du

where:
- F̂(u) is the empirical PIT function
- n is the number of forecasts
- Integration performed analytically using piecewise linear approximation

# Interpretation
- CvM = 0: Perfect calibration
- Small CvM: Good calibration
- Large CvM: Poor calibration (over/under-confident or biased)
- Can be used to compare calibration across different forecasting methods

# Example
```julia
# Assess calibration quality
cvm_stat = CvM_divergence(forecasts, horizon=1)
println("CvM divergence: ", cvm_stat)

# Compare methods
cvm_method1 = CvM_divergence(forecasts1)
cvm_method2 = CvM_divergence(forecasts2)
```
"""
function CvM_divergence(fc::Vector{Forecast}; horizon::Union{Int, Vector{Int}, UnitRange{Int}, Nothing} = nothing)
    steps = vcat((f -> create_step(f, horizon = horizon)).(fc)...)
    steps = Vector{OneStepFunction}(steps[.!ismissing.(steps)])
    F = PIT_function(steps)
    αs = sort(unique([0.0; 1.0; unique((x -> x.l).(steps)); unique((x -> x.u).(steps))]))
    τs = F.(αs)
    out = 0.0
    for i in 1:length(αs)-1
        α₁, α₂ = αs[i], αs[i+1]
        τ₁, τ₂ = τs[i], τs[i+1]

        h = α₂ - α₁
        m = (τ₂ - τ₁) / h
        b = τ₁ - m * α₁
        a = m - 1
        I = (a^2 / 3) * (α₂^3 - α₁^3) + (a * b) * (α₂^2 - α₁^2) + (b^2) * (α₂ - α₁)

        out += I
    end

    length(steps)*out
end

function CvM_divergence(steps::Vector{OneStepFunction})
    F = PIT_function(steps)
    αs = sort(unique([0.0; 1.0; unique((x -> x.l).(steps)); unique((x -> x.u).(steps))]))
    τs = F.(αs)
    out = 0.0
    for i in 1:length(αs)-1
        α₁, α₂ = αs[i], αs[i+1]
        τ₁, τ₂ = τs[i], τs[i+1]

        h = α₂ - α₁
        m = (τ₂ - τ₁) / h
        b = τ₁ - m * α₁
        a = m - 1
        I = (a^2 / 3) * (α₂^3 - α₁^3) + (a * b) * (α₂^2 - α₁^2) + (b^2) * (α₂ - α₁)

        out += I
    end

    length(steps)*out
end