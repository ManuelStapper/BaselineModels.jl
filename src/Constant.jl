#########################
### Constant Forecast ###
#########################

"""
    ConstantModel()

Constant forecasting model that predicts the last observed value.

The simplest forecasting method where all future values are predicted to equal
the final observation: X̂_{T+h} = X_T for all h > 0.

# Example
```julia
model = ConstantModel()
fitted = fit_baseline(data, model)
forecasts = point_forecast(fitted, 1:5)  # All equal to data[end]
```
"""
struct ConstantModel <: AbstractBaselineModel end

"""
    ConstantParameter(μ::Real)

Parameter for constant forecasting model.

# Fields
- `μ::Float64`: The constant value used for all forecasts (typically the last observation)
"""
struct ConstantParameter <: AbstractModelParameters
    μ::Float64
    function ConstantParameter(μ)
        μ isa Real || throw(ArgumentError("μ must be a real number"))
        new(μ)
    end
end


"""
    ConstantEstimationSetting()

Estimation settings for constant model.

Empty struct as no parameters need estimation beyond storing the final value.
Included for consistency with the general model interface.
"""
struct ConstantEstimationSetting <: AbstractEstimationSetting end

"""
    ConstantFitted

Container for fitted constant model.

# Fields
- `x::Vector{Float64}`: Original time series data
- `model::ConstantModel`: Model specification
- `par::ConstantParameter`: Estimated parameter (final observation)
- `estimation_setting::ConstantEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata
"""
struct ConstantFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::ConstantModel
    par::ConstantParameter
    estimation_setting::ConstantEstimationSetting
    temporal_info::TemporalInfo
    function ConstantFitted(x,
            model::ConstantModel,
            par::ConstantParameter,
            estimation_setting::ConstantEstimationSetting = ConstantEstimationSetting(),
            temporal_info::TemporalInfo = TemporalInfo())
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        new(x, model, par, estimation_setting, temporal_info)
    end
end

"""
    fit_baseline(x::Vector{T}, model::ConstantModel; 
                setting::Union{ConstantEstimationSetting, Nothing} = ConstantEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo()) -> ConstantFitted

Fit constant model by storing the last observed value.

# Arguments
- `x::Vector{T}`: Time series data where T <: Real
- `model::ConstantModel`: Model specification
- `setting`: Estimation settings (optional, no effect)
- `temporal_info`: Temporal metadata (optional)

# Returns
- `ConstantFitted`: Fitted model with μ = x[end]

# Method
Simply stores the final observation as the forecasting parameter.
No optimisation or statistical estimation required.

# Example
```julia
data = [1.0, 2.5, 3.1, 2.8, 3.5]
fitted = fit_baseline(data, ConstantModel())
fitted.par.μ  # Returns 3.5
```
"""
function fit_baseline(x::Vector{T},
             model::ConstantModel;
             setting::Union{ConstantEstimationSetting, Nothing} = ConstantEstimationSetting(),
             temporal_info::TemporalInfo = TemporalInfo()) where {T <: Real}
    if isnothing(setting)
        setting = ConstantEstimationSetting()
    end
    ConstantFitted(x, model, ConstantParameter(x[end]), setting, temporal_info)
end

function point_forecast(fitted::ConstantFitted,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    fill(fitted.par.μ, length(horizon))
end