# Data type to store forecasts and observed values

# Qmat:     Matrix of forecasts, rows = quantiles, cols = horizon
# quantile: Vector of quantile probabilities
# horizon:  Vector of horizons
# truth:    Vector of observations

mutable struct forecast
    Qmat::Matrix{T} where {T <: Real}
    quantile::Vector{Float64}
    horizon::Vector{Int64}
    truth::Vector{Int64}

    function forecast(Qmat::Matrix{T1},
                      quantile::Vector{Float64},
                      horizon::Vector{Int64},
                      truth::Vector{Int64} = Int64[]) where {T1 <: Real}
        #
        if length(truth) == 0
            truth = fill(-1, length(horizon))
        end
        if size(Qmat)[1] != length(quantile)
            error("Incompatible dimensions")
        end
        if size(Qmat)[2] != length(horizon)
            error("Incompatible dimensions")
        end
        if length(truth) != length(horizon)
            error("Incompatible dimensions")
        end

        new(Qmat, quantile, horizon, truth)
    end
end

function addTruth(x::forecast, truth::Vector{Int64})
    forecast(x.Qmat, x.quantile, x.horizon, truth)
end
