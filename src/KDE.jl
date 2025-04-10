######################
### Kernel Density ###
######################

struct kdeModel <: Baseline
    isPos::Bool
    function kdeModel(isPos::Bool = true)
        new(isPos)
    end
end

mutable struct kdeParameter
    x::StepRangeLen
    density::Vector{T} where {T <: Real}
end

mutable struct kdeFitted
    x::Vector{T} where {T <: Real}
    model::kdeModel
    par::kdeParameter
end

function fit(x::Vector{T}, model::kdeModel) where {T <: Real}
    dens = kde(x)
    par = kdeParameter(dens.x, dens.density)
    kdeFitted(x, model, par)
end

function predict(fit::kdeFitted,
                 h::Int64,
                 quant::Vector{Float64})
    #
    quant = sort(unique(round.([quant; 1 .- quant; 0.5], digits = 4)))

    # Approximate quantiles from KDE
    xx = cumsum(fit.par.density) ./ sum(fit.par.density)
    yy = collect(fit.par.x)

    # Extending to be sure
    xx= [0.0; xx; 1.0]
    Δy = yy[2] - yy[1]
    yy = [yy[1] - Δy; yy; yy[end] + Δy]

    Q = zeros(Float64, length(quant))

    for i = 1:length(quant)
        ind = sum(xx .<= quant[i])
        Q[i] = yy[ind] + (yy[ind + 1] - yy[ind])*(xx[ind + 1] - quant[i])/(xx[ind+1] - xx[ind])
    end

    if isPos
        Q[Q .< 0] .= 0.0
    end

    ls = Q[1:Int64((length(Q) - 1)/2)]
    us = reverse(Q[Int64((length(Q) - 1)/2) + 2:end])
    αs = quant[1:Int64((length(Q) - 1)/2)]*2

    out = fill(Q[Int64((length(Q) + 1)/2)], h)
    interval = fill(forecastInterval(αs, ls, us), h)

    return forecast(1:h, mean = out, median = out, interval = interval)
end