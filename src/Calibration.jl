# Functions for calibration

mutable struct oneStepFunction
    l::Float64
    u::Float64
    function oneStepFunction(l::T1, u::T2) where {T1, T2 <: Real}
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

function evalStep(u::T, stepFun::oneStepFunction) where {T <: Real}
    if u <= stepFun.l return 0.0 end
    if u >= stepFun.u return 1.0 end
    return (u - stepFun.l)/(stepFun.u - stepFun.l)
end

function evalStep(u::T, stepFuns::Vector{oneStepFunction}) where {T <: Real}
    out = 0.0
    ns = length(stepFuns)
    for i = 1:ns
        out += evalStep(u, stepFuns[i])
    end
    out/ns
end

function makeStep(x::forecast, horizon::Int64 = 1)
    if length(x.truth) == 0
        error("Truth must be provided")
    end
    indH = x.horizon .== horizon
    if sum(indH) == 0
        error("Cannot find horizon")
    end
    ind = findfirst(indH)
    truth = x.truth[ind]
    α = x.interval[ind].α ./ 2
    hasMedian = length(x.median) > 0

    if hasMedian
        α = [α; 0.5; reverse(1 .- α)]
        q = [x.interval[ind].l; x.median[ind]; reverse(x.interval[ind].u)]
    else
        α = [α; reverse(1 .- α)]
        q = [x.interval[ind].l; reverse(x.interval[ind].u)]
    end
    if truth <= q[1]
        return oneStepFunction(0.0, α[1])
    end
    if truth > q[end]
        return oneStepFunction(α[end], 1.0)
    end

    iBin = sum(q .< truth)
    oneStepFunction(α[iBin], α[iBin + 1])
end


function PITfun(forecasts::Vector{forecast}, horizon::Int64 = 1)
    steps = makeStep.(forecasts, horizon)
    u -> evalStep(u, steps)
end

function PITfun(forecasts::Vector{forecast}, horizon::Vector{Int64} = [1])
    steps = oneStepFunction[]
    for i = 1:length(forecasts)
        for hh = horizon
            if hh in forecasts[i].horizon
                push!(steps, makeStep(forecasts[i], hh))
            end
        end
    end

    u -> evalStep(u, steps)
end

function PIThist(forecasts::Vector{forecast}, horizon = 1, nBins::Int64 = 10)
    F1 = PITfun(forecasts, horizon)
    uSeq = 0:(1/nBins):1
    Fu = F1.(uSeq)
    diff(Fu) .* nBins
end

function CvMdivergence(forecasts::Vector{forecast}, horizon = 1)
    F1 = PITfun(forecasts, horizon)
    N = 0
    for i = 1:length(forecasts)
        N += length(intersect(forecasts[i].horizon, horizon))
    end
    quadgk(u -> (F1(u) - u)^2, 0, 1)[1] * N
end

function CvMdivergence(steps::Vector{oneStepFunction})
    N = length(steps)
    F1 = u -> evalStep(u, steps)
    quadgk(u -> (F1(u) - u)^2, 0, 1)[1] * N
end

function GoF(forecasts::Vector{forecast}, horizon = 1, returnAll::Bool = false)
    α = forecasts[1].interval[1].α
    steps = oneStepFunction[]
    binWarning = false
    hasMedian = length(forecasts[1].median) > 0

    for i = 1:length(forecasts)
        for hh in horizon
            if hh in forecasts[i].horizon
                α2 = forecasts[i].interval[findfirst(forecasts[i].horizon .== hh)].α

                if (α2 == α) & (hasMedian == (length(forecasts[i].median) > 0))
                    push!(steps, makeStep(forecasts[i], hh))
                else
                    if !binWarning
                        println("Quantile probabilities do not match")
                        binWarning = true
                    end
                end
            end
        end
    end

    binLimits = [0.0; α ./ 2; if hasMedian 0.5 else Float64[] end; reverse(1 .- α ./ 2); 1.0]
    binLimits = round.(binLimits, digits = 3)
    N = length(steps)
    E = diff(binLimits) .* N

    ls = (z -> z.l).(steps)
    O = (l -> sum(ls .== l)).(binLimits[1:end-1])
    diffs = (O .- E) .^ 2 ./ E
    if returnAll
        return diffs
    end
    stat = sum(diffs)

    pVal = ccdf(Chisq(length(binLimits) - 2), stat)

    stat, pVal
end


####################################
######## Example code below ########
####################################

# Quick example:
# Compare an uninformed forecast with a forecast that includes more true assumptions

# N = 200
# μ = rand(Normal(0, sqrt(4)), N)
# x = rand.(Normal.(μ, sqrt(0.5)))

# Prediction 1: Without knowning μ
# Prediction 2: Knowing μ
# σ1 = sqrt(var(x))
# σ2 = sqrt(var(x .- μ))

# fc1 = forecast[]
# fc2 = forecast[]

# α = collect(0.01:0.01:0.49)

# for i = 1:N
#     l = quantile.(Normal(0.0, σ1), α)
#     u = reverse(quantile.(Normal(0.0, σ1), reverse(1 .- α)))
#     int = forecastInterval(α .* 2, l, u)
#     push!(fc1, forecast([1], median = [0.0], interval = int, truth = [x[i]]))

#     l = quantile.(Normal(μ[i], σ2), α)
#     u = reverse(quantile.(Normal(μ[i], σ2), reverse(1 .- α)))
#     int = forecastInterval(α .* 2, l, u)
#     push!(fc2, forecast([1], median = [μ[i]], interval = int, truth = [x[i]]))
# end

# F1 = PITfun(fc1)
# F2 = PITfun(fc2)

# plot(F1, xlims = (0, 1), lw = 2)
# plot!(F2, xlims = (0, 1), lw = 2)
# plot!(u -> u, color = "black", lw = 2, linestyle = :dash)

# GoF(fc1)
# GoF(fc2)

