# Also include INARCH model with potential seasonality
using CountTimeSeries

mutable struct inarchModel <: Baseline
    p::Int64
    m::Int64
    nb::Bool
    function inarchModel(p, m = 0, nb = true)
        new(p, m, nb)
    end
end


mutable struct inarchFitted
    x::Vector{T} where {T <: Real}
    model::inarchModel
    par::INGARCHresults
end

function fit(x::Vector{T1}, model::inarchModel) where {T1 <: Real}
    # Create regressor matrix X
    T = length(x)
    if model.m > 0
        X = zeros(2, T)
        X[1, :] = (t -> sin(t/model.m*2*π)).(1:T)
        X[2, :] = (t -> cos(t/model.m*2*π)).(1:T)
        CTSmodel = Model(pastObs = model.p,
                            distr = ifelse(model.nb, "NegativeBinomial", "Poisson"),
                            link = "Log",
                            X = X,
                            external = [true, true])
        #
        return inarchFitted(x, model, CountTimeSeries.fit(x, CTSmodel, printResults = false))
    else
        CTSmodel = Model(pastObs = model.p,
                            distr = ifelse(model.nb, "NegativeBinomial", "Poisson"))
        #
        return inarchFitted(x, model, CountTimeSeries.fit(x, CTSmodel, printResults = false))
    end
end


# Include forecasting function here
# Again copying from CountTimeSeries and adapting!

function predict(fit::inarchFitted,
                 h::Int64,
                 quant::Vector{Float64},
                 nChain::Int64 = 10000)
    results = fit.par
    r = length(results.model.external)
    quant = sort(unique(round.([quant; 1 .- quant; 0.5], digits = 4)))

    y = results.y
    T = length(y)

    if fit.model.m > 0
        tSeq = T+1:T+h
        Xnew = zeros(2, h)
        Xnew[1, :] = (t -> sin(t/fit.model.m*2*π)).(tSeq)
        Xnew[2, :] = (t -> cos(t/fit.model.m*2*π)).(tSeq)
    end

    zi = results.model.zi
    nb = results.model.distr
    logl = results.model.link == "Log"
    lin = !logl
    pars = results.pars

    if typeof(results.model) != IIDModel
        p = length(results.model.pastObs)
    else
        p = 0
    end

    q = 0
    Q = 0

    if p == 0
        P = 0
    else
        P = maximum(results.model.pastObs)
    end

    M = P

    λ = zeros(nChain, M + h)
    ν = zeros(nChain, M + h)

    rI = 0
    rE = 0
    iI = []
    iE = []

    if r > 0
        iE = findall(results.model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    β0 = Float64(pars.β0)
    α = pars.α
    β = pars.β
    if fit.model.nb
        ϕ = pars.ϕ[1]
    else
        ϕ = 0.0
    end

    η = pars.η
    ω = pars.ω

    Y = zeros(Int64, nChain, M + h)
    λ = zeros(nChain, M + h)
    ν = zeros(nChain, M + h)

    λOld = results.λ

    if logl
        νOld = log.(λOld)
    else
        νOld = λOld
    end

    X = results.model.X
    if rE > 0
        for i = iE
            νOld = νOld .- η[i].*X[i, :]
        end
    end

    @inbounds for i = 1:nChain
        λ[i, 1:M] = λOld[(end - M + 1):end]
        ν[i, 1:M] = νOld[(end - M + 1):end]
        Y[i, 1:M] = y[(end - M + 1):end]
    end

    @simd for i = 1:nChain
        @inbounds for t = (M+1):(M+h)
            ν[i, t] = β0
            if p > 0
                if logl
                    ν[i, t] += sum(α.*log.(Y[i, t .- results.model.pastObs] .+ 1))
                else
                    ν[i, t] += sum(α.*Y[i, t .- results.model.pastObs])
                end
            end

            λ[i, t] = ν[i, t]

            if rE > 0
                λ[i, t] += sum(η[iE].*Xnew[iE, t - M])
            end

            if logl
                λ[i, t] = exp(λ[i, t])
            end

            if fit.model.nb
                p = ϕ/(ϕ + λ[i, t])
                Y[i, t] = rand(NegativeBinomial(ϕ, p))*(rand() > ω)
            else
                Y[i, t] = rand(Poisson(λ[i, t]))*(rand() > ω)
            end
        end
    end
    Yout = Y[:, M+1:end]

    meanY = mean(Yout, dims = 1)[1, :]
    Q = zeros(length(quant), h)

    for i = 1:h
        Q[:, i] = quantile(Y[:, i+1], quant)
    end

    interval = forecastInterval[]
    for hh = 1:h
        ls = Q[1:Int64((size(Q)[1] - 1)/2), hh]
        us = reverse(Q[Int64((size(Q)[1] - 1)/2) + 2:end, hh])
        αs = quant[1:Int64((size(Q)[1] - 1)/2)]*2
        push!(interval, forecastInterval(αs, ls, us))
    end

    medY = Q[Int64((size(Q)[1] - 1)/2) + 1, :]


    return forecast(1:h, mean = meanY, median = medY, interval = interval)
end