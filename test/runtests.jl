using BaselineModels
using Test

@testset "BaselineModels.jl" begin
    using Random, Distributions

    # Add some tests, will be refined later
    
    Random.seed!(1)
    x = rand(Poisson(10), 1000)
    
    # ARMA
    model = armaModel(1, 1)
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])
    
    # Constant
    model = constantModel()
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # ETS
    model = etsModel(error = "A", trend = "N", season = "N")
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # IDS
    model = idsModel()
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # INARCH
    model = inarchModel(1)
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # LSD
    model = lsdModel(10, 2)
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # Marginal
    model = marginalModel(10)
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # OLS
    model = olsModel()
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    # STL
    model = stlModel()
    res = fit(x, model)
    pred = predict(res, 10, [0.1, 0.9])

    pred.truth = rand(Poisson(10), 10)

    WIS(pred)
    QRPS(pred)
end