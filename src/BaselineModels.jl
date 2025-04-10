module BaselineModels
using Distributions, LinearAlgebra, Optim, CountTimeSeries, Random, QuadGK, KernelDensity

abstract type Baseline end
include("forecast.jl")

import Distributions.fit

include("Constant.jl")
include("Marginal.jl")
include("KDE.jl")
include("LSD.jl")

include("OLS.jl")
include("IDS.jl")

include("ARMA.jl")
include("INARCH.jl")

include("STL.jl")
include("ETS.jl")

include("Seasonality.jl")
include("scoring.jl")
include("Calibration.jl")

export Baseline, forecastInterval, forecast, addTruth, addTrajectory, addMean, addMedian, getQuantiles, getQmat
export WIS, QRPS, fit, predict
export armaModel, armaParameter, armaFitted
export constantModel, constantParameter, constantFitted
export etsModel, etsParameter, etsFiltered, etsFilter, etsFitted
export idsModel, idsParameter, idsFitted
export inarchModel, inarchParameter, inarchFitted
export kdeModel, kdeParameter, kdeFitted
export lsdModel, lsdParameter, lsdFitted
export marginalModel, marginalParameter, marginalFitted
export olsModel, olsParameter, olsFitted
export stlModel, stlParameter, stlFitted
export seasonalityParameter, fitS, preFilter, postFilter
export makeStep, PITfun, PIThist, CvMdivergence

end

