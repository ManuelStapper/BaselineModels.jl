using Distributions, Plots, LinearAlgebra, Optim

abstract type Baseline end
include("forecast.jl")

import Distributions.fit

# Update code here after revising
include("Constant.jl")
include("Marginal.jl")
include("LSD.jl")

include("OLS.jl")
include("IDS.jl")

include("ARMA.jl")
include("INGARCH.jl")

include("STL.jl")
include("ETS.jl")

include("scoring.jl")



