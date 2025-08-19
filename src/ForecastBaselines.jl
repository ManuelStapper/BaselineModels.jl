module ForecastBaselines

using LinearAlgebra, Random, Dates, Distributions, Printf
using Polynomials, Optim, Interpolations, KernelDensity
using StatsBase: sample, wsample

include("Types.jl")
include("ForecastUtils.jl")
include("IntervalMethods.jl")
include("HistoricalForecastErrors.jl")

include("EmpiricalInterval.jl")
include("NoInterval.jl")

include("forecast.jl")

include("Seasonality.jl")
include("Transformations.jl")

include("ForecastErrors.jl")
include("Scoring.jl")
include("Calibration.jl")

# All models:
include("Marginal.jl")
include("Constant.jl")
include("KDE.jl")
include("LSD.jl")
include("OLS.jl")
include("IDS.jl")
include("ARMA.jl")
include("INARCH.jl")
include("STL.jl")
include("ETS.jl")

# Exports

export AbstractBaselineModel, AbstractFittedModel, AbstractModelParameters, 
       AbstractEstimationSetting, AbstractIntervalMethod, AbstractForecast,
       ForecastInterval, Forecast, TemporalInfo
export fit_baseline, forecast, point_forecast, interval_forecast
export has_horizon, has_mean, has_median, has_intervals, has_truth, has_trajectories,
       has_reference_date, has_target_date, has_temporal_info,
       add_truth, add_median, add_intervals, add_trajectories, add_temporal_info,
       remove_trajectories, update_model_name, truncate_horizon, filter_horizons,
       filter_levels, extend_forecast, forecast_length, max_horizon, min_horizon,
       get_horizon_range, num_trajectories, get_temporal_span
export NoInterval, EmpiricalInterval, ParametricInterval, ModelTrajectoryInterval
export NoTransform, LogTransform, LogPlusOneTransform, SquareRootTransform,
       PowerTransform, PowerPlusOneTransform, TransformedModel, TransformedFitted,
       transform, inverse_transform
export STTransform, STParameter, preFilter, postFilter, getST, fitST
export ConstantModel, ConstantParameter, ConstantEstimationSetting, ConstantFitted
export MarginalModel, MarginalParameter, MarginalEstimationSetting, MarginalFitted
export KDEModel, KDEParameter, KDEEstimationSetting, KDEFitted
export LSDModel, LSDParameter, LSDEstimationSetting, LSDFitted
export OLSModel, OLSParameter, OLSEstimationSetting, OLSFitted
export IDSModel, IDSParameter, IDSEstimationSetting, IDSFitted
export ARMAModel, ARMAParameter, ARMAEstimationSetting, ARMAFitted,
       ARMAtoMA, is_stationary, is_invertible, check_stability
export INARCHModel, INARCHParameter, INARCHEstimationSetting, INARCHFitted
export ETSModel, ETSParameter, ETSTuning, ETSSpace, ETSEstimationSetting, ETSFitted,
       ETSFiltered, ETSFilter, getInitial,
       AError, MError, NTrend, ATrend, AdTrend, MTrend, MdTrend,
       NSeason, ASeason, Mseason
export STLModel, STLParameter, STLEstimationSetting, STLFitted
export AbstractError, AbstractPointError, AbstractIntervalError, AbstractTrajectoryError
export ForecastError, AbsoluteError, SquaredError, SignError, PinballError
export CoverageError, IntervalWidth, IntervalScore, CRPScore
export LogLoss, CRPScore_trajectory
export compute_error, forecast_error, all_forecast_errors, get_target_from_forecast
export AbstractScoringRule, AbstractPointScoringRule, AbstractIntervalScoringRule,
       AbstractTrajectoryScoringRule
export PointScoringRule, WIS, CRPS, CRPS_trajectory, score
export MAE, MdAE, MAPE, MSE, MSPE, RMSE, Bias, RelativeBias
export OneStepFunction, PIT_function, CvM_divergence, create_step, evaluate_step
export historical_forecast_errors

end

