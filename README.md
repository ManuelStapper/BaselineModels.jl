## File summary:
- ImportBaselines.jl: Loads packages, defines Baseline type and reads in models/functions
- forecast.jl: Object type to store forecasts
- scoring.jl Functions to evaluate forecasts (of type forecast) Currently: WIS & quantile-based CRPS
- Seasonality.jl: Functions to remove seasonality from time series and add back to forecast after fitting filtered time series


## Models:
- Constant.jl: Forecast is last available observation
- Marginal.jl: Forecast is the average of most recent observations (or all available)
- LSD.jl: "Last-Similar-Date"
- OLS: Polynomial OLS regression based on most recent observations
- IDS: "Increase/Decrease/Stable". If last few observations grow/fall monotonically, OLS regression with trend, if not, without trend
- ARMA: ARMA(p, q) model, where seasonality if given as a function of time
- INGARCH: INGARCH(p, 0) model (i.e. INARCH(p)), sine-cosine seasonality optional
- ETS: Exponential Smoothing (Error-Trend-Seasonality)
- STL: Seasonality-Trend decomposition with LOESS
