## File summary:
- ImportBaselines.jl: Loads packages, defines Baseline type and reads in models/functions. (Will be replaced by module definition later)
- forecast.jl: Object type to store forecasts. Contains:
  - `horizon`: Vector of forecast horizon(s)
  - `mean`: Vector of point forecasts
  - `median`: Vector of median forecasts
  - `interval`: Vector of interval forecasts (one element per horizon). Each of type `forecastInterval`
  - `truth`: Vector of observed data matching horizon(s)
  - `trajectory`: If sample-based forecast, matrix of trajectories

- scoring.jl Functions to evaluate forecasts (of type forecast) Currently: WIS & quantile-based CRPS
- Seasonality.jl: Functions to remove seasonality from time series and add back to forecast after fitting filtered time series


## Models

For each of the models below, the methods are implemented through the following types and methods.

Types: 
- `modelnameModel`: Type to define the model settings
- `modelnameParameter`: Type to store estimated parameters
- `modelnameFitted`: Type to store fitted models. Contains the time series, the model and estimated parameters

Methods:
- `fit`: Takes a time series `x` and a model `modelnameModel` and returns a `modelnameFitted` object
- `predict`: Takes a fitted model, maximum horizon `h` and quantiles `quant` and returns a forecasst object. Currently truth data is not stored, but can be added by `addTruth()`.

### Constant - Last available observation

For a time series, $x_1, ..., x_T$, the constant models predicts $\hat{x}_{T+h|T} = x_T$. Forecsat intervals are computed by drawing from past $h$-step ahead forecast errors. If the model setting `isPos` is set to `true`, forecast intervals are truncated at zero.

### Marginal: Forecasting from marginal distribution

The marginal model has two argumets to be set: `isPos` is set to `true` to truncate forecast intervals at zero, `p` is a smoothness or memory coefficient. It gives the number of most recent observations used to estimate the marginal mean, which is then used as forecast:
$$
    \hat{x}_{T+h|T} = \frac{1}{p}\sum_{i = 1}^{p} x_{T - i + 1}
$$
Forecast intervals are computed by sampling from past `h`-step ahead forecast errors.

### LSD - Last similar date

The last similar date model is a generalisation of the marginal model that allows for seasinality. Apart from `isPos`, the model is defined through `S` and `w`, where `S` gives the integer-values periodicity and `w` is the window size. If `w = 0`, the model forecasts the future as average of all observations with the same intra-season time (week of the year / day of the week / ...). For large periodicity and short time series, `w` can be set to include similar dates in the estimation as well. Forecast intervals are computed by sampling from past $h$-step ahead forecast errors. 

### OLS - Polynomial regression

The OLS model is specified by `isPos`, `p` and `d` where `d` gives the polynomial order of a regression model that is fit on the past `p` observations. Point forecasts are then the computed through the fitted regression equation. Forecast intervals are derived through past forecast errors, not assuming gaussian errors.

### IDS - Increase-Decrease-Stable

The IDS model is similar to the OLS model, it is defined through `p` and `isPos`. The model is an OLS model, that is fit on the past `p` observations. If those observations are all monotonically decreasing or increasing, the OLS model includes a trend and nbo trend otherwise. Forecast intervals are computed through past forecast errors.

### ARMA(p, q) with flexible seasonality

The model is defined by the AMRA orders `p` and `q`, `isPos` and the seasonality structure. The seasonality is specified by a function `µ` that has the time and a parameter vector as input. In addition `µDim` gives the dimension of the parameter vector. The model is fit by Maximum Likelihood. The seasonality is subtracted from the time series and an INARMA($p$, $q$) with mean zero is fit. The default seasonality is a constant intercept.
To simplify model definition, users may use the constructor `armaModel(p, q; m, trend)`, which defines a model with simple sine-cosine seasonality of periodicity `m` and may further include a trend. 

### INGARCH - Integer-Values GARCH

The integer-values counterpart of a GARCH process is defined by `p`, `m` and `nb`. `p` gives the "AR-order", the "MA-order" is always zero. `m` gives the periodicity, which is one by default, i.e. no seasonality. The conditional distribution is seleted by `nb`, such that a Poisson distribution is assumed if set to `false` and a Negative Binomial if it is `true`. The model is only applicable if the time series is integer-values and non-negative. 

### ETS - Exponential Smoothing Error-Trend-Seasonality

An ETS model can be defined by setting `error`, `trend`, `season` and `m`. Is is an exponential smoothing decopmposition of the time series into error trend and season. The `error` type can be either additive (`error = "A"`) or multiplicative (`error = "M"`). The `trend`can be additive (`"A"`), Multiplicative (`"M"`), or damped versions thereof (`"Ad"` and `"Md"`), or none (`"N"`). Similarly, the `season` can be additive (`"A"`), multiplicative (`"M"`) or none (`"N"`). If seasonality is selected, the periodicity can be specified by `m`. 
For details, see the R-package `forecast`.

### STL: Seasonality-Trend decomposition with LOESS

Similar to the ETS model, the STL decomposes the time series into seasonality, trend and remainder. Important to select is the periodicity `p`. The remaining model coefficients `i`, `o`, `l`, `t` and `s` are smoothing coefficients and settings for the algorithm.
Disclaimer: Will be changed in the future to have reasonable default values and removed from the settings.  