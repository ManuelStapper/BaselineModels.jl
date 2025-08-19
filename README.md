# ForecastBaselines.jl

[![CI](https://github.com/ManuelStapper/ForecastBaselines.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ManuelStapper/ForecastBaselines.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/ManuelStapper/ForecastBaselines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ManuelStapper/ForecastBaselines.jl)

A comprehensive Julia package for baseline forecasting methods.

## Overview

ForecastBaselines.jl provides a framework for time series forecasting. The package implements a collection of baselines and established forecasting methods with a consistent interface across all models.

### Key Features

- **Unified Interface**: Consistent API across all forecasting methods, from simple baselines to sophisticated time series models
- **Uncertainty Quantification**: Multiple approaches for generating prediction intervals, including empirical, parametric, and simulation-based methods
- **Data Preprocessing**: Built-in support for transformations and seasonal-trend decomposition
- **Model Evaluation**: Scoring rules and calibration assessment tools

## Installation

```julia
using Pkg
Pkg.add("https://github.com/ManuelStapper/ForecastBaselines.jl")
```

## Basic Usage

```julia
using ForecastBaselines

# Fit an ARMA model to time series data
model = ARMAModel(p=2, q=1)
fitted = fit_baseline(data, model)

# Generate forecasts with prediction intervals
forecast_result = forecast(fitted,
    interval_method = EmpiricalInterval(),
    horizon = 1:12,
    levels = [0.8, 0.95]
)

# Evaluate forecast accuracy
mae_score = score(forecast_result, MAE())
```

## Implemented Methods

### Models

<details>
<summary><b>ConstantModel</b> - Naïve forecasts using the last observed value</summary><br>

><ins>Model Formulation</ins>
>
>The model simply predicts the last observed value into the future.
>
><ins>Architecture</ins>
> 
>No settingsrequired
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `μ` | Last observation |
>
><ins>Estimation settings</ins>
> 
>No settings required
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ❌ `ParametricInterval()`<br>
>  ❌ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>MarginalModel</b> - Forecasts based on empirical marginal distribution</summary><br>

><ins>Model Formulation</ins>
>
>The model takes the most recent p observations to estimate the marginal mean.
>
><ins>Architecture</ins>
> 
>- `p`: Number of observations
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `μ` | Marginal mean |
>
><ins>Estimation settings</ins>
> 
>- `estimation_function`: Function that estimates the marginal mean from data (default: mean)
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ✅ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>KDEModel</b> - Kernel density estimation for non-parametric forecasting</summary><br>

><ins>Model Formulation</ins>
>
>The density of the marginal distribution is estimated as
>
> $$\hat{f}(y) = \frac{1}{Th} \sum_{t = 1}^T K\left(\frac{y - y_t}{h}\right)$$
>
>where K is a kernel and h is the bandwidth.
>
><ins>Architecture</ins>
> 
>No settings required
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `x_seq` | sequence of nodes |
>| `density` | density at nodes |
>
><ins>Estimation settings</ins>
> 
>- `bandwidth_selection`: Bandwidth selection method (function) or fixed value
>- `kernel`: Kernel distribution type (default: Normal()) 
>- `npoints::Int`: Number of grid points for density evaluation (default: 2048)
>- `boundary`: Boundary handling method or fixed boundaries
>- `weights`: Observation weights method or fixed weights
>
>For details, see [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl)
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ✅ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>LSDModel</b> - Last Similar Dates method for seasonal patterns</summary><br>

><ins>Model Formulation</ins>
>
>The model estimates the means of periods for a seasonal time series. For estimation, the model takes not only observations of corresponding periods but also neighbouring observations ($$\pm w$$).
>
><ins>Architecture</ins>
> 
>- `s`: Periodicity
>- `w`: Window width
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `μ` | Vector of means |
>
><ins>Estimation settings</ins>
> 
>- `estimation_function`: Function that estimates the marginal mean from data (default: mean)
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ❌ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>OLSModel</b> - Ordinary least squares with polynomial trends</summary><br>

><ins>Model Formulation</ins>
>
>Linear regression model with polynomial time trend of order d fitted to p most recent obervations
>
> $$y_t = \beta_0 + \beta_1 t + ... + \beta_d t^d + \epsilon_t$$
>
><ins>Architecture</ins>
> 
>- `p`: Number of observations
>- `d`: Order of time trend polynomial
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `β` | Parameter vector |
>
><ins>Estimation settings</ins>
> 
>No settings required
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ✅ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>IDSModel</b> - Increase-Decrease-Stable trend detection</summary><br>

><ins>Model Formulation</ins>
>
>Fits an OLS model to the p most recent observations. If all observations go into the same direction (increase/decrease), it contains a linear trend and reduces to an intercept model otherwise (stable).
>
><ins>Architecture</ins>
> 
>- `p`: Number of observations
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `a` | Intercept |
>| `b` | Slope |
>
><ins>Estimation settings</ins>
> 
>No settings required
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ✅ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>ARMAModel</b>: Autoregressive Moving Average processes</summary><br>
  
><ins>Model Formulation</ins>
>
>The ARMA(p, q) model follows the specification:
>  
> $$X_t - \mu_t = \epsilon_t + \sum_{i=1}^p \alpha_i (X_{t-i} - \mu_{t-i}) + \sum_{i=1}^q \beta_i \epsilon_{t-i}$$
>
>where $\mu_t = \mu(\theta, t)$ is a deterministic trend/seasonal function.
>
><ins>Architecture</ins>
> 
>- **`p`, `q`**: Model orders (autoregressive and moving average)
>- **`μ`**: Mean function `μ(θ, t)` where `θ` are parameters and `t` is time
>- **`μDim`**: Number of parameters in the mean function
>
>Users can specify `s` (periodicity) and `trend` (Boolean) instead of custom mean functions for convenience.
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `α` | Autoregressive coefficients [α₁, α₂, ..., αₚ] |
>| `β` | Moving average coefficients [β₁, β₂, ..., βₑ] |
>| `μ` | Mean function parameters |
>| `σ²` | Innovation variance (≥ 0) |
>
><ins>Estimation settings</ins>
> 
>- `ensure_stability` - constrains for stationarity/invertibility
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ✅ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>INARCHModel</b> - Integer-valued ARCH for count time series</summary><br>

><ins>Model Formulation</ins>
>
>INARCH(p) model with optional seasonality, either Poisson or Negative Bionomial (conditional) distribution with mean
>
> $$\lambda_t = \mu_t(\beta_0 + \sum_{i = 1}^p (y_{t-i}/\mu_{t-i}))$$
>
>where $$\mu_t$$ is the seasonality component, $$\log(\mu_t)$$ is a harmonic wave of order k.
>
><ins>Architecture</ins>
>
>- `p`: Autoregressive order (default: 1)
>- `s`: Periodicity (default: 0, i.e. no seasonality)
>- `k`: Order of harmonic waves (default: 1)
>- `nb`: Negative Binomial distribution? (default: false)
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `β0` | Intercept |
>| `α` | Autoregressive parameters |
>| `ϕ` | Overdispersion parameter |
>| `γ` | Seasonality parameters |
>
><ins>Estimation settings</ins>
> 
>No settings required
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ❌ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>ETSModel</b> - Error-Trend-Season exponential smoothing (all 30 variants)</summary><br>

><ins>Model Formulation</ins>
>
>Exponential smoothing, decomposition into error, trend and seasonality. Errors can be additive or multiplicative. Seasonality can additionally be none. Trend component can be damped, if included in the model. All model variants can be summarised in state-space form
> 
> $$x_t = w(z_{t-1}) + r(z_{t-1})\epsilon_t$$
>
> $$z_t = f(z_{t-1}) + g(z_{t-1})\epsilon_t$$
>
>where $$z_t$$ is the state vector.
>
><ins>Architecture</ins>
>
>- `error`: Error type
>- `season`: Seasonality type (including periodicity s)
>- `trend`: Trend type
>
>For convenience, a constructor is implemented that takes
>
>- `error`: `"A"` or `"M"` for additive or multiplicative errors respectively
>- `season`: `"A"`, `"M"` or `"N"` (where `"N"` = no seasonality)
>- `s`: Periodicity
>- `trend`: `"A"`, `"Ad"`, `"M"`, `"Md"` or `"N"` (where `"Ad"` and `"Md"` are damped additive/multiplicative trends)
> 
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `θ` | Smoothing coefficients |
>| `z0` | Initial state |
>
><ins>Estimation settings</ins>
> 
>No settings required
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ❌ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

<details>
<summary><b>STLModel</b> - Seasonal-Trend decomposition using Loess</summary><br>

><ins>Model Formulation</ins>
>
>For details, see [Paper](https://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf)
>
><ins>Architecture</ins>
>
>- `s`: Periodicity
>
><ins>Parameters</ins>
>
>| Parameter | Description |
>|-----------|-------------|
>| `S` | Seasonality terms |
>| `T` | Trend terms |
>| `R` | Remainder terms |
>
><ins>Estimation settings</ins>
> 
>- `ni`: Number of inner loops
>- `no`: Number of outer loops
>- `ns`: Smoothing coefficient for seasonality term
>- `nt`: Smoothing coefficient for trend term
>- `nl`: Smoothing coefficient for Loess
>- `s`: Periodicity
>
><ins>Supported intervals</ins>
>  
>  ✅ `NoInterval()`<br>
>  ✅ `EmpiricalInterval()`<br>
>  ❌ `ParametricInterval()`<br>
>  ✅ `ModelTrajectoryInterval()`
</details>

## Uncertainty Quantification

ForecastBaselines.jl provides four approaches to prediction interval construction:

<details>
<summary><b>NoInterval</b>: Point forecasts only</summary><br>

>No intervals or trajectories are returned
>
>Does not require settings
</details>

<details>
<summary><b>Empirical Interval</b>: Bootstrap resampling from historical forecast errors</summary><br>

>Uses historic forecast errors to sample forecast trajectories.
>
><ins>Settings</ins>
>
>- `n_trajectories`: Number of trajectories to be sampled
>- `min_observation`: Minimum number of observations needed in each historic fit
>- `bootstrap_distribution`: Distribution to fit to historic forecast error (for example `Normal()`), or `nothing` if trajectories are sampled from forecast errors directly
>- `seed`: Random seed for reproducibility or `nothing` if no seed to be set
>- `positivity_correction`: Shall lower bounds be truncated at zero?
>    * `:none`: No correction
>    * `:post_clip`: Set negative values to zero after complete trajectory sampling
>    * `truncate`: Sample from a truncated distribution
>    * `:zero_clip`: Censor negative samples at zero during sampling
>- `symmetry_correction`: Use forecasts of both signs to ensure zero mean/median?
>- `stepwise`: If set to true, one-step-ahead forecast errors are used successively. Otherwise, corresponding h-step-ahead forecast errors are used
>- `return_trajectories`: If set to false (default), trajectories are not returned to save memory
</details>

<details>
<summary><b>ParametricInterval</b>: Model-specific analytical prediction intervals</summary><br>

>Computes intervals analytically, if formulae are available. Trajectories are never computed.
>
><ins>Settings</ins>
>
>- `positivity_correction`: Shall lower bounds be truncated at zero?
>    * `:none`: No correction
>    * `:post_clip`: Set negative values to zero after complete trajectory sampling
</details>

<details>
<summary><b>ModelTrajectory</b>: Simulation-based intervals</summary><br>
    
>Trajectories are computed by running fitted model into the future.
>
><ins>Settings</ins>
>
>- `n_trajectories`: Number of trajectories to be sampled
>- `seed`: Random seed for reproducibility or `nothing` if no seed to be set
>- `positivity_correction`: Shall lower bounds be truncated at zero?
>    * `:none`: No correction
>    * `:post_clip`: Set negative values to zero after complete trajectory sampling
>    * `truncate`: Sample from a truncated distribution
>    * `:zero_clip`: Censor negative samples at zero during sampling
>- `return_trajectories`: If set to false (default), trajectories are not returned to save memory
>Text
>
</details>

```julia
# Empirical intervals (model-agnostic)
fc1 = forecast(fitted, interval_method=EmpiricalInterval())

# Parametric intervals (model-specific)
fc2 = forecast(fitted, interval_method=ParametricInterval())

# Trajectory-based intervals
fc3 = forecast(fitted, interval_method=ModelTrajectoryInterval(n_trajectories=5000))
```

## Data Preprocessing

The package supports data preprocessing through transformations and seasonal-trend decomposition:

```julia
# Apply transformations for non-normal data
log_model = transform(ARMAModel(p=1, q=1), transformation=LogTransform())

# Seasonal-trend preprocessing
seasonal_model = transform(ConstantModel(),
                          season_trend=STTransform(s=12, trend=true))
```

Available transformations include logarithmic, power/Box-Cox, and square root transformations. Seasonal-trend decomposition uses harmonic functions to capture periodic patterns.

## Model Evaluation

ForecastBaselines.jl implements scoring rules for forecast evaluation and tools to assess calibration:

### Point Forecast Accuracy
- Mean Absolute Error (MAE), Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE), bias measures

### Probabilistic Forecast Quality
- Weighted Interval Score (WIS)
- Continuous Ranked Probability Score (CRPS)
- PIT function
- Cramér-von-Mises divergence of PIT function

(To be extended)

```julia
# Evaluate point forecasts
mae = score(forecasts, MAE())
rmse = score(forecasts, RMSE())

# Evaluate probabilistic forecasts
wis = score(forecasts, WIS())
crps = score(forecasts, CRPS())

# Assess calibration
calibration = CvM_divergence(forecasts)
```

## Examples

### Seasonal Data Analysis
```julia
# Fit ETS model to monthly data with seasonality
model = ETSModel(error="A", trend="A", season="A", s=12)
fitted = fit_baseline(monthly_data, model)

fc = forecast(fitted,
             interval_method=ModelTrajectoryInterval(n_trajectories=2000),
             horizon=1:24)
```

### Model Comparison
```julia
models = [ARMAModel(p=2, q=1), ETSModel(error="A", trend="A", season="N")]
forecasts = [forecast(fit_baseline(data, m), horizon=1:12) for m in models]
accuracies = [score(fc, MAE()) for fc in forecasts]
```

### Count Data Modelling
```julia
# INARCH model for integer-valued time series
model = INARCHModel(p=2, s=7, nb=true)  # Weekly seasonality with overdispersion
fitted = fit_baseline(count_data, model)
fc = forecast(fitted, horizon=1:14)
```

## Documentation

Documentation is available through Julia's help system:

```julia
?ARMAModel
?forecast
?score
```

(More details to be added)

## Contributing

Contributions are welcome through issues and pull requests.
To add models, see also ModelTemplate.jl

## Citation

```bibtex
@software{ForecastBaselines.jl,
  author = {Manuel Stapper},
  title = {ForecastBaselines.jl: Baseline Forecasting Methods in Julia},
  url = {https://github.com/ManuelStapper/ForecastBaselines.jl},
  version = {0.1.0},
  year = {2025}
}
```

## Licence

This package is distributed under the MIT Licence. See the LICENCE file for details.
