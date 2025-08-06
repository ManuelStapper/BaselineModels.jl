using Test, BaselineModels, Random, Distributions
using Dates

# Set seed for reproducible tests
Random.seed!(123)

# Generate test data
function generate_test_data()
    n = 250
    trend_data = collect(1:n) + 0.1 * randn(n)
    seasonal_data = 10 .+ 2 * sin.(2π * (1:n) / 12) + 0.5 * randn(n)
    stationary_data = 5 .+ 0.3 * randn(n)
    count_data = rand(Poisson(5), n)
    return trend_data, seasonal_data, stationary_data, count_data
end

trend_data, seasonal_data, stationary_data, count_data = generate_test_data()

# Test horizons
test_horizons = [1, 3, [1, 2, 5], 1:6]

@testset "Comprehensive Model Tests" begin
    @testset "Constant Model Tests" begin
        model = ConstantModel()
        fitted = fit_baseline(stationary_data, model)
        
        @test fitted isa ConstantFitted
        @test fitted.par.μ ≈ stationary_data[end]
        
        # Test point forecasts
        for horizon in test_horizons
            fc = point_forecast(fitted, horizon)
            expected_length = horizon isa Int ? horizon : length(horizon)
            @test length(fc) == expected_length
            @test all(fc .≈ stationary_data[end])
        end
        
        # Test with forecast wrapper
        forecast_result = forecast(fitted, model_name="Constant Test")
        @test forecast_result.model_name == "Constant Test"
        @test has_mean(forecast_result)
    end
    
    @testset "Marginal Model Tests" begin
        model = MarginalModel(p = 10)
        fitted = fit_baseline(stationary_data, model)
        
        @test fitted isa MarginalFitted
        @test fitted.par.μ ≈ mean(stationary_data[end-9:end])
        
        # Test point forecasts
        fc = point_forecast(fitted, 5)
        @test length(fc) == 5
        @test all(fc .≈ fitted.par.μ)
        
        # Test parametric intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(), 1:3, [0.8, 0.9])
        @test length(fc_intervals) == 3
        @test fc_traj === nothing
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=100, return_trajectories = true), 1:3, [0.8])
        @test size(fc_traj) == (100, 3)
    end
    
    @testset "LSD Model Tests" begin
        model = LSDModel(s = 12, w = 2)  # 12 period seasonality, window ±2
        fitted = fit_baseline(seasonal_data, model)
        
        @test fitted isa LSDFitted
        @test length(fitted.par.μ) == 12
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:12)
        @test length(fc) == 12
        
        # Test parametric intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(), 1:3, [0.9])
        @test length(fc_intervals) == 3
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
    end
    
    @testset "OLS Model Tests" begin
        # Test linear trend
        model = OLSModel(p = 5, d = 1)
        fitted = fit_baseline(trend_data, model)
        
        @test fitted isa OLSFitted
        @test length(fitted.par.β) == 2  # intercept + linear
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        @test fc[2] > fc[1]  # Should show trend
        
        # Test parametric intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(), 1:2, [0.9])
        @test length(fc_intervals) == 2
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
        
        # Test quadratic trend
        model_quad = OLSModel(p = 10, d = 2)
        fitted_quad = fit_baseline(trend_data, model_quad)
        @test length(fitted_quad.par.β) == 3  # intercept + linear + quadratic
    end
    
    @testset "IDS Model Tests" begin
        model = IDSModel(p = 3)
        fitted = fit_baseline(trend_data, model)
        
        @test fitted isa IDSFitted
        @test fitted.par.a isa Float64
        @test fitted.par.b isa Float64
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        
        # Test parametric intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(), 1:2, [0.9])
        @test length(fc_intervals) == 2
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
    end
    
    @testset "ARMA Model Tests" begin
        # Test ARMA(1,1) with constant mean
        model = ARMAModel(p = 1, q = 1)
        fitted = fit_baseline(stationary_data, model)
        
        @test fitted isa ARMAFitted
        @test length(fitted.par.α) == 1
        @test length(fitted.par.β) == 1
        @test fitted.par.σ² > 0
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        
        # Test parametric intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(), 1:2, [0.9])
        @test length(fc_intervals) == 2
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
        
        # Test ARMA with seasonality
        model_seasonal = ARMAModel(p = 1, q = 1, s=12, trend=false)
        fitted_seasonal = fit_baseline(seasonal_data, model_seasonal)
        @test fitted_seasonal isa ARMAFitted
        @test length(fitted_seasonal.par.μ) == 3  # β₀ + sin + cos
    end
    
    @testset "ETS Model Tests" begin
        # Test simple ETS(A,N,N)
        model = ETSModel(error="A", trend="N", season="N")
        fitted = fit_baseline(stationary_data, model)
        
        @test fitted isa ETSFitted
        @test fitted.model.error isa AError
        @test fitted.model.trend isa NTrend
        @test fitted.model.season isa NSeason
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
        
        # Test ETS(A,A,N) with trend
        model_trend = ETSModel(error="A", trend="A", season="N")
        fitted_trend = fit_baseline(trend_data, model_trend)
        @test fitted_trend.model.trend isa ATrend
        
        # Test ETS(A,N,A) with seasonality
        model_seasonal = ETSModel(error="A", trend="N", season="A", s=12)
        fitted_seasonal = fit_baseline(seasonal_data, model_seasonal)
        @test fitted_seasonal.model.season isa ASeason
        @test fitted_seasonal.model.season.s == 12
    end
    
    @testset "STL Model Tests" begin
        model = STLModel(s = 12)  # 12-period seasonality
        fitted = fit_baseline(seasonal_data, model)
        
        @test fitted isa STLFitted
        @test length(fitted.par.S) == length(seasonal_data)
        @test length(fitted.par.T) == length(seasonal_data)
        @test length(fitted.par.R) == length(seasonal_data)
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
    end
    
    @testset "INARCH Model Tests" begin
        # Test INARCH(1) without seasonality
        model = INARCHModel(p=1, s=0, nb=false)
        fitted = fit_baseline(count_data, model)
        
        @test fitted isa INARCHFitted
        @test length(fitted.par.α) == 1
        @test fitted.par.β0 > 0
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        @test all(fc .>= 0)  # Count data should be non-negative
        
        # Test trajectory intervals (only method available for INARCH)
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
        @test all(fc_traj .>= 0)  # All trajectories should be non-negative
        
        # Test INARCH with negative binomial
        model_nb = INARCHModel(p=1, s=0, nb=true)
        fitted_nb = fit_baseline(count_data, model_nb)
        @test fitted_nb.par.ϕ > 0
        
        # Test INARCH with seasonality
        model_seasonal = INARCHModel(p=1, s=12, k=1, nb=false)
        fitted_seasonal = fit_baseline(count_data, model_seasonal)
        @test length(fitted_seasonal.par.γ) == 2  # sin + cos coefficients
    end
    
    @testset "KDE Model Tests" begin
        model = KDEModel()
        fitted = fit_baseline(stationary_data, model)
        
        @test fitted isa KDEFitted
        @test length(fitted.par.x_seq) == length(fitted.par.density)
        
        # Test point forecasts
        fc = point_forecast(fitted, 1:3)
        @test length(fc) == 3
        @test all(fc .≈ fc[1])  # KDE gives constant forecast (mean)
        
        # Test parametric intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(), 1:2, [0.9])
        @test length(fc_intervals) == 2
        
        # Test trajectory intervals
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories=50, return_trajectories = true), 1:2, [0.8])
        @test size(fc_traj) == (50, 2)
    end
    
    @testset "Interval Method Tests" begin
        # Test different interval methods with Marginal model
        model = MarginalModel(p = 10)
        fitted = fit_baseline(stationary_data, model)
        
        # Test NoInterval
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, NoInterval(), 1:3)
        @test fc_intervals === nothing
        @test fc_traj === nothing
        @test fc_median == fc_point
        
        # Test EmpiricalInterval
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, EmpiricalInterval(n_trajectories=50, min_observation=5, return_trajectories = true), 1:2, [0.8])
        @test length(fc_intervals) == 2
        @test size(fc_traj) == (50, 2)
        
        # Test ParametricInterval with positivity correction
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ParametricInterval(positivity_correction=:post_clip), 1:2, [0.8])
        @test length(fc_intervals) == 2
        
        # Test ModelTrajectoryInterval with different positivity corrections
        for correction in [:none, :post_clip, :zero_floor, :truncate]
            fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
                fitted, ModelTrajectoryInterval(n_trajectories=20, positivity_correction=correction, return_trajectories = true), 1:2, [0.8])
            @test size(fc_traj) == (20, 2)
        end
    end

    @testset "More comprehensive interval method test" begin
        all_models = [MarginalModel(p = 10),
              ConstantModel(),
              KDEModel(),
              LSDModel(s = 10, w = 1),
              OLSModel(p = 10, d = 1),
              IDSModel(p = 3),
              ARMAModel(p = 1, q = 1),
              INARCHModel(p = 1),
              STLModel(),
              ETSModel(trend = "A")]
#
        i_methods = [NoInterval(),
             EmpiricalInterval(n_trajectories = 50, positivity_correction = :none),
             EmpiricalInterval(n_trajectories = 50, positivity_correction = :zero_floor),
             EmpiricalInterval(n_trajectories = 50, positivity_correction = :truncate),
             EmpiricalInterval(n_trajectories = 50, positivity_correction = :post_clip),
             ParametricInterval(positivity_correction = :none),
             ParametricInterval(positivity_correction = :post_clip),
             ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :none),
             ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :zero_floor),
             ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :truncate),
             ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :post_clip)]
#
        hasPI = [true; false; fill(true, 5); fill(false, 3)]
        hasMTI = [true; false; fill(true, 8)]

        for i = 1:10
            if i == 8
                fitted = fit_baseline(count_data, all_models[i])
            else
                fitted = fit_baseline(stationary_data, all_models[i])
            end
            fc = forecast(fitted, interval_method = i_methods[1], horizon = 3)
            @test length(fc.mean) == 3

            fc = forecast(fitted, interval_method = i_methods[2], horizon = 3)
            @test length(fc.mean) == 3

            fc = forecast(fitted, interval_method = i_methods[3], horizon = 3)
            @test length(fc.mean) == 3

            fc = forecast(fitted, interval_method = i_methods[4], horizon = 3)
            @test length(fc.mean) == 3

            fc = forecast(fitted, interval_method = i_methods[5], horizon = 3)
            @test length(fc.mean) == 3

            if hasPI[i]
                fc = forecast(fitted, interval_method = i_methods[6], horizon = 3)
                @test length(fc.mean) == 3

                fc = forecast(fitted, interval_method = i_methods[7], horizon = 3)
                @test length(fc.mean) == 3
            end
            if hasMTI[i]
                fc = forecast(fitted, interval_method = i_methods[8], horizon = 3)
                @test length(fc.mean) == 3

                c = forecast(fitted, interval_method = i_methods[9], horizon = 3)
                @test length(fc.mean) == 3

                fc = forecast(fitted, interval_method = i_methods[10], horizon = 3)
                @test length(fc.mean) == 3

                fc = forecast(fitted, interval_method = i_methods[11], horizon = 3)
                @test length(fc.mean) == 3
            end
        end
    end
    
    @testset "Transformations Tests" begin
        # Test log transformation
        positive_data = abs.(stationary_data) .+ 1
        log_transform = LogTransform()
        transformed_data = transform(positive_data, log_transform)
        recovered_data = inverse_transform(transformed_data, log_transform)
        @test positive_data ≈ recovered_data
        
        # Test log-plus-one transformation
        log_plus_one = LogPlusOneTransform(1.0)
        transformed_data = transform(positive_data, log_plus_one)
        recovered_data = inverse_transform(transformed_data, log_plus_one)
        @test positive_data ≈ recovered_data
        
        # Test power transformation
        power_transform = PowerTransform(0.5)
        transformed_data = transform(positive_data, power_transform)
        recovered_data = inverse_transform(transformed_data, power_transform)
        @test positive_data ≈ recovered_data rtol=1e-10
        
        # Test transformed model
        base_model = ConstantModel()
        transformed_model = transform(base_model, transformation=log_transform)
        fitted = fit_baseline(positive_data, transformed_model)
        @test fitted isa TransformedFitted
        
        # Test forecasting with transformation
        forecast_result = forecast(fitted, interval_method = NoInterval(), horizon = 1:3)
        @test length(forecast_result.mean) == 3
    end
    
    @testset "Temporal Info Tests" begin
        # Test with Date temporal info
        ti = TemporalInfo(Date("2020-01-01"), Day(1))
        model = ConstantModel()
        fitted = fit_baseline(stationary_data, model, temporal_info=ti)
        
        forecast_result = forecast(fitted, interval_method = NoInterval(), horizon = 1:3, model_name="Temporal Test")
        @test has_temporal_info(forecast_result)
        @test forecast_result.reference_date isa Date
        @test length(forecast_result.target_date) == 3
        
        # Test with integer temporal info
        ti_int = TemporalInfo(1, 1)
        fitted_int = fit_baseline(stationary_data, model, temporal_info=ti_int)
        forecast_result_int = forecast(fitted_int, interval_method = NoInterval(), horizon = 1:3)
        @test forecast_result_int.reference_date isa Int
    end
    
    @testset "Forecast Object Tests" begin
        model = ConstantModel()
        fitted = fit_baseline(stationary_data, model)
        
        # Test basic forecast
        forecast_result = forecast(fitted, interval_method = NoInterval(), horizon = 1:5, model_name="Test Model")
        @test forecast_length(forecast_result) == 5
        @test max_horizon(forecast_result) == 5
        @test min_horizon(forecast_result) == 1
        @test forecast_result.model_name == "Test Model"
        
        # Test adding truth
        truth_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        forecast_with_truth = add_truth(forecast_result, truth_values)
        @test has_truth(forecast_with_truth)
        @test forecast_with_truth.truth == truth_values
        
        # Test truncating horizon
        truncated_forecast = truncate_horizon(forecast_result, 3)
        @test forecast_length(truncated_forecast) == 3
        @test max_horizon(truncated_forecast) == 3
        
        # Test filtering horizons
        filtered = filter_horizons(forecast_result, [1, 3, 5])
        @test forecast_length(filtered) == 3
        @test filtered.horizon == [1, 3, 5]
        
        # Test updating model name
        renamed = update_model_name(forecast_result, "New Name")
        @test renamed.model_name == "New Name"
    end
    
    @testset "Error Handling Tests" begin
        model = ConstantModel()
        
        # Test with insufficient data
        tiny_data = [1.0, 2.0]
        fitted = fit_baseline(tiny_data, model)
        @test fitted isa ConstantFitted
        
        # Test with negative horizons
        fitted = fit_baseline(stationary_data, model)
        @test_throws ArgumentError point_forecast(fitted, [-1, 2])
        @test_throws ArgumentError point_forecast(fitted, 0)
        
        # Test with invalid levels
        model = MarginalModel(p = 10)
        fitted = fit_baseline(stationary_data, model)
        @test_throws ArgumentError interval_forecast(fitted, ParametricInterval(), 1:2, [0.0, 1.0])
        @test_throws ArgumentError interval_forecast(fitted, ParametricInterval(), 1:2, [1.5])
        
        # Test ARMA stability checking
        arma_model = ARMAModel(p = 1, q = 1)
        unstable_data = cumsum(randn(20))  # Random walk (non-stationary)
        # Should still fit but may warn about stability
        fitted_arma = fit_baseline(unstable_data, arma_model)
        @test fitted_arma isa ARMAFitted
    end
    
    @testset "Edge Cases Tests" begin
        # Test with constant data
        constant_data = fill(5.0, 20)
        
        # Constant model
        model = ConstantModel()
        fitted = fit_baseline(constant_data, model)
        fc = point_forecast(fitted, 1:3)
        @test all(fc .== 5.0)
        
        # Marginal model
        marginal_model = MarginalModel()
        fitted_marginal = fit_baseline(constant_data, marginal_model)
        fc_marginal = point_forecast(fitted_marginal, 1:3)
        @test all(fc_marginal .== 5.0)
        
        # Test with single observation
        single_obs = [10.0]
        fitted_single = fit_baseline(single_obs, model)
        fc_single = point_forecast(fitted_single, 1)
        @test fc_single == [10.0]
    end
    
    @testset "Performance Tests" begin
        # Test that models can handle moderately large datasets
        large_data = randn(500)
        
        models_to_test = [
            ConstantModel(),
            MarginalModel(p = 50),
            LSDModel(s = 7, w = 1),
            OLSModel(p = 10, d = 1)
        ]
        
        for model in models_to_test
            fitted = fit_baseline(large_data, model)
            @test fitted isa AbstractFittedModel
            fc = point_forecast(fitted, 1:10)
            @test fc isa Vector{Float64}
            @test length(fc) == 10
        end
    end

    @testset "All Combinations" begin
        all_models = [MarginalModel(p = 10),
                  ConstantModel(),
                  KDEModel(),
                  LSDModel(s = 10, w = 1),
                  OLSModel(p = 10, d = 1),
                  IDSModel(p = 3),
                  ARMAModel(p = 1, q = 1),
                  INARCHModel(p = 1),
                  STLModel(),
                  ETSModel(trend = "A")]
    #
        i_methods = [NoInterval(),
                 EmpiricalInterval(n_trajectories = 50, positivity_correction = :none),
                 EmpiricalInterval(n_trajectories = 50, positivity_correction = :zero_floor),
                 EmpiricalInterval(n_trajectories = 50, positivity_correction = :truncate),
                 EmpiricalInterval(n_trajectories = 50, positivity_correction = :post_clip),
                 ParametricInterval(positivity_correction = :none),
                 ParametricInterval(positivity_correction = :post_clip),
                 ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :none),
                 ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :zero_floor),
                 ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :truncate),
                 ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :post_clip)]
        #
        hasPI = [true; false; fill(true, 5); fill(false, 3)]
        hasMTI = [true; false; fill(true, 8)]
    
        for i = 1:10
            if i == 8
                fitted = fit_baseline(count_data, all_models[i])
            else
                fitted = fit_baseline(stationary_data, all_models[i])
            end
        
            fc = forecast(fitted, interval_method = i_methods[1], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[2], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[3], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[4], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[5], horizon = 3)
            @test length(fc.mean) == 3
            if hasPI[i]
                fc = forecast(fitted, interval_method = i_methods[6], horizon = 3)
                @test length(fc.mean) == 3
                fc = forecast(fitted, interval_method = i_methods[7], horizon = 3)
                @test length(fc.mean) == 3
            end
            if hasMTI[i]
                fc = forecast(fitted, interval_method = i_methods[8], horizon = 3)
                @test length(fc.mean) == 3
                fc = forecast(fitted, interval_method = i_methods[9], horizon = 3)
                @test length(fc.mean) == 3
                fc = forecast(fitted, interval_method = i_methods[10], horizon = 3)
                @test length(fc.mean) == 3
                fc = forecast(fitted, interval_method = i_methods[11], horizon = 3)
                @test length(fc.mean) == 3
            end
        end
    end

    @testset "Scoring Rules Tests" begin
        # Generate forecasts with truth for scoring
        model = MarginalModel(p = 10)
        
        # Create forecasts with truth
        forecasts_with_truth = Forecast[]
        for i in 1:5
            fitted = fit_baseline(stationary_data[1:200 + i], model)
            fc = forecast(fitted, 
                         interval_method=ModelTrajectoryInterval(n_trajectories=50, return_trajectories=true),
                         horizon=1:3,
                         truth=stationary_data[201 + i:203 + i])
            push!(forecasts_with_truth, fc)
        end
        
        # Test point scoring rules
        mae_score = score(forecasts_with_truth, MAE())
        @test mae_score isa Float64
        @test mae_score >= 0
        
        mse_score = score(forecasts_with_truth, MSE())
        @test mse_score isa Float64
        @test mse_score >= 0
        
        rmse_score = score(forecasts_with_truth, RMSE())
        @test rmse_score isa Float64
        @test rmse_score >= 0
        @test rmse_score ≈ sqrt(mse_score)
        
        bias_score = score(forecasts_with_truth, Bias())
        @test bias_score isa Float64
        
        # Test by horizon scoring
        mae_by_horizon = score(forecasts_with_truth, MAE(), by_horizon=true)
        @test length(mae_by_horizon) == 3
        
        # Test specific horizon scoring
        mae_h1 = score(forecasts_with_truth, MAE(), horizon=1)
        @test mae_h1 isa Float64
        
        # Test interval scoring rules (need intervals)
        forecasts_with_intervals = Forecast[]
        for i in 1:3
            fitted = fit_baseline(stationary_data[1:200 + i], model)
            fc = forecast(fitted, 
                         interval_method=ParametricInterval(),
                         horizon=1:2,
                         levels=[0.8, 0.95],
                         include_median=true,
                         truth=stationary_data[201 + i:202 + i])
            push!(forecasts_with_intervals, fc)
        end
        
        wis_score = score(forecasts_with_intervals, WIS())
        @test wis_score isa Float64
        @test wis_score >= 0
        
        crps_score = score(forecasts_with_intervals, CRPS())
        @test crps_score isa Float64
        @test crps_score >= 0
        
        # Test trajectory scoring
        crps_traj_score = score(forecasts_with_truth, CRPS_trajectory())
        @test crps_traj_score isa Float64
        @test crps_traj_score >= 0
    end
    
    @testset "Calibration Tests" begin
        # Generate multiple forecasts for calibration testing
        model = MarginalModel(p = 20)
        fitted = fit_baseline(stationary_data, model)
        
        calibration_forecasts = Forecast[]
        for i in 1:10
            # Create synthetic truth values for testing
            synthetic_truth = [fitted.par.μ + 0.5*randn(), fitted.par.μ + 0.3*randn()]
            
            fc = forecast(fitted,
                         interval_method=ParametricInterval(),
                         horizon=1:2,
                         levels=[0.5, 0.8, 0.95],
                         truth=synthetic_truth)
            push!(calibration_forecasts, fc)
        end
        
        # Test PIT function creation
        pit_func = PIT_function(calibration_forecasts, horizon=1)
        @test pit_func isa Function
        
        # Test PIT function evaluation
        pit_values = pit_func.([0.1, 0.5, 0.9])
        @test length(pit_values) == 3
        @test all(0 .<= pit_values .<= 1)
        
        # Test CvM divergence
        cvm_score = CvM_divergence(calibration_forecasts, horizon=1)
        @test cvm_score isa Float64
        @test cvm_score >= 0
        
        # Test step function creation
        step = create_step(calibration_forecasts[1], horizon=1)
        @test step isa OneStepFunction
        @test 0 <= step.l <= 1
        @test 0 <= step.u <= 1
    end
    
    @testset "Seasonality Transformation Tests" begin
        # Test STTransform
        st_transform = STTransform(s=12, k=2, trend=true, additive=true)
        
        # Test pre-filtering
        filtered_data, st_params = preFilter(seasonal_data, st_transform)
        @test length(filtered_data) == length(seasonal_data)
        @test st_params isa STParameter
        @test length(st_params.θ) == 2
        @test length(st_params.κ) == 2
        
        # Fit model to filtered data
        model = ConstantModel()
        fitted = fit_baseline(filtered_data, model)
        
        # Generate forecast on filtered scale
        filtered_forecast = forecast(fitted, interval_method=NoInterval(), horizon=1:12)
        
        # Test post-filtering
        final_forecast = postFilter(seasonal_data, filtered_forecast, st_transform, st_params)
        @test has_mean(final_forecast)
        @test length(final_forecast.mean) == 12
        
        # Test multiplicative decomposition (with positive data)
        positive_seasonal = abs.(seasonal_data) .+ 10
        st_mult = STTransform(s=12, k=1, trend=false, additive=false)
        
        filtered_mult, params_mult = preFilter(positive_seasonal, st_mult)
        @test all(filtered_mult .> 0)  # Should remain positive after division
        
        fitted_mult = fit_baseline(filtered_mult, model)
        forecast_mult = forecast(fitted_mult, interval_method=NoInterval(), horizon=1:6)
        final_mult = postFilter(positive_seasonal, forecast_mult, st_mult, params_mult)
        @test all(final_mult.mean .> 0)
    end
    
    @testset "Historical Forecast Errors Tests" begin
        # Test with OLS model (simple enough for repeated fitting)
        model = OLSModel(p=5, d=1)
        fitted = fit_baseline(trend_data, model)
        
        # Test historical error computation
        hist_errors = historical_forecast_errors(fitted, 1:3, 10)
        @test length(hist_errors) == 3
        @test all(length.(hist_errors) .> 0)  # Should have some errors for each horizon
        
        # Test with different min_observation
        hist_errors_min = historical_forecast_errors(fitted, 1:2, 20)
        @test length(hist_errors_min) == 2
        
        # Errors should be Float64 vectors
        @test all(isa.(hist_errors[1], Float64))
    end
    
    @testset "Forecast Error Types Tests" begin
        # Create simple forecast with truth
        fc_simple = Forecast(
            horizon = collect(1:3),
            mean = [2.0, 2.5, 3.0],
            median = [1.9, 2.4, 2.9],
            truth = [2.1, 2.3, 3.2]
        )
        
        # Test different point error types
        @test forecast_error(fc_simple, ForecastError(), horizon=1) ≈ -0.1
        @test forecast_error(fc_simple, AbsoluteError(), horizon=1) ≈ 0.1
        @test forecast_error(fc_simple, SquaredError(), horizon=1) ≈ 0.01
        @test forecast_error(fc_simple, SignError(), horizon=1) == -1
        
        # Test relative errors (avoid zero truth values)
        @test forecast_error(fc_simple, ForecastError(relative=true), horizon=1) ≈ -0.1/2.1
        @test forecast_error(fc_simple, AbsoluteError(relative=true), horizon=1) ≈ 0.1/2.1
        
        # Test pinball error
        @test forecast_error(fc_simple, PinballError(0.5), horizon=1, target=:median) isa Float64
        
        # Test all forecast errors
        all_errors = all_forecast_errors(fc_simple, AbsoluteError())
        @test length(all_errors) == 3
        @test all(all_errors .>= 0)
        
        # Test with quantile targets
        fc_with_intervals = Forecast(
            horizon = collect(1:2),
            mean = [2.0, 2.5],
            intervals = [ForecastInterval([1.5], [2.5], [0.8]),
                        ForecastInterval([2.0], [3.0], [0.8])],
            truth = [2.2, 2.7]
        )
        
        # This should work if the quantile extraction is implemented
        q80_error = forecast_error(fc_with_intervals, AbsoluteError(), horizon=1, target=:q90)
        @test q80_error isa Union{Float64, Nothing, Missing}
    end
    
    @testset "Forecast Utilities Extended Tests" begin
        # Create complex forecast for testing utilities
        intervals = [ForecastInterval([1.0, 1.5], [3.0, 2.5], [0.5, 0.8]) for _ in 1:5]
        trajectories = randn(100, 5)
        
        complex_fc = Forecast(
            horizon = collect(1:5),
            mean = [2.0, 2.1, 2.2, 2.3, 2.4],
            median = [1.9, 2.0, 2.1, 2.2, 2.3],
            intervals = intervals,
            trajectories = trajectories,
            truth = [2.1, 2.0, 2.3, 2.2, 2.5],
            reference_date = Date("2023-12-31"),
            target_date = [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03"), 
                          Date("2024-01-04"), Date("2024-01-05")],
            model_name = "Complex Test"
        )
        
        # Test all has_* functions
        @test has_horizon(complex_fc)
        @test has_mean(complex_fc)
        @test has_median(complex_fc)
        @test has_intervals(complex_fc)
        @test has_truth(complex_fc)
        @test has_trajectories(complex_fc)
        @test has_reference_date(complex_fc)
        @test has_target_date(complex_fc)
        @test has_temporal_info(complex_fc)
        
        # Test trajectory properties
        @test num_trajectories(complex_fc) == 100
        
        # Test temporal span
        span = get_temporal_span(complex_fc)
        @test span == Day(4)
        
        # Test horizon range
        range = get_horizon_range(complex_fc)
        @test range == 1:5
        
        # Test filter_levels (need to implement if not already)
        filtered_fc = filter_levels(complex_fc, [0.8])
        @test has_intervals(filtered_fc)
        
        # Test forecast extension
        fc1 = Forecast(horizon=[1,2], mean=[1.0, 1.1], model_name="Model1")
        fc2 = Forecast(horizon=[3,4], mean=[1.2, 1.3], model_name="Model2")
        extended = extend_forecast(fc1, fc2)
        @test extended.horizon == [1,2,3,4]
        @test extended.mean == [1.0, 1.1, 1.2, 1.3]
        @test extended.model_name == "Model1 + Model2"
    end
    
    @testset "Complex Integration Tests" begin
        # Test model with both data transformation and seasonal preprocessing
        positive_seasonal = abs.(seasonal_data) .+ 1
        
        # Create complex transformed model
        base_model = MarginalModel(p=20)
        complex_model = transform(
            base_model,
            transformation = LogTransform(),
            season_trend = STTransform(s=12, k=1, trend=true, additive = false)
        )
        
        fitted_complex = fit_baseline(positive_seasonal, complex_model)
        @test fitted_complex isa TransformedFitted
        
        # Generate forecast with complex model
        complex_forecast = forecast(
            fitted_complex,
            interval_method = EmpiricalInterval(n_trajectories=50, return_trajectories=true),
            horizon = 1:6,
            levels = [0.8, 0.95]
        )
        
        @test has_mean(complex_forecast)
        @test has_intervals(complex_forecast)
        @test has_trajectories(complex_forecast)
        @test all(complex_forecast.mean .> 0)  # Should be positive after inverse transform
        
        # Test scoring on transformed forecasts
        truth_vals = positive_seasonal[end-5:end]
        complex_forecast_with_truth = add_truth(complex_forecast, truth_vals)
        
        mae_complex = score([complex_forecast_with_truth], MAE())
        @test mae_complex isa Float64
        @test mae_complex >= 0
    end
    
    @testset "Edge Cases and Error Handling Extended" begin
        # Test various error conditions
        model = ConstantModel()
        fitted = fit_baseline([1.0, 2.0, 3.0], model)
        
        # Test empty horizon
        @test_throws ArgumentError point_forecast(fitted, Int[])
        
        # Test invalid forecast construction
        @test_throws ArgumentError Forecast(horizon=[1,2], mean=[1.0])  # Length mismatch
        
        # Test invalid interval construction
        @test_throws ArgumentError ForecastInterval([2.0], [1.0], [0.95])  # Lower > upper
        @test_throws ArgumentError ForecastInterval([1.0], [2.0], [1.5])   # Invalid level
        
        # Test transformation with invalid data
        log_transform = LogTransform()
        @test_throws ArgumentError transform([-1.0, 2.0, 3.0], log_transform)  # Negative values
        
        # Test ARMA with insufficient data for degree
        arma_model = ARMAModel(p=5, q=3)
        tiny_data = [1.0, 2.0]
        # Should handle gracefully or throw appropriate error
        @test_throws Exception fit_baseline(tiny_data, arma_model)
    end
    
    @testset "Memory and Performance Edge Cases" begin
        # Test with moderately large trajectory matrices
        model = MarginalModel(p=50)
        fitted = fit_baseline(randn(100), model)
        
        # Large trajectory test
        large_traj_forecast = forecast(
            fitted,
            interval_method = ModelTrajectoryInterval(n_trajectories=1000, return_trajectories=true),
            horizon = 1:20
        )
        
        @test size(large_traj_forecast.trajectories) == (1000, 20)
        @test has_trajectories(large_traj_forecast)
        
        # Test forecast without returning trajectories (memory efficient)
        efficient_forecast = forecast(
            fitted,
            interval_method = ModelTrajectoryInterval(n_trajectories=1000, return_trajectories=false),
            horizon = 1:20
        )
        
        @test !has_trajectories(efficient_forecast)
        @test has_intervals(efficient_forecast)
    end
end