# Check for entries in the Forecast object
"""
    has_horizon(forecast::Forecast)

Check if forecast has horizon data.
"""
has_horizon(forecast::Forecast) = !isnothing(forecast.horizon)

"""
    has_mean(forecast::Forecast)

Check if forecast has mean predictions.
"""
has_mean(forecast::Forecast) = !isnothing(forecast.mean)

"""
    has_median(forecast::Forecast)

Check if forecast has median predictions.
"""
has_median(forecast::Forecast) = !isnothing(forecast.median)

"""
    has_intervals(forecast::Forecast)

Check if forecast has prediction intervals.
"""
has_intervals(forecast::Forecast) = !isnothing(forecast.intervals)

"""
    has_truth(forecast::Forecast)

Check if forecast has truth/observed data.
"""
has_truth(forecast::Forecast) = !isnothing(forecast.truth)

"""
    has_trajectories(forecast::Forecast)

Check if forecast has sample trajectories.
"""
has_trajectories(forecast::Forecast) = !isnothing(forecast.trajectories)

"""
    has_reference_date(forecast::Forecast)

Check if forecast has a reference date.
"""
has_reference_date(forecast::Forecast) = !isnothing(forecast.reference_date)

"""
    has_target_date(forecast::Forecast)

Check if forecast has target dates.
"""
has_target_date(forecast::Forecast) = !isnothing(forecast.target_date)

"""
    has_temporal_info(forecast::Forecast)

Check if forecast has both reference and target date information.
"""
has_temporal_info(forecast::Forecast) = has_reference_date(forecast) && has_target_date(forecast)

# Adding fields to Forecast obejcts

"""
    add_truth(forecast::Forecast, truth::Union{Real, Vector{<:Real}})

Add observed values to a forecast for evaluation.
"""
function add_truth(forecast::Forecast, truth::Union{Real, Vector{<:Real}})
    if truth isa Real
        truth = [Float64(truth)]
    else
        truth = Float64.(truth)
    end
    
    if has_horizon(forecast)
        length(truth) == length(forecast.horizon) || 
            throw(ArgumentError("Truth length ($(length(truth))) must match horizon length ($(length(forecast.horizon)))"))
    end
    
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=forecast.median,
        intervals=forecast.intervals,
        truth=truth,  # Updated field
        trajectories=forecast.trajectories,
        reference_date=forecast.reference_date,
        target_date=forecast.target_date,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    add_median(forecast::Forecast, median::Union{Real, Vector{<:Real}})
Add median predictions to a forecast.
"""
function add_median(forecast::Forecast, median::Union{Real, Vector{<:Real}})
    if median isa Real
        median = [Float64(median)]
    else
        median = Float64.(median)
    end
    
    if has_horizon(forecast)
        length(median) == length(forecast.horizon) || 
            throw(ArgumentError("Median length must match horizon length"))
    end
    
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=median,  # Updated field
        intervals=forecast.intervals,
        truth=forecast.truth,
        trajectories=forecast.trajectories,
        reference_date=forecast.reference_date,
        target_date=forecast.target_date,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    add_intervals(forecast::Forecast, intervals::Vector{ForecastInterval})

Add prediction intervals to a forecast.
"""
function add_intervals(forecast::Forecast, intervals::Vector{ForecastInterval})
    if has_horizon(forecast)
        length(intervals) == length(forecast.horizon) || 
            throw(ArgumentError("Intervals length must match horizon length"))
    end
    
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=forecast.median,
        intervals=intervals,  # Updated field
        truth=forecast.truth,
        trajectories=forecast.trajectories,
        reference_date=forecast.reference_date,
        target_date=forecast.target_date,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    add_trajectories(forecast::Forecast, trajectories::Union{Vector{<:Real}, Matrix{<:Real}})

Add sample trajectories to a forecast.
"""
function add_trajectories(forecast::Forecast, trajectories::Union{Vector{<:Real}, Matrix{<:Real}})
    if trajectories isa AbstractVector
        trajectories = reshape(Float64.(trajectories), (length(trajectories), 1))
    else
        trajectories = Float64.(trajectories)
    end
    
    if has_horizon(forecast)
        size(trajectories, 2) == length(forecast.horizon) || 
            throw(ArgumentError("Trajectories columns must match horizon length"))
    end
    
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=forecast.median,
        intervals=forecast.intervals,
        truth=forecast.truth,
        trajectories=trajectories,  # Updated field
        reference_date=forecast.reference_date,
        target_date=forecast.target_date,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    remove_trajectories(forecast::Forecast)

Removes sample trajectories from a forecast.
"""
function remove_trajectories(forecast::Forecast)
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=forecast.median,
        intervals=forecast.intervals,
        truth=forecast.truth,
        trajectories=nothing,  # Updated field
        reference_date=forecast.reference_date,
        target_date=forecast.target_date,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    add_temporal_info(forecast::Forecast, reference_date, resolution=Week(1))

Add temporal information (reference date and auto-generate target dates).
"""
function add_temporal_info(forecast::Forecast, reference_date::Union{DateTime, Date, Int}; 
                          resolution::Union{DatePeriod, Int} = Week(1))
    target_date = nothing
    if has_horizon(forecast)
        target_date = generate_target_dates(reference_date, forecast.horizon, resolution)
    end
    
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=forecast.median,
        intervals=forecast.intervals,
        truth=forecast.truth,
        trajectories=forecast.trajectories,
        reference_date=reference_date,  # Updated field
        target_date=target_date,        # Updated field
        resolution=resolution,          # Updated field
        model_name=forecast.model_name
    )
end

"""
    update_model_name(forecast::Forecast, model_name::String)

Update the model name in a forecast.
"""
function update_model_name(forecast::Forecast, model_name::String)
    return Forecast(
        horizon=forecast.horizon,
        mean=forecast.mean,
        median=forecast.median,
        intervals=forecast.intervals,
        truth=forecast.truth,
        trajectories=forecast.trajectories,
        reference_date=forecast.reference_date,
        target_date=forecast.target_date,
        resolution=forecast.resolution,
        model_name=model_name  # Updated field
    )
end

"""
    truncate_horizon(forecast::Forecast, max_horizon::Int)

Truncate forecast to a maximum horizon.
"""
function truncate_horizon(forecast::Forecast, max_horizon::Int)
    (!has_horizon(forecast) || max_horizon >= maximum(forecast.horizon)) && return forecast
    
    keep_indices = findall(h -> h <= max_horizon, forecast.horizon)
    isempty(keep_indices) && throw(ArgumentError("No horizons <= $max_horizon found"))
    
    return Forecast(
        horizon=forecast.horizon[keep_indices],
        mean=has_mean(forecast) ? forecast.mean[keep_indices] : nothing,
        median=has_median(forecast) ? forecast.median[keep_indices] : nothing,
        intervals=has_intervals(forecast) ? forecast.intervals[keep_indices] : nothing,
        truth=has_truth(forecast) ? forecast.truth[keep_indices] : nothing,
        trajectories=has_trajectories(forecast) ? forecast.trajectories[:, keep_indices] : nothing,
        reference_date=forecast.reference_date,
        target_date=has_target_date(forecast) ? forecast.target_date[keep_indices] : nothing,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    filter_horizons(forecast::Forecast, horizons::Union{Vector{Int}, Nothing}

Keep only specified horizons from a forecast.
"""
function filter_horizons(forecast::Forecast, horizons::Union{Vector{Int}, Nothing} = nothing)
    if isnothing(horizons)
        return forecast
    end
    !has_horizon(forecast) && return forecast
    
    keep_indices = [findfirst(==(h), forecast.horizon) for h in horizons]
    
    if any(isnothing, keep_indices)
        missing_horizons = horizons[isnothing.(keep_indices)]
        throw(ArgumentError("Horizons not found in forecast: $missing_horizons"))
    end
    
    return Forecast(
        horizon=forecast.horizon[keep_indices],
        mean=has_mean(forecast) ? forecast.mean[keep_indices] : nothing,
        median=has_median(forecast) ? forecast.median[keep_indices] : nothing,
        intervals=has_intervals(forecast) ? forecast.intervals[keep_indices] : nothing,
        truth=has_truth(forecast) ? forecast.truth[keep_indices] : nothing,
        trajectories=has_trajectories(forecast) ? forecast.trajectories[:, keep_indices] : nothing,
        reference_date=forecast.reference_date,
        target_date=has_target_date(forecast) ? forecast.target_date[keep_indices] : nothing,
        resolution=forecast.resolution,
        model_name=forecast.model_name
    )
end

"""
    filter_levels(forecast::Forecast, levels::Union{Vector{Float64}, Nothing})
    filter_levels(interval::ForecastInterval, levels::Union{Vector{Float64})
    
Keep only specified levels from a forecast.
"""
function filter_levels(interval::ForecastInterval, levels::Union{Vector{Float64}, Nothing} = nothing)
    if isnothing(interval)
        return interval
    end
    keep = (l -> l in levels).(interval.levels)
    ForecastInterval(interval.lower[keep], interval.upper[keep], interval.levels[keep])
end

function filter_levels(fc::Forecast, levels::Union{Vector{Float64}, Nothing} = nothing)
    if isnothing(levels) || isnothing(fc.intervals)
        return fc
    end
    
    intervalsOut = (int -> filter_levels(int, levels)).(fc.intervals)
    
    return Forecast(
        horizon = fc.horizon,
        mean = fc.mean,
        median = fc.median,
        intervals = intervalsOut,
        truth = fc.truth,
        trajectories = fc.trajectories,
        reference_date = fc.reference_date,
        target_date = fc.target_date,
        resolution = fc.resolution,
        model_name = fc.model_name
    )
end

"""
    extend_forecast(forecast1::Forecast, forecast2::Forecast)

Combine two forecasts with non-overlapping horizons.
"""
function extend_forecast(forecast1::Forecast, forecast2::Forecast)
    if has_horizon(forecast1) && has_horizon(forecast2)
        overlap = intersect(forecast1.horizon, forecast2.horizon)
        isempty(overlap) || throw(ArgumentError("Forecasts have overlapping horizons: $overlap"))
    end

    if has_reference_date(forecast1) && has_reference_date(forecast2)
        forecast1.reference_date == forecast2.reference_date || throw(ArgumentError("Forecasts must have the same reference date"))
    end

    forecast1.resolution == forecast2.resolution || throw(ArgumentError("Forecasts must have the same resolution"))
    
    combined_horizon = vcat(forecast1.horizon, forecast2.horizon)
    sort_indices = sortperm(combined_horizon)
    
    # Merge fields only if both forecasts privide it
    function combine_field(field1, field2)
        if isnothing(field1) && isnothing(field2)
            return nothing
        elseif isnothing(field1)
            return field2
        elseif isnothing(field2)
            return field1
        else
            return vcat(field1, field2)[sort_indices]
        end
    end
    
    combined_trajectories = nothing
    if has_trajectories(forecast1) && has_trajectories(forecast2)
        combined_trajectories = hcat(forecast1.trajectories, forecast2.trajectories)[:, sort_indices]
    elseif has_trajectories(forecast1)
        combined_trajectories = forecast1.trajectories
    elseif has_trajectories(forecast2)
        combined_trajectories = forecast2.trajectories
    end
    
    return Forecast(
        horizon=combined_horizon[sort_indices],
        mean=combine_field(forecast1.mean, forecast2.mean),
        median=combine_field(forecast1.median, forecast2.median),
        intervals=combine_field(forecast1.intervals, forecast2.intervals),
        truth=combine_field(forecast1.truth, forecast2.truth),
        trajectories=combined_trajectories,
        reference_date=forecast1.reference_date,
        target_date=combine_field(forecast1.target_date, forecast2.target_date),
        resolution=forecast1.resolution,
        model_name="$(forecast1.model_name) + $(forecast2.model_name)"
    )
end

# Some other functions
"""
    forecast_length(forecast::Forecast)

Get the number of forecast horizons.
"""
function forecast_length(forecast::Forecast)
    has_horizon(forecast) ? length(forecast.horizon) : 0
end

"""
    max_horizon(forecast::Forecast)

Get the maximum forecast horizon.
"""
function max_horizon(forecast::Forecast)
    has_horizon(forecast) ? maximum(forecast.horizon) : nothing
end

"""
    min_horizon(forecast::Forecast)

Get the minimum forecast horizon.
"""
function min_horizon(forecast::Forecast)
    has_horizon(forecast) ? minimum(forecast.horizon) : nothing
end

"""
    get_horizon_range(forecast::Forecast)

Get the range of forecast horizons.
"""
function get_horizon_range(forecast::Forecast)
    if has_horizon(forecast)
        return minimum(forecast.horizon):maximum(forecast.horizon)
    else
        return nothing
    end
end

"""
    num_trajectories(forecast::Forecast)

Get the number of sample trajectories.
"""
function num_trajectories(forecast::Forecast)
    has_trajectories(forecast) ? size(forecast.trajectories, 1) : 0
end

"""
    get_temporal_span(forecast::Forecast)

Get the time span covered by the forecast.
"""
function get_temporal_span(forecast::Forecast)
    if has_temporal_info(forecast) && length(forecast.target_date) > 1
        forecast.target_date[end] - forecast.target_date[1]
    else
        return nothing
    end
end

# Functions for console
"""
    summary(forecast::Forecast)

Print a summary of the forecast object.
"""
function Base.summary(forecast::Forecast)
    println("Forecast Summary:")
    println("  Model: $(forecast.model_name)")
    println("  Horizons: $(has_horizon(forecast) ? length(forecast.horizon) : 0) steps")
    
    if has_horizon(forecast)
        println("  Horizon range: $(minimum(forecast.horizon)) to $(maximum(forecast.horizon))")
    end
    
    println("  Components:")
    println("    Mean: $(has_mean(forecast) ? "✓" : "✗")")
    println("    Median: $(has_median(forecast) ? "✓" : "✗")")
    println("    Intervals: $(has_intervals(forecast) ? "✓" : "✗")")
    println("    Truth: $(has_truth(forecast) ? "✓" : "✗")")
    println("    Trajectories: $(has_trajectories(forecast) ? "$(num_trajectories(forecast)) samples" : "✗")")
    
    if has_temporal_info(forecast)
        println("  Temporal:")
        println("    Reference: $(forecast.reference_date)")
        println("    Resolution: $(forecast.resolution)")
        if length(forecast.target_date) <= 3
            println("    Targets: $(forecast.target_date)")
        else
            println("    Targets: $(forecast.target_date[1]) to $(forecast.target_date[end])")
        end
    end
end

"""
    Base.show(io::IO, forecast::Forecast)

Compact one-line display of forecast (for arrays, REPL inline display).
"""
function Base.show(io::IO, forecast::Forecast)
    if has_horizon(forecast)
        n = length(forecast.horizon)
        horizon_range = "$(minimum(forecast.horizon)):$(maximum(forecast.horizon))"
        print(io, "Forecast{$(n) horizons, $(horizon_range), $(forecast.model_name)}")
    else
        print(io, "Forecast{empty, $(forecast.model_name)}")
    end
end


"""
    show_forecast_table(io::IO, forecast::Forecast; preview::Bool=false)

Show forecast data in a tabular format.
"""
function show_forecast_table(io::IO, forecast::Forecast; preview::Bool=false)
    if !has_horizon(forecast) || !has_mean(forecast)
        println(io, "    No data to display")
        return
    end
    
    n = length(forecast.horizon)
    
    if preview && n > 7
        show_indices = [1:5; (n-1):n]
        show_ellipsis = true
    else
        show_indices = 1:n
        show_ellipsis = false
    end
    
    headers = ["Horizon"]
    has_target_date(forecast) && push!(headers, "Target")
    has_mean(forecast) && push!(headers, "Mean")
    has_median(forecast) && push!(headers, "Median")
    has_truth(forecast) && push!(headers, "Truth")
    (has_intervals(forecast) && all((int -> any(int.levels .≈ 0.95)).(forecast.intervals))) && push!(headers, "95% PI")
    
    col_widths = [max(length(h), 8) for h in headers]
    
    print(io, "    ")
    for (i, header) in enumerate(headers)
        print(io, rpad(header, col_widths[i] + 2))
    end
    println(io)
    
    print(io, "    ")
    for width in col_widths
        print(io, "─"^(width + 2))
    end
    println(io)
    
    for (row_idx, i) in enumerate(show_indices)
        if show_ellipsis && row_idx == 6
            print(io, "    ")
            for width in col_widths
                print(io, rpad("⋮", width + 2))
            end
            println(io)
        end
        
        print(io, "    ")
        
        # Horizon
        print(io, rpad(string(forecast.horizon[i]), col_widths[1] + 2))
        
        # Target date
        col = 2
        if has_target_date(forecast)
            target_str = format_target_date(forecast.target_date[i])
            print(io, rpad(target_str, col_widths[col] + 2))
            col += 1
        end
        
        # Mean
        if has_mean(forecast)
            mean_str = @sprintf("%.2f", forecast.mean[i])
            print(io, rpad(mean_str, col_widths[col] + 2))
            col += 1
        end
        
        # Median
        if has_median(forecast)
            median_str = @sprintf("%.2f", forecast.median[i])
            print(io, rpad(median_str, col_widths[col] + 2))
            col += 1
        end
        
        # Truth
        if has_truth(forecast)
            truth_str = @sprintf("%.2f", forecast.truth[i])
            print(io, rpad(truth_str, col_widths[col] + 2))
            col += 1
        end
        
        # Intervals
        if has_intervals(forecast) && all((int -> any(int.levels .≈ 0.95)).(forecast.intervals))
            interval_str = format_prediction_interval(forecast.intervals[i])
            print(io, rpad(interval_str, col_widths[col] + 2))
        end
        
        println(io)
    end
end


"""
    format_target_date(date::Union{DateTime, Date, Int})

Format target date for display.
"""
function format_target_date(date::Union{DateTime, Date, Int})
    if date isa Int
        return string(date)
    elseif date isa Date
        return string(date)
    else
        return Dates.format(date, "yyyy-mm-dd")
    end
end

"""
    format_prediction_interval(interval::ForecastInterval)

Format prediction interval for display (show 95% interval if available).
"""
function format_prediction_interval(interval::ForecastInterval)
    level = 0.95
    level_idx = findfirst(interval.levels .≈ level)
    
    lower = interval.lower[level_idx]
    upper = interval.upper[level_idx]
    # level = interval.levels[level_idx]
    
    return @sprintf("[%.1f, %.1f]", lower, upper)
end

"""
    Base.show(io::IO, ::MIME"text/plain", forecast::Forecast)

Detailed multi-line display of forecast (when forecast is the result of an expression).
"""
function Base.show(io::IO, ::MIME"text/plain", forecast::Forecast)
    println(io, "Forecast")
    println(io, "  Model: $(forecast.model_name)")
    
    if has_horizon(forecast)
        n = length(forecast.horizon)
        println(io, "  Horizons: $n step$(n == 1 ? "" : "s") ($(minimum(forecast.horizon)) to $(maximum(forecast.horizon)))")
        
        # Show temporal information if available
        if has_temporal_info(forecast)
            println(io, "  Reference: $(forecast.reference_date)")
            println(io, "  Resolution: $(forecast.resolution)")
            
            if length(forecast.target_date) <= 5
                println(io, "  Targets: $(forecast.target_date)")
            else
                println(io, "  Targets: $(forecast.target_date[1]) to $(forecast.target_date[end]) ($(length(forecast.target_date)) dates)")
            end
        end
        
        # Show available components
        components = String[]
        has_mean(forecast) && push!(components, "mean")
        has_median(forecast) && push!(components, "median")
        has_intervals(forecast) && push!(components, "intervals")
        has_truth(forecast) && push!(components, "truth")
        has_trajectories(forecast) && push!(components, "trajectories ($(size(forecast.trajectories, 1)) samples)")
        
        if !isempty(components)
            println(io, "  Components: $(join(components, ", "))")
        end
        
        # Show data preview if forecast is small
        if n <= 10 && has_mean(forecast)
            println(io, "  Data preview:")
            show_forecast_table(io, forecast)
        elseif n > 10 && has_mean(forecast)
            println(io, "  Data preview (first 5, last 2):")
            show_forecast_table(io, forecast, preview=true)
        end
    else
        println(io, "  Status: Empty forecast (no data)")
    end
end

"""
    get_times(t::Int, temporal_info::TemporalInfo)

Get times for for an index `t` using temporal information.
If `t` is a vector, it returns times for 1, ..., length(t)
"""
function get_times(t::Int, temporal_info::TemporalInfo)
    temporal_info.start + t * temporal_info.resolution
end
function get_times(t::Vector{T}, temporal_info::TemporalInfo) where {T <: Real}
    temporal_info.start .+ collect(1:length(t)) .* temporal_info.resolution
end
