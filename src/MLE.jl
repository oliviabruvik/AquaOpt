include("cleaning.jl")
include("SeaLicePOMDP.jl")
include("plot_views.jl")
include("optimization.jl")

# Environment variables
ENV["PLOTS_BROWSER"] = "true"
ENV["PLOTS_BACKEND"] = "plotlyjs"

# ----------------------------# Import required packages
using Logging
using DiscreteValueIteration
using GridInterpolations
using NativeSARSOP: SARSOPSolver
using POMDPs
using POMDPTools
using Plots
using StatsPlots  # Add StatsPlots for boxplot support
using LocalFunctionApproximation
using LocalApproximationValueIteration
using StatsPlots
using Optim
using DataFrames
using Statistics

# TODO: sensitivity analysis for epsilon
# TODO: sensitivity analysis for use_consecutive_weeks

plotlyjs()  # Activate Plotly backend

LUSEDATA_PATH = "data/processed/sealice_data.csv"
BARETSWATCH_PATH = "data/processed/bayesian_outer_data.csv"

function main()
    # @info "Running MLE analysis for Luse data"
    # luse_data_df = CSV.read(LUSEDATA_PATH, DataFrame)
    # luse_data_df = convert_lusedata_to_baretswatch_format(luse_data_df)
    # mle_analysis(luse_data_df)

    # Reduces REMSE from 8 to 0.34
    use_consecutive_weeks = true
    run_algos = true
    
    baretswatch_data_df = CSV.read(BARETSWATCH_PATH, DataFrame)
    baretswatch_data_df = clean_baretswatch_data(baretswatch_data_df, false, 1e-8)

    if run_algos

        @info "Running MLE analysis for Baretswatch data in in normal space"
        # Natural for growth in log
        location_to_growth_rate = mle_analysis_by_location(baretswatch_data_df, "log-space-normal", use_consecutive_weeks)
        location_year_to_growth_rate = mle_analysis_by_location_year(baretswatch_data_df, "log-space-normal", use_consecutive_weeks)

        # Evaluate growth rates
        evaluate_growth_rates_by_location(location_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "log-space-normal")
        evaluate_growth_rates_by_location_year(location_year_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "log-space-normal")
        plot_growth_rates_by_year_and_location(location_year_to_growth_rate, baretswatch_data_df, "log-space-normal")
        plot_mean_growth_rates_by_year_and_location(location_year_to_growth_rate, baretswatch_data_df, "log-space-normal")
        list_top_ten_locations(location_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "log-space-normal")

        @info "Running MLE analysis for Baretswatch data in in log normal space"
        # Best when working in count space
        location_to_growth_rate = mle_analysis_by_location(baretswatch_data_df, "raw-space-log-normal", use_consecutive_weeks)
        location_year_to_growth_rate = mle_analysis_by_location_year(baretswatch_data_df, "raw-space-log-normal", use_consecutive_weeks)

        # Evaluate growth rates
        evaluate_growth_rates_by_location(location_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "raw-space-log-normal")
        evaluate_growth_rates_by_location_year(location_year_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "raw-space-log-normal")
        plot_growth_rates_by_year_and_location(location_year_to_growth_rate, baretswatch_data_df, "raw-space-log-normal")
        plot_mean_growth_rates_by_year_and_location(location_year_to_growth_rate, baretswatch_data_df, "raw-space-log-normal")
        list_top_ten_locations(location_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "raw-space-log-normal")

        @info "Running MLE analysis for Baretswatch data in in raw-space"
        # Not appropriate - can go negative
        location_to_growth_rate = mle_analysis_by_location(baretswatch_data_df, "raw-space-normal", use_consecutive_weeks)
        location_year_to_growth_rate = mle_analysis_by_location_year(baretswatch_data_df, "raw-space-normal", use_consecutive_weeks)

        # Evaluate growth rates
        evaluate_growth_rates_by_location(location_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "raw-space-normal")
        evaluate_growth_rates_by_location_year(location_year_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "raw-space-normal")
        plot_growth_rates_by_year_and_location(location_year_to_growth_rate, baretswatch_data_df, "raw-space-normal")
        plot_mean_growth_rates_by_year_and_location(location_year_to_growth_rate, baretswatch_data_df, "raw-space-normal")
        list_top_ten_locations(location_to_growth_rate, baretswatch_data_df, use_consecutive_weeks, "raw-space-normal")

    end

end

########################################################
# MLE Analysis for Sea Lice Growth Rate - Lusedata
########################################################
function convert_lusedata_to_baretswatch_format(df)

    # Rename mechanical_removal to treatment
    df.treatment = df.mechanical_removal

    # Rename location_number to site_number
    df.site_number = df.location_number

    return df
end

function clean_baretswatch_data(df, filter_zeros=false, epsilon=0.0)
    
    # Handle missing values and convert treatment column to binary
    df.treatment = ifelse.(df.treatment .== "None", false, true)

    # Drop rows where lice count is not available or relevant for analyses
    df = dropmissing(df, [:adult_sealice])

    if filter_zeros
        df = filter(row -> row.adult_sealice > 0, df)
    end

    df.adult_sealice = df.adult_sealice .+ epsilon

    return df
end

########################################################
# MLE Analysis for Sea Lice Growth Rate - Baretswatchdata
########################################################
function mle_analysis_by_location(df, space="log-space-normal", use_consecutive_weeks=false)

    # Split into separate dataframes for each location
    locations = unique(df.site_number)

    # Dict of location number and year to growth rate
    location_to_growth_rate = Dict{Int64, Float64}()
    
    # Dict of location number and year to growth rate
    for location in locations

        # Get location dataframe
        location_df = df[df.site_number .== location, :]

        # Run optimization: # Initial guess: r = 0.1, σ = 0.1
        # result = optimize(p -> baretswatch_log_likelihood(p, location_df, space), [0.1, 0.1],
        #          lower=[-1.0, 1e-6], upper=[1.0, 5.0], autodiff = :forward)

        result = optimize(r -> baretswatch_log_likelihood(r, location_df, space, use_consecutive_weeks), [0.0], autodiff = :forward, show_every = 100)
        r_hat = Optim.minimizer(result)

        rounded_result = round(r_hat[1], digits=4)

        # println("MLE for growth rate at location $location: $rounded_result")
        location_to_growth_rate[location] = r_hat[1]
    end

    return location_to_growth_rate
end

function mle_analysis_by_location_year(df, space="log-space-normal", use_consecutive_weeks=false)

    # Split into separate dataframes for each location
    locations = unique(df.site_number)
    years = unique(df.year)

    println("Number of locations: $(length(locations))")
    println("Number of years: $(length(years))")

    # Dict of location number and year to growth rate
    location_year_to_growth_rate = Dict{Tuple{Int64, Int64}, Float64}()
    
    # Dict of location number and year to growth rate
    for location in locations

        # Get location dataframe
        location_df = df[df.site_number .== location, :]

        for year in years

            # Get location and year dataframe
            location_year_df = location_df[location_df.year .== year, :]

            result = optimize(r -> baretswatch_log_likelihood(r, location_year_df, space, use_consecutive_weeks), [0.0], autodiff = :forward)
            r_hat = Optim.minimizer(result)
            
            rounded_result = round(r_hat[1], digits=4)

            # println("MLE for growth rate at location $location: $rounded_result")
            location_year_to_growth_rate[(location, year)] = r_hat[1]
        end
    end

    return location_year_to_growth_rate
end

########################################################
# Log likelihood function for MLE analysis
########################################################
# Define log likelihood function for MLE analysis
function baretswatch_log_likelihood(r, df, space="log-space-normal", use_consecutive_weeks=false)

    # r = r[1]
    σ = 0.1

    log_likelihood = 0.0
    for t in 2:length(df.adult_sealice)
        if use_data_for_mle_treatment(df, t) && use_data_zeros(df, t)
            if use_consecutive_weeks && use_data_for_mle_week(df, t)
                log_likelihood += calculate_log_likelihood(r, df, t, σ, 1, space)
            elseif !use_consecutive_weeks
                # Decreases from 0.10 to 0.05
                week_delta = df.total_week[t] - df.total_week[t-1]
                log_likelihood += calculate_log_likelihood(r, df, t, σ, week_delta, space)
            end
        end
    end
    return -log_likelihood
end

function calculate_log_likelihood(r, df, t, σ, week_delta, space="log-space-normal")
    # Convert week_delta to Float64 to ensure consistent types
    week_delta = Float64(week_delta)
    
    if space == "log-space-normal"
        # Log-space normal model: log(N_k+1) ~ N(log(N_k) + r * week_delta, sd)
        # This models exponential growth in log space
        μ = log(df.adult_sealice[t-1]) + r[1] * week_delta
        return logpdf(Normal(μ, σ), log(df.adult_sealice[t]))
    elseif space == "raw-space-log-normal"
        # Raw-space log normal model: N_k+1 ~ LogNormal(log(N_k) + r * week_delta, sd)
        # This models multiplicative growth with log-normal errors
        μ = log(df.adult_sealice[t-1]) + r[1] * week_delta
        # log_μ = log(μ) - σ^2/2
        # return logpdf(LogNormal(log_μ, σ), df.adult_sealice[t])
        return logpdf(LogNormal(μ, σ), df.adult_sealice[t])
    elseif space == "raw-space-normal"
        # Raw-space normal model: N_k+1 ~ N(N_k * exp(r * week_delta), sd)
        # This models exponential growth with normal errors
        μ = df.adult_sealice[t-1] * exp(r[1] * week_delta)
        return logpdf(Normal(μ, σ), df.adult_sealice[t])
    else
        error("Invalid space type: $space. Must be one of: log-space-normal, raw-space-log-normal, raw-space-normal")
    end
end



########################################################
# Evaluate growth rates by location and year
########################################################
function evaluate_growth_rates_by_location(location_to_growth_rate, df, use_consecutive_weeks, space, save_plots=false)
    mean_growth_rate = round(mean(values(location_to_growth_rate)), digits=4)
    var_growth_rate = round(var(values(location_to_growth_rate)), digits=4)
    preds, actuals = evaluate_r(mean_growth_rate, df, use_consecutive_weeks)
    mae, rmse, r2 = evaluate_metrics(preds, actuals)
    mae, rmse, r2 = round(mae, digits=4), round(rmse, digits=4), round(r2, digits=4)
    println("By location: Mean growth rate: $mean_growth_rate, Variance: $var_growth_rate, MAE: $mae, RMSE: $rmse, R2: $r2")

    # Save mae, rmse, r2 for all locations as csv
    locations = []
    maes = []
    rmses = []
    r2s = []
    growth_rates = []

    for location in keys(location_to_growth_rate)
        push!(locations, location)
        location_df = df[df.site_number .== location, :]
        growth_rate = location_to_growth_rate[location]
        preds, actuals = evaluate_r(growth_rate, location_df, use_consecutive_weeks)
        mae, rmse, r2 = evaluate_metrics(preds, actuals)

        # Save results
        push!(maes, mae)
        push!(rmses, rmse)
        push!(r2s, r2)
        push!(growth_rates, growth_rate)

        if save_plots
            p = plot_predictions(preds, actuals, space)
            mkpath("results/figures/MLE/$space")
            savefig(p, "results/figures/MLE/$space/location_$(location)_predictions.png")
        end
    end

    mean_growth_rate = round(mean(growth_rates), digits=4)
    mean_mae = round(mean(maes), digits=4)
    mean_rmse = round(mean(rmses), digits=4)
    mean_r2 = round(mean(r2s), digits=4)
    var_growth_rate = round(var(growth_rates), digits=4)

    println("By location (individual): Mean growth rate: $mean_growth_rate, Variance: $var_growth_rate, MAE: $mean_mae, RMSE: $mean_rmse, R2: $mean_r2")

    # Save mae, rmse, r2 for each location and year as csv
    save_df = DataFrame(location=locations, growth_rate=growth_rates, mae=maes, rmse=rmses, r2=r2s)
    CSV.write("results/figures/MLE/$space/location_predictions.csv", save_df)
end

function evaluate_growth_rates_by_location_year(location_year_to_growth_rate, df, use_consecutive_weeks, space, save_plots=false)
    mean_growth_rate = round(mean(values(location_year_to_growth_rate)), digits=4)
    var_growth_rate = round(var(values(location_year_to_growth_rate)), digits=4)
    preds, actuals = evaluate_r(mean_growth_rate, df, use_consecutive_weeks)
    mae, rmse, r2 = evaluate_metrics(preds, actuals)
    mae, rmse, r2 = round(mae, digits=4), round(rmse, digits=4), round(r2, digits=4)
    println("By location, year:Mean growth rate: $mean_growth_rate, Variance: $var_growth_rate, MAE: $mae, RMSE: $rmse, R2: $r2")

    p = plot_predictions(preds, actuals, space)
    mkpath("results/figures/MLE/$space")
    savefig(p, "results/figures/MLE/$space/mean_predictions_$(space).png")
    println("Saved plot to results/figures/MLE/$space/mean_predictions_$(space).png")

    # Save mae, rmse, r2 for all locations as csv
    locations = []
    maes = []
    rmses = []
    r2s = []
    years = []
    growth_rates = []

    for location_year in keys(location_year_to_growth_rate)
        location_year_df = df[(df.site_number .== location_year[1]) .& (df.year .== location_year[2]), :]
        growth_rate = location_year_to_growth_rate[location_year]
        preds, actuals = evaluate_r(growth_rate, location_year_df, use_consecutive_weeks)
        mae, rmse, r2 = evaluate_metrics(preds, actuals)

        # Save results
        push!(maes, mae)
        push!(rmses, rmse)
        push!(r2s, r2)
        push!(years, location_year[2])
        push!(locations, location_year[1])
        push!(growth_rates, growth_rate)
        
        if save_plots
            p = plot_predictions(preds, actuals, space)
            mkpath("results/figures/MLE/$space")
            savefig(p, "results/figures/MLE/$space/location_$(location_year[1])_year_$(location_year[2])_predictions.png")
        end

    end

    mean_growth_rate = round(mean(growth_rates), digits=4)
    mean_mae = round(mean(maes), digits=4)
    mean_rmse = round(mean(rmses), digits=4)
    mean_r2 = round(mean(r2s), digits=4)
    var_growth_rate = round(var(growth_rates), digits=4)

    println("By location, year (individual): Mean growth rate: $mean_growth_rate, Variance: $var_growth_rate, MAE: $mean_mae, RMSE: $mean_rmse, R2: $mean_r2")


    # Save mae, rmse, r2 for each location and year as csv
    save_df = DataFrame(location=locations, year=years, growth_rate=growth_rates, mae=maes, rmse=rmses, r2=r2s)
    CSV.write("results/figures/MLE/$space/location_year_predictions.csv", save_df)

    return save_df
end


########################################################
# Evaluate r for a given r and data
########################################################
function evaluate_r(r, df, use_consecutive_weeks)
    preds = Float64[]
    actuals = Float64[]
    
    # Extract scalar value from vector
    r_value = r[1]

    for t in 2:length(df.adult_sealice)
        Δt = df.total_week[t] - df.total_week[t-1]
        if use_data_for_mle_treatment(df, t) && use_data_zeros(df, t)
            if use_consecutive_weeks && use_data_for_mle_week(df, t)
                predicted = df.adult_sealice[t-1] * exp(r_value)
                push!(preds, predicted)
                push!(actuals, df.adult_sealice[t])
            elseif !use_consecutive_weeks
                predicted = df.adult_sealice[t-1] * exp(r_value * Δt)
                push!(preds, predicted)
                push!(actuals, df.adult_sealice[t])
            end
        end
    end
    return preds, actuals
end

function evaluate_metrics(preds, actuals)
    residuals = actuals .- preds
    mae = mean(abs.(residuals))
    rmse = sqrt(mean(residuals.^2))
    r2 = 1 - sum(residuals.^2) / sum((actuals .- mean(actuals)) .^ 2)
    return mae, rmse, r2
end

########################################################
# Plot predictions
########################################################
function plot_predictions(preds, actuals, space="log-space-normal")

    title_name = "Predicted vs Actual Lice Counts"
    if space == "log-space-normal"
        title_name = "Predicted vs Actual Lice Counts (Log Space)"
    elseif space == "raw-space-log-normal"
        title_name = "Predicted vs Actual Lice Counts (Raw Space Log-Normal)"
    elseif space == "raw-space-normal"
        title_name = "Predicted vs Actual Lice Counts (Raw Space)"
    end

    p = scatter(actuals, preds,
            xlabel="Actual Sea Lice Level",
            ylabel="Predicted Sea Lice Level",
            title="$title_name",
            label="Predicted",
            ylims=(0, 30),
            legend=:bottomright)
    plot!(x -> x, label="Perfect Prediction", linestyle=:dash)
    return p
end

function plot_growth_rates_by_year_and_location(location_year_to_growth_rate, df, space="log-space-normal")

    # Get unique years
    years = sort(unique(df.year))

    # Get unique locations
    locations = unique(df.site_number)

    p = plot(xlabel="Year", ylabel="Growth Rate", title="Growth Rate by Year and Location", legend=:topleft)

    # For each location, get the growth rate for each year
    for location in locations
        x = Int64[]
        y = Float64[]
        for year in years
            if (location, year) in keys(location_year_to_growth_rate)
                push!(x, year)
                push!(y, location_year_to_growth_rate[(location, year)])
            end
        end
        if length(x) > 0
            plot!(p, x, y, label="Location $location")
        end
    end

    # Save plot
    mkpath("results/figures/MLE/$space")
    savefig(p, "results/figures/MLE/$space/growth_rates_by_year_and_location.png")

    return p
end

function plot_mean_growth_rates_by_year_and_location(location_year_to_growth_rate, df, space="log-space-normal")

    # Get unique years and locations
    years = sort(unique(df.year))
    locations = unique(df.site_number)

    title_name = "Mean Growth Rate by Year with 95% CI"
    if space == "log-space-normal"
        title_name = "Mean Growth Rate by Year with 95% CI (Log Space)"
    elseif space == "raw-space-log-normal"
        title_name = "Mean Growth Rate by Year with 95% CI (Raw Space Log-Normal)"
    elseif space == "raw-space-normal"
        title_name = "Mean Growth Rate by Year with 95% CI (Raw Space)"
    end

    p = plot(xlabel="Year", ylabel="Growth Rate", title="$title_name", legend=:bottomright)

    x = Int64[]
    y = Float64[]
    lower_ci = Float64[]
    upper_ci = Float64[]

    for year in years
        push!(x, year)
        ys = Float64[]
        
        for location in locations
            if (location, year) in keys(location_year_to_growth_rate)
                push!(ys, location_year_to_growth_rate[(location, year)])
            end
        end
        
        mean_growth = mean(ys)
        push!(y, mean_growth)
        
        # Calculate 95% confidence interval
        n = length(ys)
        se = std(ys) / sqrt(n)
        t_value = quantile(TDist(n-1), 0.975)
        margin_error = t_value * se
        
        push!(lower_ci, mean_growth - margin_error)
        push!(upper_ci, mean_growth + margin_error)
    end

    plot!(p, x, y, ribbon=(y .- lower_ci, upper_ci .- y), fillalpha=0.3, label="Mean Growth Rate with 95% CI")

    # Save plot
    mkpath("results/figures/MLE/$space")
    savefig(p, "results/figures/MLE/$space/mean_growth_rates_by_year_and_location_with_ci_$(space).png")

    return p
end

function list_top_ten_locations(location_to_growth_rate, df, use_consecutive_weeks, space="log-space-normal")
    # Create arrays to store results
    locations = Int64[]
    growth_rates = Float64[]
    r2_scores = Float64[]
    
    # Calculate R² for each location
    for location in keys(location_to_growth_rate)
        location_df = df[df.site_number .== location, :]
        growth_rate = location_to_growth_rate[location]
        preds, actuals = evaluate_r(growth_rate, location_df, use_consecutive_weeks)
        _, _, r2 = evaluate_metrics(preds, actuals)
        
        push!(locations, location)
        push!(growth_rates, growth_rate)
        push!(r2_scores, r2)
    end
    
    # Create DataFrame and sort by R²
    results_df = DataFrame(
        location = locations,
        growth_rate = growth_rates,
        r2 = r2_scores
    )
    
    # Sort by R² in descending order and get top 10
    sorted_results = sort(results_df, :r2, rev=true)

    # Filter out locations with NaN and negative growth rates
    sorted_results = filter(row -> !isnan(row.r2) && row.growth_rate >= 0, sorted_results)

    # Get top 10 locations
    top_ten = first(sorted_results, 10)
    
    # Print results
    println("\nTop 10 Locations by Model Fit ($space):")
    println("----------------------------------------")
    for (i, row) in enumerate(eachrow(top_ten))
        println("Rank $i:")
        println("  Location: $(row.location)")
        println("  Growth Rate: $(round(row.growth_rate, digits=4))")
        println("  R² Score: $(round(row.r2, digits=4))")
        println("----------------------------------------")
    end
    
    # Save results to CSV
    mkpath("results/figures/MLE/$space")
    CSV.write("results/figures/MLE/$space/top_ten_locations.csv", top_ten)
    
    return top_ten
end

########################################################
# Constraints for MLE analysis
########################################################
# Constraints for MLE analysis
function use_data_for_mle_treatment(df, t)

    # If treatment is not conducted, then count
    if df.treatment[t] == false
        return true
    end

    # If treatment is conducted, then don't count if sea lice is not increasing
    if df.treatment[t] == true && df.adult_sealice[t-1] >= df.adult_sealice[t]
        # If treatment is conducted and sea lice is not increasing, then don't count
        return false
    elseif df.treatment[t-1] == true && df.treatment[t] == true
        # If treatment is conducted and the previous week was also treated, then don't count
        return false
    elseif df.treatment[t] == true && df.adult_sealice[t-1] < df.adult_sealice[t]
        # Count if treatment is conducted and sea lice is increasing
        # Represents that they counted sea lice next week before the treatment was conducted
        return true
    else
        return true
    end
end

function use_data_for_mle_week(df, t)

    # If week is not available, then don't count
    if df.total_week[t] != df.total_week[t-1] + 1
        return false
    else
        return true
    end
end

function use_data_zeros(df, t)

    # If adult sea lice is 0, then don't count
    if df.adult_sealice[t] == 0 || df.adult_sealice[t-1] == 0
        return false
    else
        return true
    end
end

function plot_epsilon_sensitivity_analysis(epsilon_values)
    # Initialize arrays to store metrics
    location_maes = Float64[]
    location_rmses = Float64[]
    location_r2s = Float64[]
    location_year_maes = Float64[]
    location_year_rmses = Float64[]
    location_year_r2s = Float64[]
    
    # Read results for each epsilon value
    for epsilon in epsilon_values
        # Read location results
        location_df = CSV.read("results/figures/MLE/epsilon_$(epsilon)_location_predictions.csv", DataFrame)
        push!(location_maes, location_df.mae)
        push!(location_rmses, location_df.rmse)
        push!(location_r2s, location_df.r2)
        
        # Read location-year results
        location_year_df = CSV.read("results/figures/MLE/epsilon_$(epsilon)_location_year_predictions.csv", DataFrame)
        push!(location_year_maes, location_year_df.mae[1])
        push!(location_year_rmses, location_year_df.rmse[1])
        push!(location_year_r2s, location_year_df.r2[1])
    end
    
    # Create plots
    p1 = plot(epsilon_values, [location_maes, location_year_maes],
             xscale=:log10,
             xlabel="Epsilon",
             ylabel="MAE",
             label=["By Location" "By Location-Year"],
             title="MAE vs Epsilon",
             legend=:topleft)
    
    p2 = plot(epsilon_values, [location_rmses, location_year_rmses],
             xscale=:log10,
             xlabel="Epsilon",
             ylabel="RMSE",
             label=["By Location" "By Location-Year"],
             title="RMSE vs Epsilon",
             legend=:topleft)
    
    p3 = plot(epsilon_values, [location_r2s, location_year_r2s],
             xscale=:log10,
             xlabel="Epsilon",
             ylabel="R²",
             label=["By Location" "By Location-Year"],
             title="R² vs Epsilon",
             legend=:topleft)
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
    
    # Save plot
    mkpath("results/figures/MLE")
    savefig(p, "results/figures/MLE/epsilon_sensitivity_analysis.png")
    
    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end