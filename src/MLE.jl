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

plotlyjs()  # Activate Plotly backend

LICE_PATH = "data/processed/sealice_data.csv"

function main()
    @info "Loading cleaned data"
    df = CSV.read(LICE_PATH, DataFrame)
    #println(df)
    println(names(df))
    #println(unique(df.location_number))

    # Split into separate dataframes for each location
    locations = unique(df.location_number)

    growth_rates = []
    for location in locations
        location_df = df[df.location_number .== location, :]
        growth_rate_df = filter(row -> row.mechanical_removal == false, location_df)
        growth_rate_df = filter(row -> row.adult_sealice > 0, growth_rate_df)
        result = mle_analysis(growth_rate_df)
        println("MLE for growth rate at location $location: $result")
        push!(growth_rates, result)
    end

    println(growth_rates)
    println(mean(growth_rates))

    @info "Running MLE analysis"
    #println(df.adult_sealice)
    # growth_rate_df = filter(row -> row.mechanical_removal == false, df)
    # result = mle_analysis(growth_rate_df)
    # println("MLE for growth rate: $result")
end

########################################################
# MLE Analysis for Sea Lice Growth Rate
########################################################

# Define log likelihood function for MLE analysis
function log_likelihood(r, y)
    log_likelihood = 0.0
    for t in 2:length(y)
        μ = log(y[t-1]) + r[1]
        log_likelihood += logpdf(Normal(μ, 0.1), log(y[t]))
    end
    return -log_likelihood
end

# Run minimization to find the MLE for the growth rate r
function mle_analysis(df)
    result = optimize(r -> log_likelihood(r, df.adult_sealice), [0.0])
    r_hat = Optim.minimizer(result)
    return r_hat
end





function bayesian_analysis_sealice_disease()

    # Load data
    df_inner = CSV.read("data/processed/bayesian_inner_data.csv", DataFrame)
    df_outer = CSV.read("data/processed/bayesian_outer_data.csv", DataFrame)

    # Handle missing values and convert treatment column to binary
    df_inner.treatment = ifelse.(df_inner.treatment .== "None", false, true)
    df_outer.treatment = ifelse.(df_outer.treatment .== "None", false, true)

    # Drop rows where lice count is not available or relevant for analyses
    lice_disease_data = dropmissing(df_inner, [:adult_sealice, :disease])
    lice_treatment_data = dropmissing(df_inner, [:adult_sealice, :treatment])
    disease_treatment_data = dropmissing(df_inner, [:disease, :treatment])

    # Set up plot layout
    p = plot(layout = (1, 3), size = (1800, 500))

    # Relationship between lice and disease
    @df lice_disease_data boxplot!(p[1], :disease, :adult_sealice, 
        xlabel="Disease Status", ylabel="Adult Sea Lice", 
        xticks=([0, 1], ["No Disease", "Disease"]), 
        title="Lice count vs Disease")

    # Relationship between lice and treatment
    @df lice_treatment_data boxplot!(p[2], :treatment, :adult_sealice, 
        xlabel="Treatment Status", ylabel="Adult Sea Lice", 
        xticks=([0, 1], ["No Treatment", "Treatment"]), 
        title="Lice count vs Treatment")

    # Count occurrences for disease-treatment relationship
    counts = combine(groupby(disease_treatment_data, [:treatment, :disease]), nrow => :count)
    counts.treatment = string.(counts.treatment)
    counts.disease = string.(counts.disease)

    # Ensure the directory exists
    mkpath("results/figures/bayesian_analysis")

    # Save plot
    savefig(p, "results/figures/bayesian_analysis/lice_disease_treatment_analysis.png")

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end