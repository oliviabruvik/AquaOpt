#!/usr/bin/env julia

#=
run_experiments.jl

Runs the full suite of experiments across locations, reward weights, and filter types.

Usage:
    julia --project scripts/run_experiments.jl <mode>
    julia --project scripts/run_experiments.jl debug
    julia --project scripts/run_experiments.jl paper
=#

using AquaOpt

function run_experiments(mode)

    plot_flag = true

    # Reward lambda scenarios [treatment, regulatory, biomass, health, sea lice]
    lambda_scenarios = Dict(
        "balanced" => [0.46, 0.12, 0.12, 0.18, 0.12],
        "cost"     => [0.55, 0.10, 0.20, 0.05, 0.10],
        "welfare"  => [0.15, 0.05, 0.10, 0.35, 0.35],
    )

    locations = ["north", "west", "south"]

    # Full 3x3 factorial: 3 locations × 3 lambda scenarios
    for location in locations
        for (scenario_name, lambdas) in lambda_scenarios
            @info "Running: location=$location, scenario=$scenario_name"
            AquaOpt.main(
                log_space=true,
                experiment_name="log_space_ukf_$(scenario_name)",
                mode=mode,
                location=location,
                ekf_filter=false,
                plot=plot_flag,
                reward_lambdas=lambdas,
                sim_reward_lambdas=lambdas,
            )
        end
    end

    # Ablations: Raw space and EKF (balanced lambdas, north only)
    balanced = lambda_scenarios["balanced"]
    AquaOpt.main(log_space=false, experiment_name="raw_space_ukf", mode=mode, location="north", ekf_filter=false, plot=plot_flag, reward_lambdas=balanced, sim_reward_lambdas=balanced)
    AquaOpt.main(log_space=true, experiment_name="log_space_ekf", mode=mode, location="north", ekf_filter=true, plot=plot_flag, reward_lambdas=balanced, sim_reward_lambdas=balanced)

    return
end

# CLI entry point
if length(ARGS) < 1
    println("Usage: julia --project scripts/run_experiments.jl <mode>")
    println("  mode: 'debug' or 'paper'")
    exit(1)
end

run_experiments(ARGS[1])
