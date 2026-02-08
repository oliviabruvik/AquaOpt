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

    plot_flag = false

    # Option 1: Balanced [treatment, regulatory, biomass, health, sea lice]
    reward_lambdas1 = [0.46, 0.12, 0.12, 0.18, 0.12]
    AquaOpt.main(log_space=true, experiment_name="log_space_ukf", mode=mode, location="north", ekf_filter=false, plot=plot_flag, reward_lambdas=reward_lambdas1, sim_reward_lambdas=reward_lambdas1)
    AquaOpt.main(log_space=true, experiment_name="log_space_ukf", mode=mode, location="west", ekf_filter=false, plot=plot_flag, reward_lambdas=reward_lambdas1, sim_reward_lambdas=reward_lambdas1)
    AquaOpt.main(log_space=true, experiment_name="log_space_ukf", mode=mode, location="south", ekf_filter=false, plot=plot_flag, reward_lambdas=reward_lambdas1, sim_reward_lambdas=reward_lambdas1)

    # Option 2: Cost-focused (prioritize economics over welfare)
    reward_lambdas2 = [0.55, 0.10, 0.20, 0.05, 0.10]
    AquaOpt.main(log_space=true, experiment_name="log_space_ukf", mode=mode, location="south", ekf_filter=false, plot=plot_flag, reward_lambdas=reward_lambdas2, sim_reward_lambdas=reward_lambdas2)

    # Option 3: Welfare-focused (prioritize fish health and avoid over-treatment)
    reward_lambdas3 = [0.15, 0.05, 0.10, 0.35, 0.35]
    AquaOpt.main(log_space=true, experiment_name="log_space_ukf", mode=mode, location="south", ekf_filter=false, plot=plot_flag, reward_lambdas=reward_lambdas3, sim_reward_lambdas=reward_lambdas3)

    # Raw space and EKF
    AquaOpt.main(log_space=false, experiment_name="raw_space_ukf", mode=mode, location="north", ekf_filter=false, plot=plot_flag, reward_lambdas=reward_lambdas1, sim_reward_lambdas=reward_lambdas1)
    AquaOpt.main(log_space=true, experiment_name="log_space_ekf", mode=mode, location="north", ekf_filter=true, plot=plot_flag, reward_lambdas=reward_lambdas1, sim_reward_lambdas=reward_lambdas1)

    return
end

# CLI entry point
if length(ARGS) < 1
    println("Usage: julia --project scripts/run_experiments.jl <mode>")
    println("  mode: 'debug' or 'paper'")
    exit(1)
end

run_experiments(ARGS[1])
