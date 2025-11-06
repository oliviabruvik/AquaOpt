# ----------------------------
# Plot Plos One Plots
# ----------------------------
function plot_plos_one_plots(parallel_data, config)
    @info "Plotting Plos One Plots"
    plos_one_plot_kalman_filter_belief_trajectory(parallel_data, "NUS_SARSOP_Policy", config, 0.6)
    plos_one_sealice_levels_over_time(parallel_data, config)
    plos_one_combined_treatment_probability_over_time(parallel_data, config)
    plos_one_sarsop_dominant_action(parallel_data, config, 0.6)
    @info "Saved all plos one plots to $(config.figures_dir)"
end

# ----------------------------
# Plot Parallel Plots
# ----------------------------
function plot_parallel_plots(parallel_data, config)

    # Print treatment frequency
    print_treatment_frequency(parallel_data, config)

    # Print histories
    print_histories(parallel_data, config)

    # Plot heuristic vs sarsop sealice levels over time
    plot_heuristic_vs_sarsop_sealice_levels_over_time_latex(parallel_data, config)

    # Plot one simulation with all state variables over time
    plot_one_simulation_with_all_state_variables_over_time(parallel_data, config, "NUS_SARSOP_Policy")
    plot_one_simulation_with_all_state_variables_over_time(parallel_data, config, "Heuristic_Policy")

    # Plot treatment distribution comparison
    plot_treatment_distribution_comparison_latex(parallel_data, config)
    
    # Plot SARSOP policy action heatmap
    plot_sarsop_policy_action_heatmap(config, 0.6)
    
    # Plot Heuristic policy action heatmap
    plot_heuristic_policy_action_heatmap(config, 0.6)
    
    # Plot combined policy action heatmaps side by side
    plot_combined_policy_action_heatmaps(config, 0.6)

    plot_beliefs_over_time(parallel_data, "NUS_SARSOP_Policy", config, 0.6)
    plot_beliefs_over_time(parallel_data, "Heuristic_Policy", config, 0.6)

    # Plot combined treatment probability over time
    plot_combined_treatment_probability_over_time(parallel_data, config)
end

# ----------------------------
# Plot results
# ----------------------------
function plot_results(algorithms, config)

    @info "Plotting results"
    for algo in algorithms

        # Load histories
        histories_dir = joinpath(config.simulations_dir, "$(algo.solver_name)")
        histories_path = joinpath(histories_dir, "$(algo.solver_name)_histories.jld2")
        if !isfile(histories_path)
            @error "Histories file not found at $histories_path. Run simulation first. with --simulate"
            continue
        end
        @load histories_path histories

        # Load data from parallel simulations
        data_path = joinpath(config.simulations_dir, "all_policies_simulation_data.jld2")
        if !isfile(data_path)
            @error "Data file not found at $data_path. Run simulation first. with --simulate"
            continue
        end
        @load data_path data

        # Load avg results
        avg_results_path = joinpath(config.results_dir, "$(algo.solver_name)_avg_results.jld2")
        if !isfile(avg_results_path)
            @error "Avg results file not found at $avg_results_path. Run simulation first. with --simulate"
            continue
        end
        @load avg_results_path avg_results

        # Plot belief means and variances
        plot_beliefs_over_time(data, algo.solver_name, config, 0.6)

        # Plot policy cost vs sealice
        plot_policy_cost_vs_sealice(histories, avg_results, algo.solver_name, config)

        # Plot policy belief levels
        plot_policy_belief_levels(histories, algo.solver_name, config, 0.6)

        # Plot treatment heatmap
        plot_treatment_heatmap(algo, config)

        # Plot simulation treatment heatmap
        # plot_simulation_treatment_heatmap(algo, config; use_observations=false, n_bins=50)

        plot_algo_sealice_levels_over_time(config, algo.solver_name, 0.6)
        plot_algo_adult_predicted_over_time(config, algo.solver_name, 0.6)

    end
    
    # Plot comparison plots
    plot_all_cost_vs_sealice(config)
    plot_policy_sealice_levels_over_lambdas(config)
    plot_policy_treatment_cost_over_lambdas(config)
    plot_policy_sealice_levels_over_time(config, 0.6)
    plot_policy_treatment_cost_over_time(config, 0.6)
    plot_policy_reward_over_lambdas(config)

    @info "Saved all plots to $(config.figures_dir)"
end