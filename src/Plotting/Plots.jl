_normalize_time_plot_legends(selection) =
    selection === nothing ? nothing :
    selection isa AbstractSet{Symbol} ? selection :
    Set(Symbol.(selection))

_time_plot_legend_enabled(selection, key::Symbol) = selection === nothing ? true : key in selection

# ----------------------------
# Plot Plos One Plots
# ----------------------------
function plot_plos_one_plots(parallel_data, config; policies_to_plot=nothing, time_plot_legends=nothing)
    @info "Plotting Plos One Plots"
    legend_selection = _normalize_time_plot_legends(time_plot_legends)
    plos_one_plot_kalman_filter_belief_trajectory(parallel_data, "NUS_SARSOP_Policy", config, 0.6)
    plos_one_sealice_levels_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :sealice),
    )
    plos_one_reward_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :reward),
    )
    plos_one_biomass_loss_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :biomass),
    )
    plos_one_regulatory_penalty_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :regulatory),
    )
    plos_one_fish_disease_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :fish_disease),
    )
    plos_one_treatment_cost_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :treatment_cost),
    )
    plos_one_combined_treatment_probability_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :treatment_probability),
    )
    plos_one_sarsop_dominant_action(parallel_data, config, 0.6)
    plos_one_algo_sealice_levels_over_time(config, "NUS_SARSOP_Policy", 0.6)
    plos_one_episode_sealice_levels_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :episode_sealice),
    )
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