_normalize_time_plot_legends(selection) =
    selection === nothing ? nothing :
    selection isa AbstractSet{Symbol} ? selection :
    Set(Symbol.(selection))

_time_plot_legend_enabled(selection, key::Symbol) = selection === nothing ? true : key in selection

# ----------------------------
# Plot Plos One Plots
# ----------------------------
function plot_plos_one_plots(parallel_data, config, algorithms; policies_to_plot=nothing, time_plot_legends=nothing)
    @info "Plotting Plos One Plots"
    legend_selection = _normalize_time_plot_legends(time_plot_legends)
    algo_names = Set(a.solver_name for a in algorithms)
    # Default policies_to_plot to the algorithms list
    if policies_to_plot === nothing
        policies_to_plot = algo_names
    end

    # Determine which SARSOP variant is available (if any)
    sarsop_name = if "NUS_SARSOP_Policy" in algo_names
        "NUS_SARSOP_Policy"
    elseif "Native_SARSOP_Policy" in algo_names
        "Native_SARSOP_Policy"
    else
        nothing
    end

    if sarsop_name !== nothing
        plos_one_plot_kalman_filter_belief_trajectory(parallel_data, sarsop_name, config)
        plos_one_algo_sealice_levels_over_time(parallel_data, config, sarsop_name)
        plos_one_sarsop_dominant_action(parallel_data, config, sarsop_name)
    end

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
    plos_one_episode_sealice_levels_over_time(
        parallel_data,
        config;
        policies_to_plot=policies_to_plot,
        show_legend=_time_plot_legend_enabled(legend_selection, :episode_sealice),
    )
    @info "Saved all plos one plots to $(config.figures_dir)"
end
