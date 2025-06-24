using Plots
using JLD2
plotlyjs()  # Set the backend to PlotlyJS

# ----------------------------
# Plot 1: Time series by location
# ----------------------------
function plot_sealice_levels_over_time(df)
    p = plot(
        df.total_week, 
        df.adult_sealice, 
        title="Sealice levels over time", 
        xlabel="Total weeks from start (2012)", 
        ylabel="Sealice levels",
        color=df.location_number
    )
    savefig(p, "results/figures/sealice_levels_over_time.png")
    return p
end

# ----------------------------
# Plot 2: Cost vs Sea Lice for one policy
# ----------------------------
function plot_policy_cost_vs_sealice(results, title, config, pomdp_config)
    p = scatter(
        results.avg_treatment_cost,
        results.avg_sealice,
        marker_z=results.lambda,
        title="Treatment Cost vs Sea Lice Levels ($title)",
        xlabel="Average Treatment Cost (MNOK / year)", 
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        colorbar_title="λ",
        marker=:circle,
        markersize=6,
        legend=false,
        grid=true,
        framestyle=:box,
        c=:viridis, # Add colormap
        colorbar=true # Explicitly show colorbar
    )

    mkpath(joinpath(config.figures_dir, "treatment_cost_vs_sealice_plots", title))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_vs_sealice_plots/$(title)/results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 3: Overlay all policies
# ----------------------------
function plot_all_cost_vs_sealice(config, pomdp_config)
    # Create a new plot
    p = plot(
        title="Treatment Cost vs Sea Lice Levels (All Policies)",
        xlabel="Average Treatment Cost (MNOK / year)", 
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:topleft,
        grid=true,
        framestyle=:box
    )
    
    # Define colors for each policy
    policy_colors = Dict(
        "Heuristic_Policy" => :blue,
        "VI_Policy" => :red,
        "SARSOP_Policy" => :green,
        "QMDP_Policy" => :purple
    )
    
    # Load and plot each policy's results
    for (policy_name, color) in policy_colors
        try
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Sort the data by treatment cost
            sort_indices = sortperm(results.avg_treatment_cost)
            sorted_costs = results.avg_treatment_cost[sort_indices]
            sorted_sealice = results.avg_sealice[sort_indices]
            
            # Add the line plot to the main plot
            plot!(
                p,
                sorted_costs,
                sorted_sealice,
                linewidth=2,
                color=color,
                label=policy_name
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    
    # Save the figure
    mkpath(joinpath(config.figures_dir, "treatment_cost_vs_sealice_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_vs_sealice_plots/All_policies/results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 4: Time-series of sea lice for each policy
# ----------------------------
function plot_policy_sealice_levels(config, pomdp_config)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Average Sea Lice Levels Over Time",
        xlabel="Time Step",
        ylabel="Average Sea Lice Levels",
        legend=:topleft,
        grid=true
    )
    
    # Define colors and markers for each policy
    policy_styles = Dict(
        "Heuristic_Policy" => (color=:blue, marker=:circle),
        "VI_Policy" => (color=:red, marker=:square),
        "SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle)
    )
    
    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Add the scatter plot to the main plot
            scatter!(
                p,
                1:length(results.avg_sealice),
                results.avg_sealice,
                label=policy_name,
                color=style.color,
                marker=style.marker,
                alpha=0.7
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "sealice_time_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "sealice_time_plots/All_policies/sealice_time_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 5: Time-series of belief for each policy
# NOTE: RESULTS ARE IN LOG SPACE
# ----------------------------
function plot_policy_belief_levels(results, title, config, pomdp_config, lambda; show_actual_states=true)
    # Initialize the plot with the y-axis range starting at 0 always
    p = plot(
        title="Belief State Evolution ($title)",
        xlabel="Time Step (Weeks)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:topleft,
        grid=true,
        ylims=(0, 15)
    )

    # Get values for first episode of lambda
    lambda_index = findfirst(isequal(lambda), results.lambda)

    @assert results.lambda[lambda_index] == lambda "Lambda index not found"

    # Get values for first episode of lambda
    first_episode_belief_hist = results.belief_hists[lambda_index][1]
    first_episode_state_hist = results.state_hists[lambda_index][1]
    first_episode_action_hist = results.action_hists[lambda_index][1]

    # Convert Gaussian belief to mean and variance
    if pomdp_config.log_space
        belief_means = [exp(belief.μ[1]) for belief in first_episode_belief_hist]
        belief_stds = [exp(sqrt(belief.Σ[1,1])) for belief in first_episode_belief_hist]
        actual_states = [exp(s.SeaLiceLevel) for s in first_episode_state_hist]
    else
        belief_means = [belief.μ[1] for belief in first_episode_belief_hist]
        belief_stds = [sqrt(belief.Σ[1,1]) for belief in first_episode_belief_hist]
        actual_states = [s.SeaLiceLevel for s in first_episode_state_hist]
    end
    
    # Add the scatter plot to the main plot with variance as error bars
    scatter!(
        p,
        1:length(belief_means),
        belief_means,
        label="Belief",
        color=:blue,
        linewidth=2,
        errorbar=true,
        yerror=belief_stds
    )
    
    # Plot actual states if requested
    if show_actual_states
        scatter!(
            p,
            1:length(actual_states),
            actual_states,
            label="Actual",
            color=:red,
            marker=:circle,
            alpha=0.7
        )
    end

    # Add treatment annotations
    actions = [first_episode_action_hist[i] == Treatment ? "T" : "N" for i in 1:length(first_episode_action_hist)]
    for (t, a) in zip(1:length(actions), actions)
        annotate!(p, t, 0.4, text(a, 8, :black))
    end

    # Save to both directories for backward compatibility
    mkpath(joinpath(config.figures_dir, "belief_plots", title))
    mkpath(joinpath(config.figures_dir, "research_plots"))
    
    # Save to belief_plots for individual policy analysis
    savefig(p, joinpath(config.figures_dir, "belief_plots/$(title)/beliefs_$(lambda)_lambda_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    
    # Save to research_plots with actual states for research paper
    if show_actual_states
        savefig(p, joinpath(config.figures_dir, "research_plots", "belief_evolution_$(title)_$(lambda)_lambda_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    end
    
    return p
end

# ----------------------------
# Plot 5: Time-series of belief for each policy
# ----------------------------
function plot_all_policy_belief_levels(results, title, config, pomdp_config)
    # Initialize the plot with the y-axis range starting at 0 always
    p = plot(
        title="Belief Levels Over Time",
        xlabel="Time Step (Weeks)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:topleft,
        grid=true,
        ylims=(0, 15)
    )

    # Get values for first episode of first lambda
    lambda = results.lambda[1]
    first_episode_belief_hist = results.belief_hists[1][1]
    first_episode_action_hist = results.action_hists[1][1]

    # Convert Gaussian belief to mean and variance
    belief_means = [belief.μ[1] for belief in first_episode_belief_hist]
    belief_vars = [belief.Σ[1,1] for belief in first_episode_belief_hist]
    actions = [first_episode_action_hist[i] == Treatment ? "T" : "N" for i in 1:length(first_episode_action_hist)]

    # Add the scatter plot to the main plot with variance as error bars
    scatter!(
        p,
        1:length(belief_means),
        belief_means,
        label=title,
        color=:blue,
        marker=:circle,
        alpha=0.7,
        yerror=belief_vars
    )

    for (t, a) in zip(1:length(actions), actions)
        annotate!(p, t, 0.4, text(a, 8, :black))
    end

    mkpath(joinpath(config.figures_dir, "belief_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "belief_plots/All_policies/policy_belief_comparison_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Research Paper Visualizations
# ----------------------------

# Pareto Frontier Plot
function plot_pareto_frontier(config, pomdp_config)
    # Load all results
    results_file_path = joinpath(config.data_dir, "avg_results", "All_policies", "all_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2")
    if !isfile(results_file_path)
        @warn "Results file not found at $results_file_path"
        return nothing
    end
    @load results_file_path all_results

    p = plot(
        title="Pareto Frontier: Treatment Cost vs Sea Lice Levels",
        xlabel="Average Treatment Cost (MNOK / year)",
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:bottomleft,
        grid=true,
        framestyle=:box
    )

    # Define colors and markers for each policy
    policy_styles = Dict(
        "Heuristic_Policy" => (color=:blue, marker=:circle),
        "VI_Policy" => (color=:red, marker=:square),
        "SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle)
    )

    # Plot each policy's results
    for (policy_name, style) in policy_styles
        if haskey(all_results, policy_name)
            results = all_results[policy_name]
            scatter!(
                p,
                results.avg_treatment_cost,
                results.avg_sealice,
                label=policy_name,
                color=style.color,
                marker=style.marker,
                alpha=0.7
            )
        end
    end

    # Save plot
    mkpath(joinpath(config.figures_dir, "research_plots"))
    savefig(p, joinpath(config.figures_dir, "research_plots", "pareto_frontier_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# Policy Value Function Visualization
function plot_value_function(algorithm, config, pomdp_config, λ)

    # Load policy
    policies_dir = joinpath(config.data_dir, "policies", algorithm.solver_name)
    policy_file_path = joinpath(policies_dir, "policy_pomdp_mdp_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2")
    if !isfile(policy_file_path)
        @warn "Policy file not found at $policy_file_path"
        return nothing
    end
    @load policy_file_path policy pomdp mdp

    # Create state space grid
    if pomdp_config.log_space
        pomdp_states = [SeaLiceLogState(x) for x in pomdp.log_sea_lice_range]
        x_vals = [s.SeaLiceLevel for s in pomdp_states]
    else
        pomdp_states = [SeaLiceState(x) for x in pomdp.sea_lice_range]
        x_vals = [s.SeaLiceLevel for s in pomdp_states]
    end

    # Calculate values based on policy type
    values = if algorithm.solver_name == "Heuristic_Policy" || algorithm.solver_name == "VI_Policy" || algorithm.solver_name == "QMDP_Policy"
        # For MDP policies, we can evaluate directly on states
        [value(policy, s) for s in pomdp_states]
    else
        # For POMDP policies (like SARSOP), we need to create belief vectors
        state_space = states(pomdp)
        n_states = length(state_space)
        
        # Create a function to get the index of a state in the state space
        function get_state_index(s)
            for (i, sp) in enumerate(state_space)
                if s.SeaLiceLevel == sp.SeaLiceLevel
                    return i
                end
            end
            return 1  # Default to first state if not found
        end
        
        # Create belief vectors for each state
        [begin
            idx = get_state_index(s)
            b = zeros(n_states)
            b[idx] = 1.0
            value(policy, b)
        end for s in pomdp_states]
    end

    p = plot(
        x_vals,
        values,
        title="Policy Value Function",
        xlabel="Sea Lice Level",
        ylabel="Value",
        label="Value",
        linewidth=2,
        grid=true,
        framestyle=:box
    )

    # Save plot
    plot_dir = joinpath(config.figures_dir, "research_plots", "value_functions", algorithm.solver_name)
    mkpath(plot_dir)
    savefig(p, joinpath(plot_dir, "value_function_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# Sensitivity Analysis Plot
function plot_sensitivity_analysis(config, pomdp_config, param_name, param_values)
    p = plot(
        title="Sensitivity Analysis: $param_name",
        xlabel=param_name,
        ylabel="Average Sea Lice Level",
        legend=:topleft,
        grid=true,
        framestyle=:box
    )

    # Define colors for each policy
    policy_colors = Dict(
        "Heuristic_Policy" => :blue,
        "VI_Policy" => :red,
        "SARSOP_Policy" => :green,
        "QMDP_Policy" => :purple
    )

    # Plot results for each policy
    for (policy_name, color) in policy_colors
        results = []
        for param_value in param_values
            # Load results for this parameter value
            results_file = joinpath(config.data_dir, "sensitivity", "$(policy_name)_$(param_name)_$(param_value).jld2")
            if isfile(results_file)
                @load results_file avg_sealice
                push!(results, avg_sealice)
            end
        end
        
        if !isempty(results)
            plot!(
                p,
                param_values,
                results,
                label=policy_name,
                color=color,
                marker=:circle,
                linewidth=2
            )
        end
    end

    mkpath(joinpath(config.figures_dir, "research_plots"))
    savefig(p, joinpath(config.figures_dir, "research_plots", "sensitivity_$(param_name)_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# Treatment Decision Heatmap
function plot_treatment_heatmap(algorithm, config, pomdp_config, λ)

    # Load policy, pomdp, and mdp
    policies_dir = joinpath(config.data_dir, "policies", algorithm.solver_name)
    policy_file_path = joinpath(policies_dir, "policy_pomdp_mdp_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2")
    if !isfile(policy_file_path)
        @warn "Policy file not found at $policy_file_path"
        return nothing
    end
    @load policy_file_path policy pomdp mdp

    # Create state space grid
    if pomdp_config.log_space
        states = [SeaLiceLogState(x) for x in pomdp.log_sea_lice_range]
        decisions = [action(policy, s) == Treatment ? 1 : 0 for s in states]
        x_vals = [s.SeaLiceLevel for s in states]
    else
        states = [SeaLiceState(x) for x in pomdp.sea_lice_range]
        decisions = [action(policy, s) == Treatment ? 1 : 0 for s in states]
        x_vals = [s.SeaLiceLevel for s in states]
    end

    p = heatmap(
        reshape(decisions, 1, :),
        title="Treatment Decision Heatmap",
        xlabel="Sea Lice Level",
        ylabel="",
        yticks=[],
        colorbar_title="Treatment Decision",
        c=:viridis
    )

    # Save plot
    plot_dir = joinpath(config.figures_dir, "research_plots", "treatment_heatmaps", algorithm.solver_name)
    mkpath(plot_dir)
    savefig(p, joinpath(plot_dir, "treatment_heatmap_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end