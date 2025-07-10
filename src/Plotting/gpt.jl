


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




# if algo.solver_name != "Heuristic_Policy" && algo.solver_name != "SARSOP_Policy"

        #     # Plot value function
        #     # plot_value_function(algo, CONFIG, POMDP_CONFIG, 0.5)

        #     # TODO: fix this function for SARSOP
        #     # Plot treatment heatmap
        #     plot_treatment_heatmap(algo, CONFIG, POMDP_CONFIG, 0.5)
        # end



        

mkpath(joinpath(CONFIG.figures_dir, "research_plots"))

    # Generate Pareto frontier
    plot_pareto_frontier(CONFIG, POMDP_CONFIG)

    # Run sensitivity analysis for key parameters
    param_values = [0.5, 0.7, 0.9]  # Example values for treatment effectiveness
    # plot_sensitivity_analysis(config, pomdp_config, "treatment_effectiveness", param_values)

    # Generate treatment decision heatmaps
    # for (policy_name, policy) in policies
    #     plot_treatment_heatmap(policy, pomdp, config, pomdp_config)
    # end



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
        "QMDP_Policy" => (color=:purple, marker=:dtriangle),
        "Random_Policy" => (color=:orange, marker=:rect)
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
        "QMDP_Policy" => :purple,
        "Random_Policy" => :orange
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

# ----------------------------
# Plot: Lambda vs average reward for each policy
# ----------------------------
function plot_policy_reward_over_lambdas(config, pomdp_config)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Average Reward Over Lambda",
        xlabel="Lambda (λ)",
        ylabel="Average Reward",
        legend=:topleft,
        grid=true
    )
    
    # Define colors and markers for each policy
    policy_styles = Dict(
        "Heuristic_Policy" => (color=:blue, marker=:circle),
        "VI_Policy" => (color=:red, marker=:square),
        "SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle),
        "Random_Policy" => (color=:orange, marker=:rect)
    )
    
    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try
            # Load the simulation histories from the JLD2 file
            histories_dir = joinpath(config.data_dir, "simulation_histories", policy_name)
            lambda_values = config.lambda_values
            mean_rewards = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for λ in lambda_values
                try
                    # Load simulation histories for this lambda
                    history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
                    history_file_path = joinpath(histories_dir, history_filename)
                    
                    if !isfile(history_file_path)
                        @warn "History file not found at $history_file_path for λ=$λ"
                        push!(mean_rewards, NaN)
                        push!(ci_lower, NaN)
                        push!(ci_upper, NaN)
                        continue
                    end
                    
                    @load history_file_path histories
                    
                    # Extract total rewards for all episodes
                    r_total_hists = histories["r_total_hists"]
                    
                    # Calculate mean and 95% CI
                    mean_reward = mean(r_total_hists)
                    std_reward = std(r_total_hists)
                    n_episodes = length(r_total_hists)
                    se_reward = std_reward / sqrt(n_episodes)  # Standard error
                    ci_margin = 1.96 * se_reward  # 95% CI margin
                    
                    push!(mean_rewards, mean_reward)
                    push!(ci_lower, mean_reward - ci_margin)
                    push!(ci_upper, mean_reward + ci_margin)
                    
                catch e
                    @warn "Failed to process λ=$λ for $policy_name: $e"
                    push!(mean_rewards, NaN)
                    push!(ci_lower, NaN)
                    push!(ci_upper, NaN)
                end
            end
            
            # Remove NaN values
            valid_indices = .!isnan.(mean_rewards)
            valid_lambdas = lambda_values[valid_indices]
            valid_mean = mean_rewards[valid_indices]
            valid_ci_lower = ci_lower[valid_indices]
            valid_ci_upper = ci_upper[valid_indices]
            
            if !isempty(valid_mean)
                # Add the line plot with 95% confidence interval ribbon
                plot!(
                    p,
                    valid_lambdas,
                    valid_mean,
                    ribbon=(valid_mean .- valid_ci_lower, valid_ci_upper .- valid_mean),
                    label=policy_name,
                    color=style.color,
                    linewidth=2,
                    fillalpha=0.3,
                    alpha=0.7
                )
            end
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "reward_lambda_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "reward_lambda_plots/All_policies/reward_lambda_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Convergence Plots
# ----------------------------

# Value Iteration Convergence Plot
function plot_value_iteration_convergence(algorithm, config, pomdp_config, λ)
    if algorithm.solver_name != "VI_Policy"
        @warn "This function is only for Value Iteration policies"
        return nothing
    end
    
    # Generate policy with detailed solver tracking
    if pomdp_config.log_space
        pomdp = SeaLiceLogMDP(lambda=λ, costOfTreatment=pomdp_config.costOfTreatment, growthRate=pomdp_config.growthRate, rho=pomdp_config.rho, discount_factor=pomdp_config.discount_factor)
    else
        pomdp = SeaLiceMDP(lambda=λ, costOfTreatment=pomdp_config.costOfTreatment, growthRate=pomdp_config.growthRate, rho=pomdp_config.rho, discount_factor=pomdp_config.discount_factor)
    end
    mdp = UnderlyingMDP(pomdp)
    
    # Create solver with tracking
    solver = ValueIterationSolver(max_iterations=1000, belres=1e-6, verbose=true)
    
    # Solve and capture convergence data
    policy = solve(solver, mdp)
    
    # Extract convergence information (this depends on the solver implementation)
    # You might need to modify the solver to track residuals
    iterations = 1:length(policy.util)  # Assuming util contains value function history
    
    p = plot(
        title="Value Iteration Convergence (λ = $λ)",
        xlabel="Iteration",
        ylabel="Bellman Residual",
        yscale=:log10,
        grid=true,
        linewidth=2
    )
    
    # This is a placeholder - you'd need to modify the solver to track residuals
    # plot!(p, iterations, residuals, label="Bellman Residual", color=:blue)
    
    mkpath(joinpath(config.figures_dir, "convergence_plots", "value_iteration"))
    savefig(p, joinpath(config.figures_dir, "convergence_plots/value_iteration/vi_convergence_lambda_$(λ)_$(pomdp_config.log_space)_log_space.png"))
    return p
end

# Simulation Performance Convergence Plot
function plot_simulation_convergence(policy_name, config, pomdp_config, λ; window_size=5)
    try
        # Load simulation histories for this lambda
        histories_dir = joinpath(config.data_dir, "simulation_histories", policy_name)
        history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
        history_file_path = joinpath(histories_dir, history_filename)
        
        if !isfile(history_file_path)
            @warn "History file not found at $history_file_path"
            return nothing
        end
        
        @load history_file_path histories
        
        # Extract episode rewards
        r_total_hists = histories["r_total_hists"]
        episodes = 1:length(r_total_hists)
        
        # Calculate moving average
        moving_avg = Float64[]
        cumulative_avg = Float64[]
        
        for i in episodes
            # Moving average over window
            start_idx = max(1, i - window_size + 1)
            window_rewards = r_total_hists[start_idx:i]
            push!(moving_avg, mean(window_rewards))
            
            # Cumulative average
            push!(cumulative_avg, mean(r_total_hists[1:i]))
        end
        
        p = plot(
            title="Simulation Convergence: $policy_name (λ = $λ)",
            xlabel="Episode",
            ylabel="Total Reward",
            grid=true,
            legend=:bottomright
        )
        
        # Plot individual episode rewards
        scatter!(p, episodes, r_total_hists, 
                label="Episode Rewards", alpha=0.6, color=:lightblue, markersize=3)
        
        # Plot moving average
        plot!(p, episodes, moving_avg, 
              label="Moving Average (window=$window_size)", linewidth=2, color=:blue)
        
        # Plot cumulative average
        plot!(p, episodes, cumulative_avg, 
              label="Cumulative Average", linewidth=2, color=:red)
        
        mkpath(joinpath(config.figures_dir, "convergence_plots", "simulation", policy_name))
        savefig(p, joinpath(config.figures_dir, "convergence_plots/simulation/$(policy_name)/sim_convergence_lambda_$(λ)_$(pomdp_config.log_space)_log_space.png"))
        return p
        
    catch e
        @warn "Could not create convergence plot for $policy_name: $e"
        return nothing
    end
end

# Multi-Policy Convergence Comparison
function plot_multi_policy_convergence(config, pomdp_config, λ; window_size=5)
    p = plot(
        title="Policy Convergence Comparison (λ = $λ)",
        xlabel="Episode",
        ylabel="Cumulative Average Reward",
        grid=true,
        legend=:bottomright
    )
    
    # Define colors for each policy
    policy_styles = Dict(
        "Heuristic_Policy" => (color=:blue, linestyle=:solid),
        "VI_Policy" => (color=:red, linestyle=:solid),
        "SARSOP_Policy" => (color=:green, linestyle=:solid),
        "QMDP_Policy" => (color=:purple, linestyle=:solid),
        "Random_Policy" => (color=:orange, linestyle=:dash)
    )
    
    for (policy_name, style) in policy_styles
        try
            # Load simulation histories for this lambda
            histories_dir = joinpath(config.data_dir, "simulation_histories", policy_name)
            history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
            history_file_path = joinpath(histories_dir, history_filename)
            
            if !isfile(history_file_path)
                @warn "History file not found for $policy_name"
                continue
            end
            
            @load history_file_path histories
            
            # Extract episode rewards
            r_total_hists = histories["r_total_hists"]
            episodes = 1:length(r_total_hists)
            
            # Calculate cumulative average
            cumulative_avg = [mean(r_total_hists[1:i]) for i in episodes]
            
            # Plot cumulative average
            plot!(p, episodes, cumulative_avg, 
                  label=policy_name, 
                  linewidth=2, 
                  color=style.color,
                  linestyle=style.linestyle)
            
        catch e
            @warn "Could not load data for $policy_name: $e"
            continue
        end
    end
    
    mkpath(joinpath(config.figures_dir, "convergence_plots", "multi_policy"))
    savefig(p, joinpath(config.figures_dir, "convergence_plots/multi_policy/multi_policy_convergence_lambda_$(λ)_$(pomdp_config.log_space)_log_space.png"))
    return p
end

# Belief State Convergence Plot  
function plot_belief_convergence(policy_name, config, pomdp_config, λ; episode_idx=1)
    try
        # Load simulation histories for this lambda
        histories_dir = joinpath(config.data_dir, "simulation_histories", policy_name)
        history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
        history_file_path = joinpath(histories_dir, history_filename)
        
        if !isfile(history_file_path)
            @warn "History file not found at $history_file_path"
            return nothing
        end
        
        @load history_file_path histories
        
        # Get belief history for specified episode
        belief_hist = histories["belief_hists"][episode_idx]
        state_hist = histories["state_hists"][episode_idx]
        
        # Extract belief means and variances
        time_steps = 1:length(belief_hist)
        if pomdp_config.log_space
            belief_means = [exp(b.μ[1]) for b in belief_hist]
            belief_vars = [exp(b.Σ[1,1]) for b in belief_hist]
            actual_states = [exp(s.SeaLiceLevel) for s in state_hist]
        else
            belief_means = [b.μ[1] for b in belief_hist]
            belief_vars = [b.Σ[1,1] for b in belief_hist]
            actual_states = [s.SeaLiceLevel for s in state_hist]
        end
        
        # Create subplots
        p1 = plot(
            title="Belief Mean vs True State ($policy_name, λ = $λ)",
            xlabel="Time Step",
            ylabel="Sea Lice Level",
            grid=true,
            legend=:topleft
        )
        
        plot!(p1, time_steps, belief_means, label="Belief Mean", linewidth=2, color=:blue)
        plot!(p1, time_steps, actual_states, label="True State", linewidth=2, color=:red, linestyle=:dash)
        
        p2 = plot(
            title="Belief Uncertainty Over Time",
            xlabel="Time Step", 
            ylabel="Belief Variance",
            grid=true,
            legend=:topright
        )
        
        plot!(p2, time_steps, belief_vars, label="Belief Variance", linewidth=2, color=:green)
        
        # Combine plots
        p = plot(p1, p2, layout=(2,1), size=(800, 600))
        
        mkpath(joinpath(config.figures_dir, "convergence_plots", "belief", policy_name))
        savefig(p, joinpath(config.figures_dir, "convergence_plots/belief/$(policy_name)/belief_convergence_lambda_$(λ)_episode_$(episode_idx)_$(pomdp_config.log_space)_log_space.png"))
        return p
        
    catch e
        @warn "Could not create belief convergence plot for $policy_name: $e"
        return nothing
    end
end

# SARSOP Convergence Plot (requires custom solver tracking)
function plot_sarsop_convergence(algorithm, config, pomdp_config, λ)
    if algorithm.solver_name != "SARSOP_Policy"
        @warn "This function is only for SARSOP policies"
        return nothing
    end
    
    # Note: This would require modifying the SARSOP solver to track convergence
    # For now, this is a template showing what could be tracked
    
    p = plot(
        title="SARSOP Convergence (λ = $λ)",
        xlabel="Iteration",
        ylabel="Value Bounds Gap",
        yscale=:log10,
        grid=true,
        linewidth=2,
        legend=:topright
    )
    
    # Placeholder data - in practice you'd extract this from the solver
    iterations = 1:100
    upper_bounds = exp.(-0.1 .* iterations) .+ 50  # Decreasing upper bound
    lower_bounds = 50 .- exp.(-0.1 .* iterations)  # Increasing lower bound
    gap = upper_bounds - lower_bounds
    
    plot!(p, iterations, gap, label="Upper-Lower Bound Gap", color=:blue)
    plot!(p, iterations, upper_bounds, label="Upper Bound", color=:red, linestyle=:dash)
    plot!(p, iterations, lower_bounds, label="Lower Bound", color=:green, linestyle=:dash)
    
    mkpath(joinpath(config.figures_dir, "convergence_plots", "sarsop"))
    savefig(p, joinpath(config.figures_dir, "convergence_plots/sarsop/sarsop_convergence_lambda_$(λ)_$(pomdp_config.log_space)_log_space.png"))
    return p
end

# Convergence Rate Analysis
function plot_convergence_rate_analysis(config, pomdp_config, λ; threshold=0.01)
    p = plot(
        title="Convergence Rate Analysis (λ = $λ)",
        xlabel="Episode",
        ylabel="Relative Change in Cumulative Average",
        yscale=:log10,
        grid=true,
        legend=:topright
    )
    
    # Define colors for each policy
    policy_colors = Dict(
        "Heuristic_Policy" => :blue,
        "VI_Policy" => :red,
        "SARSOP_Policy" => :green,
        "QMDP_Policy" => :purple,
        "Random_Policy" => :orange
    )
    
    convergence_episodes = Dict{String, Int}()
    
    for (policy_name, color) in policy_colors
        try
            # Load simulation histories
            histories_dir = joinpath(config.data_dir, "simulation_histories", policy_name)
            history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
            history_file_path = joinpath(histories_dir, history_filename)
            
            if !isfile(history_file_path)
                continue
            end
            
            @load history_file_path histories
            
            # Extract episode rewards
            r_total_hists = histories["r_total_hists"]
            episodes = 2:length(r_total_hists)  # Start from episode 2
            
            # Calculate relative changes in cumulative average
            relative_changes = Float64[]
            
            for i in episodes
                cum_avg_prev = mean(r_total_hists[1:i-1])
                cum_avg_curr = mean(r_total_hists[1:i])
                
                if abs(cum_avg_prev) > 1e-10  # Avoid division by zero
                    rel_change = abs(cum_avg_curr - cum_avg_prev) / abs(cum_avg_prev)
                    push!(relative_changes, rel_change)
                else
                    push!(relative_changes, NaN)
                end
            end
            
            # Find convergence episode (first episode where relative change < threshold)
            convergence_episode = findfirst(x -> x < threshold, relative_changes)
            if convergence_episode !== nothing
                convergence_episodes[policy_name] = episodes[convergence_episode]
            end
            
            # Plot relative changes
            plot!(p, episodes, relative_changes, 
                  label=policy_name, linewidth=2, color=color)
            
        catch e
            @warn "Could not analyze convergence for $policy_name: $e"
            continue
        end
    end
    
    # Add horizontal line for threshold
    hline!(p, [threshold], linestyle=:dash, color=:black, label="Convergence Threshold")
    
    # Print convergence episodes
    println("Convergence Episodes (threshold = $threshold):")
    for (policy, episode) in convergence_episodes
        println("  $policy: Episode $episode")
    end
    
    mkpath(joinpath(config.figures_dir, "convergence_plots", "convergence_rate"))
    savefig(p, joinpath(config.figures_dir, "convergence_plots/convergence_rate/convergence_rate_lambda_$(λ)_$(pomdp_config.log_space)_log_space.png"))
    return p, convergence_episodes
end

# ----------------------------
# Example Usage Functions for Convergence Plots
# ----------------------------

# Generate all convergence plots for a specific configuration
function generate_all_convergence_plots(config, pomdp_config; λ_examples=[0.2, 0.5, 0.8])
    
    println("Generating convergence plots...")
    
    # 1. Simulation convergence for each policy
    for policy_name in ["Heuristic_Policy", "VI_Policy", "SARSOP_Policy", "QMDP_Policy", "Random_Policy"]
        for λ in λ_examples
            println("  - Simulation convergence: $policy_name, λ=$λ")
            plot_simulation_convergence(policy_name, config, pomdp_config, λ)
        end
    end
    
    # 2. Multi-policy convergence comparison
    for λ in λ_examples
        println("  - Multi-policy convergence: λ=$λ")
        plot_multi_policy_convergence(config, pomdp_config, λ)
    end
    
    # 3. Belief convergence for POMDP policies
    for policy_name in ["SARSOP_Policy", "QMDP_Policy", "Heuristic_Policy"]
        for λ in λ_examples
            println("  - Belief convergence: $policy_name, λ=$λ")
            plot_belief_convergence(policy_name, config, pomdp_config, λ)
        end
    end
    
    # 4. Convergence rate analysis
    for λ in λ_examples
        println("  - Convergence rate analysis: λ=$λ")
        plot_convergence_rate_analysis(config, pomdp_config, λ)
    end
    
    println("Convergence plots generation complete!")
end

# Quick convergence check function
function quick_convergence_check(policy_name, config, pomdp_config, λ; n_last_episodes=5)
    """
    Quick check to see if a policy has converged by comparing 
    the variance of the last n episodes
    """
    try
        # Load simulation histories
        histories_dir = joinpath(config.data_dir, "simulation_histories", policy_name)
        history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
        history_file_path = joinpath(histories_dir, history_filename)
        
        if !isfile(history_file_path)
            return false, "No history file found"
        end
        
        @load history_file_path histories
        
        # Extract episode rewards
        r_total_hists = histories["r_total_hists"]
        
        if length(r_total_hists) < n_last_episodes
            return false, "Not enough episodes"
        end
        
        # Calculate variance of last n episodes
        last_rewards = r_total_hists[end-n_last_episodes+1:end]
        reward_variance = var(last_rewards)
        reward_mean = mean(last_rewards)
        cv = sqrt(reward_variance) / abs(reward_mean)  # Coefficient of variation
        
        is_converged = cv < 0.05  # Less than 5% coefficient of variation
        
        return is_converged, "CV = $(round(cv, digits=4))"
        
    catch e
        return false, "Error: $e"
    end
end











