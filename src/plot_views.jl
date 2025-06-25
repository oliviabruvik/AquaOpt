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
    # Calculate confidence intervals for each lambda
    lambda_values = results.lambda
    mean_costs = Float64[]
    mean_sealice = Float64[]
    cost_ci_lower = Float64[]
    cost_ci_upper = Float64[]
    sealice_ci_lower = Float64[]
    sealice_ci_upper = Float64[]
    
    for i in 1:length(lambda_values)
        # Get episode-level data
        action_hists = results.action_hists[i]
        state_hists = results.state_hists[i]
        
        # Calculate treatment costs for each episode
        episode_costs = Float64[]
        episode_sealice = Float64[]
        
        for j in 1:length(action_hists)
            # Episode treatment cost
            episode_cost = sum(a == Treatment for a in action_hists[j]) * pomdp_config.costOfTreatment
            episode_cost_per_step = episode_cost / config.steps_per_episode
            push!(episode_costs, episode_cost_per_step)
            
            # Episode sea lice level
            if pomdp_config.log_space
                episode_avg_sealice = mean(exp(s.SeaLiceLevel) for s in state_hists[j])
            else
                episode_avg_sealice = mean(s.SeaLiceLevel for s in state_hists[j])
            end
            push!(episode_sealice, episode_avg_sealice)
        end
        
        # Calculate means and 95% CI
        mean_cost = mean(episode_costs)
        std_cost = std(episode_costs)
        n_episodes = length(episode_costs)
        se_cost = std_cost / sqrt(n_episodes)
        cost_ci_margin = 1.96 * se_cost
        
        mean_lice = mean(episode_sealice)
        std_lice = std(episode_sealice)
        se_lice = std_lice / sqrt(n_episodes)
        lice_ci_margin = 1.96 * se_lice
        
        push!(mean_costs, mean_cost)
        push!(mean_sealice, mean_lice)
        push!(cost_ci_lower, mean_cost - cost_ci_margin)
        push!(cost_ci_upper, mean_cost + cost_ci_margin)
        push!(sealice_ci_lower, mean_lice - lice_ci_margin)
        push!(sealice_ci_upper, mean_lice + lice_ci_margin)
        
        # Verify our calculation matches the stored average
        if abs(mean_cost - results.avg_treatment_cost[i]) > 1e-10
            @warn "Calculated mean cost ($mean_cost) doesn't match stored average ($(results.avg_treatment_cost[i])) for λ=$(lambda_values[i])"
        end
        if abs(mean_lice - results.avg_sealice[i]) > 1e-10
            @warn "Calculated mean sea lice ($mean_lice) doesn't match stored average ($(results.avg_sealice[i])) for λ=$(lambda_values[i])"
        end
    end

    p = plot(
        title="Treatment Cost vs Sea Lice Levels ($title)",
        xlabel="Average Treatment Cost (MNOK / year)", 
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=false,
        grid=true,
        framestyle=:box
    )
    
    # Plot main scatter points
    scatter!(
        p,
        mean_costs,
        mean_sealice,
        marker_z=lambda_values,
        marker=:circle,
        markersize=8,
        c=:viridis,
        colorbar=true,
        colorbar_title="λ (Cost-Benefit Trade-off)",
        alpha=0.8
    )
    
    # Add error bars for each point
    for i in 1:length(mean_costs)
        # Vertical error bar (sea lice uncertainty)
        plot!(
            p,
            [mean_costs[i], mean_costs[i]],
            [sealice_ci_lower[i], sealice_ci_upper[i]],
            color=:black,
            alpha=0.6,
            linewidth=1.5,
            label=""
        )
        
        # Horizontal error bar (cost uncertainty)  
        plot!(
            p,
            [cost_ci_lower[i], cost_ci_upper[i]],
            [mean_sealice[i], mean_sealice[i]],
            color=:black,
            alpha=0.6,
            linewidth=1.5,
            label=""
        )
    end

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
        title="Pareto Frontier: Treatment Cost vs Sea Lice Levels",
        xlabel="Average Treatment Cost (MNOK / year)", 
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:bottomleft,
        grid=true,
        framestyle=:box
    )
    
    # Define colors for each policy
    policy_colors = Dict(
        "VI_Policy" => :red,
        "SARSOP_Policy" => :green,
        "QMDP_Policy" => :purple,
        "Heuristic_Policy" => :blue,
        "Random_Policy" => :orange
    )
    
    # Load and plot each policy's results
    for (policy_name, color) in policy_colors
        try
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Calculate confidence intervals for each lambda
            lambda_values = results.lambda
            mean_costs = Float64[]
            mean_sealice = Float64[]
            cost_ci_lower = Float64[]
            cost_ci_upper = Float64[]
            sealice_ci_lower = Float64[]
            sealice_ci_upper = Float64[]
            
            for i in 1:length(lambda_values)
                # Get episode-level data
                action_hists = results.action_hists[i]
                state_hists = results.state_hists[i]
                
                # Calculate treatment costs for each episode
                episode_costs = Float64[]
                episode_sealice = Float64[]
                
                for j in 1:length(action_hists)
                    # Episode treatment cost
                    episode_cost = sum(a == Treatment for a in action_hists[j]) * pomdp_config.costOfTreatment
                    episode_cost_per_step = episode_cost / config.steps_per_episode
                    push!(episode_costs, episode_cost_per_step)
                    
                    # Episode sea lice level
                    if pomdp_config.log_space
                        episode_avg_sealice = mean(exp(s.SeaLiceLevel) for s in state_hists[j])
                    else
                        episode_avg_sealice = mean(s.SeaLiceLevel for s in state_hists[j])
                    end
                    push!(episode_sealice, episode_avg_sealice)
                end
                
                # Calculate means and 95% CI
                mean_cost = mean(episode_costs)
                std_cost = std(episode_costs)
                n_episodes = length(episode_costs)
                se_cost = std_cost / sqrt(n_episodes)
                cost_ci_margin = 1.96 * se_cost
                
                mean_lice = mean(episode_sealice)
                std_lice = std(episode_sealice)
                se_lice = std_lice / sqrt(n_episodes)
                lice_ci_margin = 1.96 * se_lice
                
                push!(mean_costs, mean_cost)
                push!(mean_sealice, mean_lice)
                push!(cost_ci_lower, mean_cost - cost_ci_margin)
                push!(cost_ci_upper, mean_cost + cost_ci_margin)
                push!(sealice_ci_lower, mean_lice - lice_ci_margin)
                push!(sealice_ci_upper, mean_lice + lice_ci_margin)
            end
            
            # Sort the data by treatment cost for smooth lines
            sort_indices = sortperm(mean_costs)
            sorted_costs = mean_costs[sort_indices]
            sorted_sealice = mean_sealice[sort_indices]
            sorted_cost_ci_lower = cost_ci_lower[sort_indices]
            sorted_cost_ci_upper = cost_ci_upper[sort_indices]
            sorted_sealice_ci_lower = sealice_ci_lower[sort_indices]
            sorted_sealice_ci_upper = sealice_ci_upper[sort_indices]
            
            # Plot the main line
            plot!(
                p,
                sorted_costs,
                sorted_sealice,
                linewidth=2,
                color=color,
                label=policy_name,
                alpha=0.8
            )
            
            # Add error bars for both dimensions
            for i in 1:length(sorted_costs)
                # Vertical error bar (sea lice uncertainty)
                plot!(
                    p,
                    [sorted_costs[i], sorted_costs[i]],
                    [sorted_sealice_ci_lower[i], sorted_sealice_ci_upper[i]],
                    color=color,
                    alpha=0.4,
                    linewidth=1,
                    label=""
                )
                
                # Horizontal error bar (cost uncertainty)
                plot!(
                    p,
                    [sorted_cost_ci_lower[i], sorted_cost_ci_upper[i]],
                    [sorted_sealice[i], sorted_sealice[i]],
                    color=color,
                    alpha=0.4,
                    linewidth=1,
                    label=""
                )
            end
            
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
# Plot 4: Lambda vs sea lice levels for each policy
# ----------------------------
function plot_policy_sealice_levels_over_lambdas(config, pomdp_config)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Average Sea Lice Levels over Lambda",
        xlabel="Lambda (λ)",
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:bottomleft,
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
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Calculate per-episode sea lice levels and 95% CI for each lambda
            lambda_values = results.lambda
            mean_sealice = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for i in 1:length(lambda_values)
                state_hists = results.state_hists[i]
                
                # Calculate sea lice level for each episode
                episode_sealice = Float64[]
                for episode_states in state_hists
                    # Handle both regular and log space states
                    if pomdp_config.log_space
                        episode_avg = mean(exp(s.SeaLiceLevel) for s in episode_states)
                    else
                        episode_avg = mean(s.SeaLiceLevel for s in episode_states)
                    end
                    push!(episode_sealice, episode_avg)
                end
                
                # Calculate mean and 95% CI
                mean_level = mean(episode_sealice)
                std_level = std(episode_sealice)
                n_episodes = length(episode_sealice)
                se_level = std_level / sqrt(n_episodes)  # Standard error
                ci_margin = 1.96 * se_level  # 95% CI margin
                
                push!(mean_sealice, mean_level)
                push!(ci_lower, mean_level - ci_margin)
                push!(ci_upper, mean_level + ci_margin)
                
                # Verify our calculation matches the stored average
                stored_avg = results.avg_sealice[i]
                if abs(mean_level - stored_avg) > 1e-10
                    @warn "Calculated mean ($mean_level) doesn't match stored average ($stored_avg) for λ=$(lambda_values[i])"
                end
            end
            
            # Add the line plot with 95% confidence interval ribbon
            plot!(
                p,
                lambda_values,
                mean_sealice,
                ribbon=(mean_sealice .- ci_lower, ci_upper .- mean_sealice),
                label=policy_name,
                color=style.color,
                linewidth=2,
                fillalpha=0.3,
                alpha=0.7
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "sealice_lambda_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "sealice_lambda_plots/All_policies/sealice_lambda_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 5: Lambda vs treatment cost for each policy
# ----------------------------
function plot_policy_treatment_cost_over_lambdas(config, pomdp_config)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Average Treatment Cost Over Lambda",
        xlabel="Lambda (λ)",
        ylabel="Average Treatment Cost (MNOK / year)",
        legend=:bottomleft,
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
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Calculate per-episode treatment costs and 95% CI for each lambda
            lambda_values = results.lambda
            mean_costs = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for i in 1:length(lambda_values)
                action_hists = results.action_hists[i]
                
                # Calculate treatment cost for each episode
                episode_costs = Float64[]
                for episode_actions in action_hists
                    episode_cost = sum(a == Treatment for a in episode_actions) * pomdp_config.costOfTreatment
                    episode_cost_per_step = episode_cost / config.steps_per_episode
                    push!(episode_costs, episode_cost_per_step)
                end
                
                # Calculate mean and 95% CI
                mean_cost = mean(episode_costs)
                std_cost = std(episode_costs)
                n_episodes = length(episode_costs)
                se_cost = std_cost / sqrt(n_episodes)  # Standard error
                ci_margin = 1.96 * se_cost  # 95% CI margin
                
                push!(mean_costs, mean_cost)
                push!(ci_lower, mean_cost - ci_margin)
                push!(ci_upper, mean_cost + ci_margin)
                
                # Verify our calculation matches the stored average
                stored_avg = results.avg_treatment_cost[i]
                if abs(mean_cost - stored_avg) > 1e-10
                    @warn "Calculated mean ($mean_cost) doesn't match stored average ($stored_avg) for λ=$(lambda_values[i])"
                end
            end
            
            # Add the line plot with 95% confidence interval ribbon
            plot!(
                p,
                lambda_values,
                mean_costs,
                ribbon=(mean_costs .- ci_lower, ci_upper .- mean_costs),
                label=policy_name,
                color=style.color,
                linewidth=2,
                fillalpha=0.3,
                alpha=0.7
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "treatment_cost_lambda_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_lambda_plots/All_policies/treatment_cost_lambda_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 6: Time-series of belief for each policy
# NOTE: RESULTS ARE IN LOG SPACE
# ----------------------------
function plot_policy_belief_levels(results, title, config, pomdp_config, lambda; show_actual_states=true)

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
        belief_stds = [sqrt(exp(belief.Σ[1,1])) for belief in first_episode_belief_hist]
        actual_states = [exp(s.SeaLiceLevel) for s in first_episode_state_hist]
    else
        belief_means = [belief.μ[1] for belief in first_episode_belief_hist]
        belief_stds = [sqrt(belief.Σ[1,1]) for belief in first_episode_belief_hist]
        actual_states = [s.SeaLiceLevel for s in first_episode_state_hist]
    end

    y_lim = max(maximum(belief_means), maximum(actual_states)) * 1.1

    # Initialize the plot with the y-axis range starting at 0 always
    p = plot(
        title="Belief State Evolution ($title)",
        xlabel="Time Step (Weeks)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:topleft,
        grid=true,
        ylims=(0, y_lim)
    )
    
    # Calculate 95% confidence intervals (mean ± 1.96 * std)
    ci_lower = belief_means .- 1.96 .* belief_stds
    ci_upper = belief_means .+ 1.96 .* belief_stds
    
    # Add the line plot with 95% confidence interval ribbon
    plot!(
        p,
        1:length(belief_means),
        belief_means,
        ribbon=(belief_means .- ci_lower, ci_upper .- belief_means),
        label="Belief (95% CI)",
        color=:blue,
        linewidth=2,
        fillalpha=0.3
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
    
    return p
end

# ----------------------------
# Plot 7: Time-series of sea lice levels for each policy at specific lambda
# ----------------------------
function plot_policy_sealice_levels_over_time(config, pomdp_config, lambda_value)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Sea Lice Levels Over Time (λ = $lambda_value)",
        xlabel="Time Step",
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:bottomright,
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
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Find the index for the specified lambda value
            lambda_index = findfirst(λ -> abs(λ - lambda_value) < 1e-10, results.lambda)
            if lambda_index === nothing
                @warn "Lambda value $lambda_value not found for $policy_name"
                continue
            end
            
            # Get state histories for this lambda
            state_hists = results.state_hists[lambda_index]
            
            # Calculate mean and 95% CI for each time step
            time_steps = 1:config.steps_per_episode
            mean_sealice = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract sea lice level at time step t from all episodes
                step_sealice = Float64[]
                for episode_states in state_hists
                    if t <= length(episode_states)
                        # Handle both regular and log space states
                        if pomdp_config.log_space
                            sealice_level = exp(episode_states[t].SeaLiceLevel)
                        else
                            sealice_level = episode_states[t].SeaLiceLevel
                        end
                        push!(step_sealice, sealice_level)
                    end
                end
                
                if !isempty(step_sealice)
                    # Calculate mean and 95% CI
                    mean_level = mean(step_sealice)
                    std_level = std(step_sealice)
                    n_episodes = length(step_sealice)
                    se_level = std_level / sqrt(n_episodes)  # Standard error
                    ci_margin = 1.96 * se_level  # 95% CI margin
                    
                    push!(mean_sealice, mean_level)
                    push!(ci_lower, mean_level - ci_margin)
                    push!(ci_upper, mean_level + ci_margin)
                else
                    push!(mean_sealice, NaN)
                    push!(ci_lower, NaN)
                    push!(ci_upper, NaN)
                end
            end
            
            # Remove NaN values
            valid_indices = .!isnan.(mean_sealice)
            valid_time_steps = time_steps[valid_indices]
            valid_mean = mean_sealice[valid_indices]
            valid_ci_lower = ci_lower[valid_indices]
            valid_ci_upper = ci_upper[valid_indices]
            
            # Add the line plot with 95% confidence interval ribbon
            plot!(
                p,
                valid_time_steps,
                valid_mean,
                ribbon=(valid_mean .- valid_ci_lower, valid_ci_upper .- valid_mean),
                label=policy_name,
                color=style.color,
                linewidth=2,
                fillalpha=0.3,
                alpha=0.7
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "sealice_time_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "sealice_time_plots/All_policies/sealice_time_lambda_$(lambda_value)_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 8: Time-series of treatment cost for each policy at specific lambda
# ----------------------------
function plot_policy_treatment_cost_over_time(config, pomdp_config, lambda_value)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Treatment Cost Over Time (λ = $lambda_value)",
        xlabel="Time Step (Weeks)",
        ylabel="Treatment Probability",
        legend=:bottomleft,
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
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Find the index for the specified lambda value
            lambda_index = findfirst(λ -> abs(λ - lambda_value) < 1e-10, results.lambda)
            if lambda_index === nothing
                @warn "Lambda value $lambda_value not found for $policy_name"
                continue
            end
            
            # Get ACTION histories for this lambda (not state histories!)
            action_hists = results.action_hists[lambda_index]
            
            # Calculate mean treatment probability and 95% CI for each time step
            time_steps = 1:config.steps_per_episode
            mean_treatment_prob = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract treatment decisions at time step t from all episodes
                step_treatments = Float64[]
                for episode_actions in action_hists
                    if t <= length(episode_actions)
                        # Treatment probability: 1 if Treatment, 0 if NoTreatment
                        treatment_indicator = episode_actions[t] == Treatment ? 1.0 : 0.0
                        push!(step_treatments, treatment_indicator)
                    end
                end
                
                if !isempty(step_treatments)
                    # Calculate mean and 95% CI
                    mean_prob = mean(step_treatments)
                    std_prob = std(step_treatments)
                    n_episodes = length(step_treatments)
                    se_prob = std_prob / sqrt(n_episodes)  # Standard error
                    ci_margin = 1.96 * se_prob  # 95% CI margin
                    
                    push!(mean_treatment_prob, mean_prob)
                    push!(ci_lower, mean_prob - ci_margin)
                    push!(ci_upper, mean_prob + ci_margin)
                else
                    push!(mean_treatment_prob, NaN)
                    push!(ci_lower, NaN)
                    push!(ci_upper, NaN)
                end
            end
            
            # Remove NaN values
            valid_indices = .!isnan.(mean_treatment_prob)
            valid_time_steps = time_steps[valid_indices]
            valid_mean = mean_treatment_prob[valid_indices]
            valid_ci_lower = ci_lower[valid_indices]
            valid_ci_upper = ci_upper[valid_indices]
            
            # Add the line plot with 95% confidence interval ribbon
            plot!(
                p,
                valid_time_steps,
                valid_mean,
                ribbon=(valid_mean .- valid_ci_lower, valid_ci_upper .- valid_mean),
                label=policy_name,
                color=style.color,
                linewidth=2,
                fillalpha=0.3,
                alpha=0.7
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "treatment_cost_time_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_time_plots/All_policies/treatment_cost_time_lambda_$(lambda_value)_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 8b: Time-series of actual treatment cost (probability * cost) for each policy at specific lambda
# ----------------------------
function plot_policy_actual_treatment_cost_over_time(config, pomdp_config, lambda_value)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Actual Treatment Cost Over Time (λ = $lambda_value)",
        xlabel="Time Step (Weeks)",
        ylabel="Expected Treatment Cost per Step",
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
            # Load the results from the JLD2 file
            @load "results/data/avg_results/$(policy_name)/avg_results_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.jld2" results
            
            # Find the index for the specified lambda value
            lambda_index = findfirst(λ -> abs(λ - lambda_value) < 1e-10, results.lambda)
            if lambda_index === nothing
                @warn "Lambda value $lambda_value not found for $policy_name"
                continue
            end
            
            # Get ACTION histories for this lambda
            action_hists = results.action_hists[lambda_index]
            
            # Calculate mean treatment cost and 95% CI for each time step
            time_steps = 1:config.steps_per_episode
            mean_treatment_cost = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract treatment costs at time step t from all episodes
                step_costs = Float64[]
                for episode_actions in action_hists
                    if t <= length(episode_actions)
                        # Treatment cost: costOfTreatment if Treatment, 0 if NoTreatment
                        treatment_cost = episode_actions[t] == Treatment ? pomdp_config.costOfTreatment : 0.0
                        push!(step_costs, treatment_cost)
                    end
                end
                
                if !isempty(step_costs)
                    # Calculate mean and 95% CI
                    mean_cost = mean(step_costs)
                    std_cost = std(step_costs)
                    n_episodes = length(step_costs)
                    se_cost = std_cost / sqrt(n_episodes)  # Standard error
                    ci_margin = 1.96 * se_cost  # 95% CI margin
                    
                    push!(mean_treatment_cost, mean_cost)
                    push!(ci_lower, mean_cost - ci_margin)
                    push!(ci_upper, mean_cost + ci_margin)
                else
                    push!(mean_treatment_cost, NaN)
                    push!(ci_lower, NaN)
                    push!(ci_upper, NaN)
                end
            end
            
            # Remove NaN values
            valid_indices = .!isnan.(mean_treatment_cost)
            valid_time_steps = time_steps[valid_indices]
            valid_mean = mean_treatment_cost[valid_indices]
            valid_ci_lower = ci_lower[valid_indices]
            valid_ci_upper = ci_upper[valid_indices]
            
            # Add the line plot with 95% confidence interval ribbon
            plot!(
                p,
                valid_time_steps,
                valid_mean,
                ribbon=(valid_mean .- valid_ci_lower, valid_ci_upper .- valid_mean),
                label=policy_name,
                color=style.color,
                linewidth=2,
                fillalpha=0.3,
                alpha=0.7
            )
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end
    mkpath(joinpath(config.figures_dir, "actual_treatment_cost_time_plots", "All_policies"))
    savefig(p, joinpath(config.figures_dir, "actual_treatment_cost_time_plots/All_policies/actual_treatment_cost_time_lambda_$(lambda_value)_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 9: Treatment Decision Heatmap
# ----------------------------
function plot_treatment_heatmap(algorithm, config, pomdp_config)

    # Get lambda values from config
    lambda_values = config.lambda_values
    
    # Initialize matrix to store treatment decisions
    # Rows = states, Columns = lambda values (transposed from before)
    treatment_matrix = nothing
    y_vals = nothing
    
    policies_dir = joinpath(config.data_dir, "policies", algorithm.solver_name)
    
    for (i, λ) in enumerate(lambda_values)
        # Load policy, pomdp, and mdp for this lambda
        policy_file_path = joinpath(policies_dir, "policy_pomdp_mdp_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2")
        
        if !isfile(policy_file_path)
            @warn "Policy file not found at $policy_file_path for λ=$λ"
            continue
        end
        
        @load policy_file_path policy pomdp mdp

        # Create state space grid
        if pomdp_config.log_space
            states = [SeaLiceLogState(x) for x in pomdp.log_sea_lice_range]
            if y_vals === nothing
                y_vals = [exp(s.SeaLiceLevel) for s in states]  # Convert back to original space for y-axis
            end
        else
            states = [SeaLiceState(x) for x in pomdp.sea_lice_range]
            if y_vals === nothing
                y_vals = [s.SeaLiceLevel for s in states]
            end
        end

        function getAction(s, policy)
            if typeof(policy) <: ValueIterationPolicy
                return action(policy, s)
            else
                # For other POMDP policies, create a deterministic belief vector
                # where the current state has probability 1
                state_space = states
                n_states = length(state_space)
                bvec = zeros(n_states)
                
                # Find the index of the current state
                state_idx = findfirst(st -> abs(st.SeaLiceLevel - s.SeaLiceLevel) < 1e-6, state_space)
                if state_idx !== nothing
                    bvec[state_idx] = 1.0
                else
                    # If exact match not found, find closest state
                    distances = [abs(st.SeaLiceLevel - s.SeaLiceLevel) for st in state_space]
                    state_idx = argmin(distances)
                    bvec[state_idx] = 1.0
                end
                
                return action(policy, bvec)
            end
        end

        ## Get a hundred actions to see percentage of treatment decisions
        function getPercentageOfTreatmentDecisions(policy, s)
            actions = [getAction(s, policy) for _ in 1:100]
            return sum(actions .== Treatment) / 100
        end

        percentage_of_treatment_decisions = [getPercentageOfTreatmentDecisions(policy, s) for s in states]
        
        # Initialize matrix on first iteration
        # Now: Rows = states, Columns = lambda values
        if treatment_matrix === nothing
            treatment_matrix = zeros(length(states), length(lambda_values))
        end
        
        # Store decisions for this lambda (column i, all states)
        treatment_matrix[:, i] = percentage_of_treatment_decisions
    end
    
    if treatment_matrix === nothing
        @warn "No policies found to create heatmap"
        return nothing
    end

    p = heatmap(
        lambda_values,      # x-axis: lambda values
        y_vals,            # y-axis: sea lice levels
        treatment_matrix,  # matrix: states × lambdas
        title="Treatment Decision Heatmap: λ vs Sea Lice Level",
        xlabel="λ (Cost-Benefit Trade-off Parameter)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        colorbar_title="Treatment Decision",
        c=:RdYlBu,  # Red-Yellow-Blue colormap (red=treat, blue=no treat)
        aspect_ratio=:auto
    )

    # Save plot
    plot_dir = joinpath(config.figures_dir, "treatment_heatmaps", algorithm.solver_name)
    mkpath(plot_dir)
    savefig(p, joinpath(plot_dir, "treatment_heatmap_lambda_vs_state_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 10: Simulation-Based Treatment Decision Heatmap
# ----------------------------
function plot_simulation_treatment_heatmap(algorithm, config, pomdp_config; use_observations=false, n_bins=50)
    
    # Get lambda values from config
    lambda_values = config.lambda_values
    
    # Initialize matrix to store treatment frequencies
    # Rows = sea lice level bins, Columns = lambda values (transposed from before)
    treatment_freq_matrix = nothing
    bin_centers = nothing
    
    histories_dir = joinpath(config.data_dir, "simulation_histories", algorithm.solver_name)
    
    for (i, λ) in enumerate(lambda_values)
        try
            # Load simulation histories for this lambda
            history_filename = "hists_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps_$(λ)_lambda.jld2"
            history_file_path = joinpath(histories_dir, history_filename)
            
            if !isfile(history_file_path)
                @warn "History file not found at $history_file_path for λ=$λ"
                continue
            end
            
            @load history_file_path histories
            
            # Extract data from all episodes
            all_states = []
            all_actions = []
            
            for episode in 1:length(histories["state_hists"])
                state_hist = histories["state_hists"][episode]
                action_hist = histories["action_hists"][episode]
                
                # Convert states to sea lice levels
                if use_observations
                    # Use measurements instead of states if requested
                    measurement_hist = histories["measurement_hists"][episode]
                    sea_lice_levels = if pomdp_config.log_space
                        [exp(o.SeaLiceLevel) for o in measurement_hist]
                    else
                        [o.SeaLiceLevel for o in measurement_hist]
                    end
                else
                    # Use actual states
                    sea_lice_levels = if pomdp_config.log_space
                        [exp(s.SeaLiceLevel) for s in state_hist]
                    else
                        [s.SeaLiceLevel for s in state_hist]
                    end
                end
                
                append!(all_states, sea_lice_levels)
                append!(all_actions, action_hist)
            end
            
            # Create bins for sea lice levels on first iteration
            if bin_centers === nothing
                min_level = minimum(all_states)
                max_level = maximum(all_states)
                # Add small buffer to ensure all data points fit
                min_level = max(0.0, min_level - 0.1)
                max_level = max_level + 0.1
                
                bin_edges = range(min_level, stop=max_level, length=n_bins+1)
                bin_centers = [(bin_edges[j] + bin_edges[j+1]) / 2 for j in 1:n_bins]
                
                # Initialize matrix: Rows = bins, Columns = lambda values
                treatment_freq_matrix = zeros(n_bins, length(lambda_values))
            end
            
            # Bin the data and calculate treatment frequencies
            bin_edges = range(minimum(bin_centers) - (bin_centers[2] - bin_centers[1])/2, 
                            stop=maximum(bin_centers) + (bin_centers[2] - bin_centers[1])/2, 
                            length=n_bins+1)
            
            for bin_idx in 1:n_bins
                # Find states in this bin
                in_bin = (all_states .>= bin_edges[bin_idx]) .& (all_states .< bin_edges[bin_idx+1])
                
                if sum(in_bin) > 0
                    # Calculate treatment frequency for this bin
                    actions_in_bin = all_actions[in_bin]
                    treatment_freq = sum(a == Treatment for a in actions_in_bin) / length(actions_in_bin)
                    treatment_freq_matrix[bin_idx, i] = treatment_freq  # Row = bin, Column = lambda
                else
                    # No data in this bin
                    treatment_freq_matrix[bin_idx, i] = NaN
                end
            end
            
        catch e
            @warn "Failed to process simulation histories for λ=$λ: $e"
            continue
        end
    end
    
    if treatment_freq_matrix === nothing
        @warn "No simulation histories could be loaded to create heatmap"
        return nothing
    end
    
    # Replace NaN values with a neutral color (0.5)
    treatment_freq_matrix[isnan.(treatment_freq_matrix)] .= 0.5

    data_type = use_observations ? "Observations" : "States"
    p = heatmap(
        lambda_values,           # x-axis: lambda values  
        bin_centers,            # y-axis: sea lice level bins
        treatment_freq_matrix,  # matrix: bins × lambdas
        title="Simulation Treatment Frequency: λ vs Sea Lice Level ($data_type)",
        xlabel="λ (Cost-Benefit Trade-off Parameter)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        colorbar_title="Treatment Frequency",
        c=:RdYlBu,  # Red-Yellow-Blue colormap
        aspect_ratio=:auto,
        clims=(0, 1)  # Ensure color scale goes from 0 to 1
    )

    # Save plot
    plot_dir = joinpath(config.figures_dir, "simulation_treatment_heatmaps", algorithm.solver_name)
    mkpath(plot_dir)
    filename_suffix = use_observations ? "observations" : "states"
    savefig(p, joinpath(plot_dir, "simulation_treatment_heatmap_$(filename_suffix)_$(pomdp_config.log_space)_log_space_$(config.num_episodes)_episodes_$(config.steps_per_episode)_steps.png"))
    return p
end

# ----------------------------
# Plot 11: Lambda vs average reward for each policy
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