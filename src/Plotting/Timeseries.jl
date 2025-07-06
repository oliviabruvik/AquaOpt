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
        "Random_Policy" => (color=:orange, marker=:rect),
        "NoTreatment_Policy" => (color=:black, marker=:star)
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
        "Random_Policy" => (color=:orange, marker=:rect),
        "NoTreatment_Policy" => (color=:black, marker=:star)
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
        "Random_Policy" => (color=:orange, marker=:rect),
        "NoTreatment_Policy" => (color=:black, marker=:star)
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