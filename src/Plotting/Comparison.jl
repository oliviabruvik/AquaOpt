using Plots
using JLD2
plotlyjs()  # Set the backend to PlotlyJS

# ----------------------------
# Plot 2: Cost vs Sea Lice for one policy
# ----------------------------
function plot_policy_cost_vs_sealice(histories, avg_results, title, config)

    # Calculate confidence intervals for each lambda
    lambda_values = collect(keys(histories))
    mean_costs = Float64[]
    mean_sealice = Float64[]
    cost_ci_lower = Float64[]
    cost_ci_upper = Float64[]
    sealice_ci_lower = Float64[]
    sealice_ci_upper = Float64[]
    plotted_lambdas = Float64[]

    for λ in config.lambda_values

        # Get episode-level data
        histories_lambda = histories[λ]
        
        # Calculate treatment costs for each episode
        episode_costs = Float64[]
        episode_sealice = Float64[]
        
        for episode in 1:config.num_episodes

            # Get episode history
            episode_history = histories_lambda[episode]

            # Get action, state, and reward histories
            actions = collect(action_hist(episode_history))
            states = collect(state_hist(episode_history))
            rewards = collect(reward_hist(episode_history))
            
            # Episode treatment cost
            episode_cost = sum(a == Treatment for a in actions) * config.costOfTreatment
            push!(episode_costs, episode_cost)
            
            # Episode sea lice level
            # if config.log_space
            #     episode_avg_sealice = mean(exp(s.SeaLiceLevel) for s in states)
            # else
            episode_avg_sealice = mean(s.SeaLiceLevel for s in states)
            # end
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
        push!(plotted_lambdas, λ)
        
        # Find the corresponding row in avg_results for this lambda
        avg_row = avg_results[avg_results.lambda .== λ, :]
        if !isempty(avg_row)
            stored_cost = avg_row.avg_treatment_cost[1]
            stored_lice = avg_row.avg_sealice[1]
            
            # Verify our calculation matches the stored average
            if abs(mean_cost - stored_cost) > 1e-10
                @warn "Calculated mean cost ($mean_cost) doesn't match stored average ($stored_cost) for λ=$λ"
            end
            if abs(mean_lice - stored_lice) > 1e-10
                @warn "Calculated mean sea lice ($mean_lice) doesn't match stored average ($stored_lice) for λ=$λ"
            end
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
        marker_z=plotted_lambdas,
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

    mkpath(joinpath(config.figures_dir, "treatment_cost_vs_sealice_plots"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_vs_sealice_plots/$(title)_results.png"))
    return p
end

# ----------------------------
# Plot 3: Overlay all policies
# ----------------------------
function plot_all_cost_vs_sealice(config)
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
        "NUS_SARSOP_Policy" => :green,
        "QMDP_Policy" => :purple,
        "Heuristic_Policy" => :blue,
        "Random_Policy" => :orange,
        # "NeverTreat_Policy" => :black,
        "AlwaysTreat_Policy" => :brown
    )
    
    # Load and plot each policy's results
    for (policy_name, color) in policy_colors
        try
            # Load the results from the JLD2 files
            @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
            @load joinpath(config.simulations_dir, "$(policy_name)", "$(policy_name)_histories.jld2") histories
            
            # Calculate confidence intervals for each lambda
            lambda_values = config.lambda_values
            mean_costs = Float64[]
            mean_sealice = Float64[]
            cost_ci_lower = Float64[]
            cost_ci_upper = Float64[]
            sealice_ci_lower = Float64[]
            sealice_ci_upper = Float64[]
            
            for λ in config.lambda_values
                # Get episode-level data
                histories_lambda = histories[λ]
                
                # Calculate treatment costs for each episode
                episode_costs = Float64[]
                episode_sealice = Float64[]
                
                for episode in 1:config.num_episodes
                    # Get episode history
                    episode_history = histories_lambda[episode]

                    # Get action, state, and reward histories
                    actions = collect(action_hist(episode_history))
                    states = collect(state_hist(episode_history))

                    # Episode treatment cost
                    episode_cost = sum(a == Treatment for a in actions) * config.costOfTreatment
                    push!(episode_costs, episode_cost)
                    
                    # Episode sea lice level
                    # if config.log_space
                    #     episode_avg_sealice = mean(exp(s.SeaLiceLevel) for s in states)
                    # else
                    episode_avg_sealice = mean(s.SeaLiceLevel for s in states)
                    # end
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
    mkpath(joinpath(config.figures_dir, "Policy_Comparison"))
    savefig(p, joinpath(config.figures_dir, "Policy_Comparison", "treatment_cost_vs_sealice_plot.png"))
    return p
end

# ----------------------------
# Plot 4: Lambda vs sea lice levels for each policy
# ----------------------------
function plot_policy_sealice_levels_over_lambdas(config)
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
        "NUS_SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle),
        "Random_Policy" => (color=:orange, marker=:rect),
        # "NeverTreat_Policy" => (color=:black, marker=:star),
        "AlwaysTreat_Policy" => (color=:brown, marker=:dtriangle)
    )
    
    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try
            # Load the results from the JLD2 file
            @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
            @load joinpath(config.simulations_dir, "$(policy_name)", "$(policy_name)_histories.jld2") histories
            
            # Calculate per-episode sea lice levels and 95% CI for each lambda
            lambda_values = avg_results.lambda
            mean_sealice = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for (i, λ) in enumerate(config.lambda_values)
                histories_lambda = histories[λ]

                # Calculate sea lice level for each episode
                episode_sealice = Float64[]
                for episode in 1:config.num_episodes
                    episode_history = histories_lambda[episode]
                    states = collect(state_hist(episode_history))
                    # if config.log_space
                    #     episode_avg = mean(exp(s.SeaLiceLevel) for s in states)
                    # else
                    #     episode_avg = mean(s.SeaLiceLevel for s in states)
                    # end
                    episode_avg = mean(s.SeaLiceLevel for s in states)
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
                stored_avg = avg_results.avg_sealice[i]
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
    mkpath(joinpath(config.figures_dir, "Policy_Comparison"))
    savefig(p, joinpath(config.figures_dir, "Policy_Comparison", "sealice_lambda_plot.png"))
    return p
end

# ----------------------------
# Plot 5: Lambda vs treatment cost for each policy
# ----------------------------
function plot_policy_treatment_cost_over_lambdas(config)
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
        "NUS_SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle),
        "Random_Policy" => (color=:orange, marker=:rect),
        # "NeverTreat_Policy" => (color=:black, marker=:star),
        "AlwaysTreat_Policy" => (color=:brown, marker=:dtriangle)
    )
    
    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try
            # Load the results from the JLD2 file
            @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
            @load joinpath(config.simulations_dir, "$(policy_name)", "$(policy_name)_histories.jld2") histories
            
            # Calculate per-episode treatment costs and 95% CI for each lambda
            lambda_values = avg_results.lambda
            mean_costs = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for (i, λ) in enumerate(config.lambda_values)
                histories_lambda = histories[λ]
                
                # Calculate treatment cost for each episode
                episode_costs = Float64[]
                for episode in 1:config.num_episodes
                    episode_history = histories_lambda[episode]
                    actions = collect(action_hist(episode_history))
                    episode_cost = sum(a == Treatment for a in actions) * config.costOfTreatment
                    push!(episode_costs, episode_cost)
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
                stored_avg = avg_results.avg_treatment_cost[i]
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
    mkpath(joinpath(config.figures_dir, "Policy_Comparison"))
    savefig(p, joinpath(config.figures_dir, "Policy_Comparison", "treatment_cost_lambda_plot.png"))
    return p
end

# ----------------------------
# Plot 11: Lambda vs average reward for each policy
# ----------------------------
function plot_policy_reward_over_lambdas(config)
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
        "NUS_SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle),
        "Random_Policy" => (color=:orange, marker=:rect),
        # "NeverTreat_Policy" => (color=:black, marker=:star),
        "AlwaysTreat_Policy" => (color=:brown, marker=:dtriangle)
    )
    
    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try
            # Load the simulation histories from the JLD2 file
            histories_dir = joinpath(config.simulations_dir, "$(policy_name)")
            histories_filename = "$(policy_name)_histories"
            @load joinpath(histories_dir, "$(histories_filename).jld2") histories

            mean_rewards = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            lambda_values = config.lambda_values
            
            for λ in config.lambda_values
                try
                    # Load simulation histories for this lambda
                    histories_lambda = histories[λ]
                    if isempty(histories_lambda)
                        @warn "No histories found for λ=$λ"
                        push!(mean_rewards, NaN)
                        push!(ci_lower, NaN)
                        push!(ci_upper, NaN)
                        continue
                    end

                    episode_rewards = Float64[]

                    for episode in 1:config.num_episodes
                        episode_history = histories_lambda[episode]
                        rewards = collect(reward_hist(episode_history))
                        episode_reward = mean(rewards)
                        push!(episode_rewards, episode_reward)
                    end
                    
                    # Calculate mean and 95% CI
                    mean_reward = mean(episode_rewards)
                    std_reward = std(episode_rewards)
                    n_episodes = length(episode_rewards)
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
    mkpath(joinpath(config.figures_dir, "Policy_Comparison"))
    savefig(p, joinpath(config.figures_dir, "Policy_Comparison", "reward_lambda_plot.png"))
    return p
end

# Pareto Frontier Plot
function plot_pareto_frontier(config)
   
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
        "NUS_SARSOP_Policy" => (color=:green, marker=:diamond),
        "QMDP_Policy" => (color=:purple, marker=:dtriangle)
    )

    # Plot each policy's results
    for (policy_name, style) in policy_styles
        @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
        scatter!(
            p,
            avg_results.avg_treatment_cost,
            avg_results.avg_sealice,
            label=policy_name,
            color=style.color,
            marker=style.marker,
            alpha=0.7
        )
    end

    # Save plot
    mkpath(joinpath(config.figures_dir, "research_plots"))
    savefig(p, joinpath(config.figures_dir, "research_plots", "pareto_frontier.png"))
    return p
end