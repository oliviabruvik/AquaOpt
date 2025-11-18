using Plots
using JLD2
using GaussianFilters

const TIMESERIES_ACTION_TAG = Dict(
    NoTreatment => "",
    MechanicalTreatment => "M",
    ChemicalTreatment => "C",
    ThermalTreatment => "Th",
)

action_short_label(a) = get(TIMESERIES_ACTION_TAG, a, "")


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
# Plot 2: Time series of belief means and variances (Plots.jl version)
# Creates 6 plots:
# 1. Belief means with ribbon and true values for each observation variable (Adult, Motile, Sessile, Temperature) *4 plots*
# 2. Belief variances over time, overlay of all observation variables *1 plot*
# 3. Side-by-side plot of belief means and variances for each observation variable (Adult, Motile, Sessile) *1 plot*
# Inputs:
# - data: DataFrame with simulation data
# - algo_name: String, name of the algorithm
# - config: ExperimentConfig, configuration object
# - lambda: Float, lambda value
# Outputs:
# - Saves plots to config.figures_dir/belief_plots/algo_name/
# - Returns nothing
# ----------------------------
function plot_beliefs_over_time_plotsjl(data, algo_name, config, lambda)

    # Create directory for belief plots
    output_dir = joinpath(config.figures_dir, "belief_plots", algo_name)
    mkpath(output_dir)

    # Filter the data to only include the algorithm and chosen lambda
    data = filter(row -> row.policy == algo_name, data)
    data = filter(row -> row.lambda == lambda, data)

    # Extract first belief history for given solver
    history = data.history[1]

    # Extract beliefs
    beliefs = belief_hist(history)
    belief_means, belief_covariances = unpack(beliefs)

    # Extract belief variances (diagonal of covariance matrices)
    belief_variances = [diag(belief_covariances[i, :, :]) for i in 1:size(belief_covariances, 1)]
    belief_variances_array = hcat(belief_variances...)'

    # Extract states
    states = state_hist(history)
    states_df = DataFrame(
        Adult = [s.Adult for s in states],
        Motile = [s.Motile for s in states],
        Sessile = [s.Sessile for s in states],
        Temperature = [s.Temperature for s in states]
    )
    observations = observation_hist(history)
    observations_df = DataFrame(
        Adult = [o.Adult for o in observations],
        Motile = [o.Motile for o in observations],
        Sessile = [o.Sessile for o in observations],
        Temperature = [o.Temperature for o in observations]
    )
    actions = collect(action_hist(history))
    action_tags = [action_short_label(a) for a in actions]

    labels = ["Adult", "Motile", "Sessile", "Temperature"]
    colors = [:blue, :green, :orange, :purple]

    belief_plots = []

    # Plot 1–3: Adult, Motile, Sessile
    for i in 1:3
        belief_plot = plot(
            belief_means[:, i],
            ribbon=sqrt.(belief_variances_array[:, i]),
            label="Belief mean",
            title="KF Belief Trajectory of $(labels[i]) Abundance (λ = $lambda)",
            xlabel="Timestep (weeks)",
            ylabel="Abundance (average $(labels[i]) units per fish)",
            linewidth=2,
            color=colors[i],
            alpha=0.5,
            legend=:topright
        )
        scatter!(belief_plot, 1:size(states_df,1), states_df[:, i], label="True value", marker=:x, markersize=3, color=colors[i])
        scatter!(belief_plot, 1:size(observations_df,1), observations_df[:, i], label="Observation", marker=:circle, markersize=3, color=colors[i])

        # Add treatment annotations
        for (t, tag) in enumerate(action_tags)
            annotate!(belief_plot, t, 0.0, text(tag, 8, :black))
        end
        
        push!(belief_plots, belief_plot)
        savefig(belief_plots[i], joinpath(output_dir, "belief_means_$(labels[i])_lambda_$(lambda).png"))
    end

    # Plot 4: Temperature (not added to belief_plots)
    p_temp = plot(
        belief_means[:, 4],
        ribbon=sqrt.(belief_variances_array[:, 4]),
        label="Belief mean",
        title="KF Belief Trajectory of Temperature (λ = $lambda)",
        xlabel="Timestep (weeks)",
        ylabel="Temperature (°C)",
        linewidth=2,
        color=colors[4],
        alpha=0.5,
        legend=:topright
    )
    scatter!(p_temp, 1:size(states_df,1), states_df[:, 4], label="True value", marker=:x, markersize=3, color=colors[4])
    scatter!(p_temp, 1:size(observations_df,1), observations_df[:, 4], label="Observation", marker=:circle, markersize=3, color=colors[4])
    savefig(p_temp, joinpath(output_dir, "belief_means_Temperature_lambda_$(lambda).png"))

    # Arrange 3 plots side by side
    belief_plot_grid = plot(belief_plots[1:3]..., layout=(1, 3), size=(2000, 400))
    savefig(belief_plot_grid, joinpath(output_dir, "belief_means_and_variances_split_lambda_$(lambda).png"))

    # Plot variances over time
    variance_plot = plot(title="KF Belief Variance Trajectory (λ = $lambda)", xlabel="Timestep (weeks)", ylabel="Variance")
    for i in 1:length(labels)
        plot!(variance_plot, belief_variances_array[:, i], label="Belief $(labels[i])")
    end
    savefig(variance_plot, joinpath(output_dir, "belief_variances_lambda_$(lambda).png"))
end

# ----------------------------
# Plot 6: Time-series of belief for each policy
# ----------------------------
function plot_policy_belief_levels(histories, title, config, lambda; show_actual_states=true)

    # Get values for first episode of lambda
    histories_lambda = histories[lambda]
    first_episode_history = histories_lambda[1]

    # Extract belief, state, and action histories from the episode
    beliefs = collect(belief_hist(first_episode_history))
    states = collect(state_hist(first_episode_history))
    actions = collect(action_hist(first_episode_history))

    # Convert Gaussian belief to mean and variance
    belief_means = [belief.μ[1] for belief in beliefs]
    belief_stds = [sqrt(belief.Σ[1,1]) for belief in beliefs]
    actual_states = [s.SeaLiceLevel for s in states]

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
    action_labels = [action_short_label(a) for a in actions]
    for (t, label) in enumerate(action_labels)
        display_label = isempty(label) ? "N" : label
        annotate!(p, t, 0.4, text(display_label, 8, :black))
    end

    # Save to both directories for backward compatibility
    mkpath(joinpath(config.figures_dir, "belief_plots", title))
    mkpath(joinpath(config.figures_dir, "research_plots"))
    
    # Save to belief_plots for individual policy analysis
    savefig(p, joinpath(config.figures_dir, "belief_plots/$(title)/beliefs_$(lambda)_lambda.png"))
    
    return p
end

# ----------------------------
# Plot 7: Time-series of sea lice levels for each policy at specific lambda
# ----------------------------
function plot_policy_sealice_levels_over_time(config, lambda_value)
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
            
            # Get histories for this lambda
            histories_lambda = histories[lambda_value]
            
            # Calculate mean and 95% CI for each time step
            time_steps = 1:config.simulation_config.steps_per_episode
            mean_sealice = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract sea lice level at time step t from all episodes
                step_sealice = Float64[]
                for episode_history in histories_lambda
                    states = collect(state_hist(episode_history))
                    if t <= length(states)
                        sealice_level = states[t].SeaLiceLevel
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
    mkpath(joinpath(config.figures_dir, "sealice_time_plots"))
    savefig(p, joinpath(config.figures_dir, "sealice_time_plots/All_policies_sealice_time_lambda_$(lambda_value).png"))
    return p
end

# ----------------------------
# Plot 7b: Time-series of sea lice levels for specific policy only at specific lambda
# ----------------------------
function plot_algo_sealice_levels_over_time(config, algo_name, lambda_value)
    # Initialize the plot
    p = plot(
        title="$algo_name Policy: Sea Lice Levels Over Time (λ = $lambda_value)",
        xlabel="Time Step",
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:bottomright,
        grid=true
    )
    
    policy_name = algo_name
    
    try
        # Load the results from the JLD2 file
        @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
        @load joinpath(config.simulations_dir, "$(policy_name)", "$(policy_name)_histories.jld2") histories
        
        # Get histories for this lambda
        histories_lambda = histories[lambda_value]
        
        # Calculate mean and 95% CI for each time step for all sea lice stages
        time_steps = 1:config.simulation_config.steps_per_episode
        mean_adult = Float64[]
        mean_sessile = Float64[]
        mean_motile = Float64[]
        mean_predicted = Float64[]
        
        ci_lower_adult = Float64[]
        ci_upper_adult = Float64[]
        ci_lower_sessile = Float64[]
        ci_upper_sessile = Float64[]
        ci_lower_motile = Float64[]
        ci_upper_motile = Float64[]
        ci_lower_predicted = Float64[]
        ci_upper_predicted = Float64[]
        
        for t in time_steps
            # Extract sea lice levels at time step t from all episodes
            step_adult = Float64[]
            step_sessile = Float64[]
            step_motile = Float64[]
            step_predicted = Float64[]
            
            for episode_history in histories_lambda
                states = collect(state_hist(episode_history))
                observations = collect(observation_hist(episode_history))
                
                if t <= length(states) && t <= length(observations)
                    push!(step_adult, states[t].Adult)
                    push!(step_sessile, states[t].Sessile)
                    push!(step_motile, states[t].Motile)
                    push!(step_predicted, observations[t].SeaLiceLevel)
                end
            end
            
            # Calculate mean and 95% CI for each stage
            for (step_data, mean_vec, ci_lower_vec, ci_upper_vec) in [
                (step_adult, mean_adult, ci_lower_adult, ci_upper_adult),
                (step_sessile, mean_sessile, ci_lower_sessile, ci_upper_sessile),
                (step_motile, mean_motile, ci_lower_motile, ci_upper_motile),
                (step_predicted, mean_predicted, ci_lower_predicted, ci_upper_predicted)
            ]
                if !isempty(step_data)
                    mean_level = mean(step_data)
                    std_level = std(step_data)
                    n_episodes = length(step_data)
                    se_level = std_level / sqrt(n_episodes)  # Standard error
                    ci_margin = 1.96 * se_level  # 95% CI margin
                    
                    push!(mean_vec, mean_level)
                    push!(ci_lower_vec, mean_level - ci_margin)
                    push!(ci_upper_vec, mean_level + ci_margin)
                else
                    push!(mean_vec, NaN)
                    push!(ci_lower_vec, NaN)
                    push!(ci_upper_vec, NaN)
                end
            end
        end
        
        # Remove NaN values and plot each stage
        stages = [
            ("Adult", mean_adult, ci_lower_adult, ci_upper_adult, :blue),
            ("Sessile", mean_sessile, ci_lower_sessile, ci_upper_sessile, :green),
            ("Motile", mean_motile, ci_lower_motile, ci_upper_motile, :orange),
            ("Predicted", mean_predicted, ci_lower_predicted, ci_upper_predicted, :red)
        ]
        
        for (stage_name, mean_data, ci_lower_data, ci_upper_data, color) in stages
            valid_indices = .!isnan.(mean_data)
            if any(valid_indices)
                valid_time_steps = time_steps[valid_indices]
                valid_mean = mean_data[valid_indices]
                valid_ci_lower = ci_lower_data[valid_indices]
                valid_ci_upper = ci_upper_data[valid_indices]
                
                plot!(
                    p,
                    valid_time_steps,
                    valid_mean,
                    ribbon=(valid_mean .- valid_ci_lower, valid_ci_upper .- valid_mean),
                    label="$stage_name",
                    color=color,
                    linewidth=2,
                    fillalpha=0.3,
                    alpha=0.7
                )
            end
        end
        
    catch e
        @warn "Could not load results for $policy_name: $e"
    end
    
    mkpath(joinpath(config.figures_dir, "sealice_time_plots"))
    savefig(p, joinpath(config.figures_dir, "sealice_time_plots/$(algo_name)_sealice_time_lambda_$(lambda_value).png"))
    return p
end

# ----------------------------
# Plot 7c: Time-series of adult and predicted sea lice levels for NUS_SARSOP policy only at specific lambda
# ----------------------------
function plot_algo_adult_predicted_over_time(config, algo_name, lambda_value)
    # Initialize the plot
    p = plot(
        title="$algo_name Policy: Adult vs Predicted Sea Lice Levels Over Time (λ = $lambda_value)",
        xlabel="Time Step",
        ylabel="Average Sea Lice Level (Avg. Adult Female Lice per Fish)",
        legend=:bottomright,
        grid=true
    )
    
    policy_name = algo_name
    
    try
        # Load the results from the JLD2 file
        @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
        @load joinpath(config.simulations_dir, "$(policy_name)", "$(policy_name)_histories.jld2") histories
        
        # Get histories for this lambda
        histories_lambda = histories[lambda_value]
        
        # Calculate mean and 95% CI for each time step for adult and predicted
        time_steps = 1:config.simulation_config.steps_per_episode
        mean_adult = Float64[]
        mean_predicted = Float64[]
        
        ci_lower_adult = Float64[]
        ci_upper_adult = Float64[]
        ci_lower_predicted = Float64[]
        ci_upper_predicted = Float64[]
        
        for t in time_steps
            # Extract sea lice levels at time step t from all episodes
            step_adult = Float64[]
            step_predicted = Float64[]
            
            for episode_history in histories_lambda
                states = collect(state_hist(episode_history))
                observations = collect(observation_hist(episode_history))
                
                if t <= length(states) && t <= length(observations)
                    push!(step_adult, states[t].Adult)
                    push!(step_predicted, observations[t].SeaLiceLevel)
                end
            end
            
            # Calculate mean and 95% CI for each stage
            for (step_data, mean_vec, ci_lower_vec, ci_upper_vec) in [
                (step_adult, mean_adult, ci_lower_adult, ci_upper_adult),
                (step_predicted, mean_predicted, ci_lower_predicted, ci_upper_predicted)
            ]
                if !isempty(step_data)
                    mean_level = mean(step_data)
                    std_level = std(step_data)
                    n_episodes = length(step_data)
                    se_level = std_level / sqrt(n_episodes)  # Standard error
                    ci_margin = 1.96 * se_level  # 95% CI margin
                    
                    push!(mean_vec, mean_level)
                    push!(ci_lower_vec, mean_level - ci_margin)
                    push!(ci_upper_vec, mean_level + ci_margin)
                else
                    push!(mean_vec, NaN)
                    push!(ci_lower_vec, NaN)
                    push!(ci_upper_vec, NaN)
                end
            end
        end
        
        # Remove NaN values and plot each stage
        stages = [
            ("Adult", mean_adult, ci_lower_adult, ci_upper_adult, :blue),
            ("Predicted", mean_predicted, ci_lower_predicted, ci_upper_predicted, :red)
        ]
        
        for (stage_name, mean_data, ci_lower_data, ci_upper_data, color) in stages
            valid_indices = .!isnan.(mean_data)
            if any(valid_indices)
                valid_time_steps = time_steps[valid_indices]
                valid_mean = mean_data[valid_indices]
                valid_ci_lower = ci_lower_data[valid_indices]
                valid_ci_upper = ci_upper_data[valid_indices]
                
                plot!(
                    p,
                    valid_time_steps,
                    valid_mean,
                    ribbon=(valid_mean .- valid_ci_lower, valid_ci_upper .- valid_mean),
                    label="$stage_name",
                    color=color,
                    linewidth=2,
                    fillalpha=0.3,
                    alpha=0.7
                )
            end
        end
        
    catch e
        @warn "Could not load results for $policy_name: $e"
    end
    
    mkpath(joinpath(config.figures_dir, "sealice_time_plots"))
    savefig(p, joinpath(config.figures_dir, "sealice_time_plots/$(algo_name)_adult_predicted_lambda_$(lambda_value).png"))
    return p
end

# ----------------------------
# Plot 8: Time-series of treatment cost for each policy at specific lambda
# ----------------------------
function plot_policy_treatment_cost_over_time(config, lambda_value)
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
            
            # Get histories for this lambda
            histories_lambda = histories[lambda_value]
            
            # Calculate mean treatment probability and 95% CI for each time step
            time_steps = 1:config.simulation_config.steps_per_episode
            mean_treatment_prob = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract treatment decisions at time step t from all episodes
                step_treatments = Float64[]
                for episode_history in histories_lambda
                    actions = collect(action_hist(episode_history))
                    if t <= length(actions)
                        # Treatment probability: 1 if Treatment, 0 if NoTreatment
                        treatment_indicator = actions[t] == NoTreatment ? 0.0 : 1.0
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
    mkpath(joinpath(config.figures_dir, "treatment_cost_time_plots"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_time_plots/All_policies_treatment_cost_time_lambda_$(lambda_value).png"))
    return p
end

# ----------------------------
# Plot 8b: Time-series of actual treatment cost (probability * cost) for each policy at specific lambda
# ----------------------------
function plot_policy_actual_treatment_cost_over_time(config, lambda_value)
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
            
            # Get histories for this lambda
            histories_lambda = histories[lambda_value]
            
            # Calculate mean treatment cost and 95% CI for each time step
            time_steps = 1:config.simulation_config.steps_per_episode
            mean_treatment_cost = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract treatment costs at time step t from all episodes
                step_costs = Float64[]
                for episode_history in histories_lambda
                    actions = collect(action_hist(episode_history))
                    if t <= length(actions)
                        # Treatment cost: costOfTreatment if Treatment, 0 if NoTreatment
                        treatment_cost = actions[t] == NoTreatment ? 0.0 : get_treatment_cost(actions[t])
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
    mkpath(joinpath(config.figures_dir, "treatment_cost_time_plots"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_time_plots/All_policies_actual_treatment_cost_time_lambda_$(lambda_value).png"))
    return p
end

# ----------------------------
# Plot 8a: Time-series of treatment probability for each policy at specific lambda
# ----------------------------
function plot_policy_treatment_probability_over_time(config, lambda_value)
    # Initialize the plot
    p = plot(
        title="Policy Comparison: Treatment Probability Over Time (λ = $lambda_value)",
        xlabel="Time Step (Weeks)",
        ylabel="Treatment Probability",
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
            # Load the results from the JLD2 file
            @load joinpath(config.results_dir, "$(policy_name)_avg_results.jld2") avg_results
            @load joinpath(config.simulations_dir, "$(policy_name)", "$(policy_name)_histories.jld2") histories
            
            # Get histories for this lambda
            histories_lambda = histories[lambda_value]
            
            # Calculate mean treatment probability and 95% CI for each time step
            time_steps = 1:config.simulation_config.steps_per_episode
            mean_treatment_prob = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract treatment decisions at time step t from all episodes
                step_treatments = Float64[]
                for episode_history in histories_lambda
                    actions = collect(action_hist(episode_history))
                    if t <= length(actions)
                        # Treatment probability: 1 if Treatment, 0 if NoTreatment
                        treatment_indicator = actions[t] == NoTreatment ? 0.0 : 1.0
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
    mkpath(joinpath(config.figures_dir, "treatment_cost_time_plots"))
    savefig(p, joinpath(config.figures_dir, "treatment_cost_time_plots/All_policies_treatment_cost_time_lambda_$(lambda_value).png"))
    return p
end
