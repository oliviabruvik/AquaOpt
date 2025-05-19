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
function plot_mdp_results(results, title)
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
    return p
end

# ----------------------------
# Plot 3: Overlay all policies
# ----------------------------
function plot_mdp_results_overlay(num_episodes, steps_per_episode)
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
            @load "results/data/$(policy_name)/results_$(num_episodes)_episodes_$(steps_per_episode)_steps.jld2" results
            
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
    fig_path = joinpath("results", "figures", "all_policies_overlay_$(num_episodes)_episodes_$(steps_per_episode)_steps.png")
    @info "Saving figure to $fig_path"
    savefig(p, fig_path)
    return p
end

# ----------------------------
# Plot 4: Time-series of sea lice for each policy
# ----------------------------
function plot_policy_sealice_levels(num_episodes, steps_per_episode)
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
            @load "results/data/$(policy_name)/results_$(num_episodes)_episodes_$(steps_per_episode)_steps.jld2" results
            
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
    savefig(p, "results/figures/policy_sealice_comparison_$(num_episodes)_$(steps_per_episode).png")
    return p
end

# ----------------------------
# Plot 5: Time-series of belief for each policy
# ----------------------------
function plot_policy_belief_levels(results, title)
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

    savefig(p, "results/figures/$(title)/policy_belief_comparison.png")
    return p
end