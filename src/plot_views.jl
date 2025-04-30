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
    
    # Define colors and markers for each policy
    policy_styles = Dict(
        "Heuristic Policy" => (:circle, :blue),
        "VI Policy" => (:square, :red),
        "SARSOP Policy" => (:diamond, :green),
        "QMDP Policy" => (:dtriangle, :purple)
    )
    
    # Load and plot each policy's results
    for (policy_name, (marker, color)) in policy_styles
        try
            # Load the results from the JLD2 file
            @load "results/data/$(policy_name)_$(num_episodes)_$(steps_per_episode).jld2" results
            
            # Add the scatter plot to the main plot
            scatter!(
                p,
                results.avg_treatment_cost,
                results.avg_sealice,
                marker_z=results.lambda,
                marker=marker,
                markersize=6,
                label=policy_name,
                colorbar=false,
                c=:viridis
            )
        catch e
            println("Warning: Could not load results for $policy_name: $e")
        end
    end
    
    # Add a single colorbar for all plots
    savefig(p, "results/figures/all_policies_overlay_$(num_episodes)_$(steps_per_episode).png")
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
        "Heuristic Policy" => (color=:blue, marker=:circle),
        "VI Policy" => (color=:red, marker=:square),
        "SARSOP Policy" => (color=:green, marker=:diamond),
        "QMDP Policy" => (color=:purple, marker=:dtriangle)
    )
    
    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try
            # Load the results from the JLD2 file
            @load "results/data/$(policy_name)_$(num_episodes)_$(steps_per_episode).jld2" results
            
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
            println("Warning: Could not load results for $policy_name: $e")
        end
    end
    savefig(p, "results/figures/policy_sealice_comparison_$(num_episodes)_$(steps_per_episode).png")
    return p
end