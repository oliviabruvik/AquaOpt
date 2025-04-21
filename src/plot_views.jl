module PlotViews

using Plots
plotlyjs()  # Set the backend to PlotlyJS

# Plot sealice levels over time, colored by location number
function plot_sealice_levels_over_time(df)
    plot(
        df.total_week, 
        df.adult_sealice, 
        title="Sealice levels over time", 
        xlabel="Total weeks from start (2012)", 
        ylabel="Sealice levels",
        color=df.location_number
    )
end

# Plot mdp results across lambda values
function plot_mdp_results(results)
    p = scatter(
        results.avg_treatment_cost,
        results.avg_sealice,
        marker_z=results.lambda,
        title="Treatment Cost vs Sea Lice Levels",
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

end