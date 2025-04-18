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

end