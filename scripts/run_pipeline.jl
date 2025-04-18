include("../src/cleaning.jl")
include("../src/optimize.jl")
include("../src/plot_views.jl")

using .Cleaning, .PlotViews, Plots #.Optimize,
ENV["PLOTS_BROWSER"] = "true"

# Clean data
df = Cleaning.load_and_clean("data/raw/licedata.csv")

# Plot sealice levels over time
sealice_levels_over_time_plot = PlotViews.plot_sealice_levels_over_time(df)
Plots.savefig(sealice_levels_over_time_plot, "results/figures/sealice_levels_over_time.png")