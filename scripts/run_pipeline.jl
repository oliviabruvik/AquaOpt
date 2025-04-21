include("../src/cleaning.jl")
include("../src/optimize_mdp.jl")
include("../src/optimize_sarsop.jl")
include("../src/plot_views.jl")

using .Cleaning, .PlotViews, .OptimizeMDP, Plots
ENV["PLOTS_BROWSER"] = "true"

# Clean data
# Data downloaded from https://lusedata.hubocean.earth/
# Data limited to production area 5: Stadt til Hustadvika for 2012-2025
df = Cleaning.load_and_clean("data/raw/licedata.csv")

# Plot sealice levels over time
sealice_levels_over_time_plot = PlotViews.plot_sealice_levels_over_time(df)
Plots.savefig(sealice_levels_over_time_plot, "results/figures/sealice_levels_over_time.png")

# Run optimizations
#sarsop_policy = Optimize.sarsop_optimize(df)
mdp_policy = OptimizeMDP.mdp_optimize(df)