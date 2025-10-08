using PGFPlotsX
using PGFPlotsX: Options
using Statistics
using JLD2
using POMDPs
using POMDPTools

# Include necessary files for types and functions
include("../Utils/SharedTypes.jl")
include("../Utils/Utils.jl")
include("../Models/SeaLiceLogPOMDP.jl")
include("../Models/SeaLicePOMDP.jl")
include("../Algorithms/Policies.jl")

# Preamble: enable fillbetween and modern pgfplots behavior
PGFPlotsX.DEFAULT_PREAMBLE = [
    raw"\usepackage{pgfplots}",
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usepgfplotslibrary{groupplots}",
    raw"\usetikzlibrary{intersections}",
    raw"\pgfplotsset{compat=newest}",
    raw"\pgfplotsset{legend style={text=white,fill=none,draw=none}}"
]

# Transparent axis & legend bundle (string keys for special pgf keys)
const AXIS_TRANSPARENT = (
    "axis background/.style" => Options("fill" => "none"),                 # transparent axis rect
    "legend style"           => Options("fill" => "none", "draw" => "none", "text" => "white"),# transparent legend box with white text
    "axis on top"            => true
)

# Save a transparent PNG via poppler (pdftocairo -transp). Requires: brew install poppler
function save_transparent_png(pdf_path::AbstractString, ax::Axis; dpi::Int=300)
    PGFPlotsX.save(pdf_path, ax)  # writes the PDF first
    stem = replace(pdf_path, r"\.pdf$" => "")
    run(`pdftocairo -png -transp -r $dpi $pdf_path $stem`)
    return stem * ".png"
end

# Save a transparent PNG for GroupPlot objects
function save_transparent_png(pdf_path::AbstractString, ax::GroupPlot; dpi::Int=300)
    PGFPlotsX.save(pdf_path, ax)  # writes the PDF first
    stem = replace(pdf_path, r"\.pdf$" => "")
    run(`pdftocairo -png -transp -r $dpi $pdf_path $stem`)
    return stem * ".png"
end

# ----------------------------
# Plot 7: Time-series of sea lice levels heuristic versus sarsop
# ----------------------------
function plot_heuristic_vs_sarsop_sealice_levels_over_time(parallel_data, config)
    # Initialize the plot
    p = plot(
        title="Population Dynamics Over Time (North Sea)",
        xlabel="Time Since Production Start (Weeks)",
        ylabel="Avg. Adult Female Sea Lice per Fish",
        legend=:right,
        grid=true,
        ylims=(0, 0.6),
        xlims=(0, config.steps_per_episode),
        size=(1200, 400)  # Wider plot: 1200px wide, 400px tall
    )
    
    # Define colors and markers for each policy
    policy_styles = Dict(
        "Heuristic_Policy" => (color=:blue, marker=:circle),
        "NUS_SARSOP_Policy" => (color=:red, marker=:square),
    )

    # Load and plot results for each policy
    for (policy_name, style) in policy_styles
        try

            # Filter data for this policy
            data_filtered = filter(row -> row.policy == policy_name, parallel_data)
            
            # Get all unique seeds
            seeds = unique(data_filtered.seed)

            # Calculate mean and 95% CI for each time step
            time_steps = 1:config.steps_per_episode
            mean_sealice = Float64[]
            ci_lower = Float64[]
            ci_upper = Float64[]
            
            for t in time_steps
                # Extract sea lice level at time step t from all episodes
                step_sealice = Float64[]
                for seed in seeds
                    data_seed = filter(row -> row.seed == seed, data_filtered)
                    h = data_seed.history[1]
                    states = collect(state_hist(h))
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
    
    # Add regulatory limit line
    hline!([0.5], linestyle=:dash, color=:black, label="Reg. Limit", linewidth=2)
    
    # mkpath(joinpath(config.figures_dir, "research_plots/sealice_time_plots"))
    savefig(p, "debug_plots/heuristic_vs_sarsop_sealice_time.png")
    return p
end

# ----------------------------
# Plot 7b: Bar chart comparing treatment distributions between Heuristic and SARSOP
# ----------------------------
function plot_treatment_distribution_comparison(parallel_data, config)
    # Initialize the plot
    p = bar(
        title="Treatment Distribution Comparison: Heuristic vs SARSOP",
        xlabel="Treatment Type",
        ylabel="Average Number of Treatments per Episode",
        legend=:topright,
        grid=true,
        ylims=(0, :auto)  # Let y-axis auto-scale based on data
    )
    
    # Define treatment types and their display names
    treatment_types = ["NoTreatment", "Treatment", "ThermalTreatment"]
    treatment_labels = ["None", "Chemical", "Thermal"]
    
    # Define colors for each policy
    policy_colors = Dict(
        "Heuristic_Policy" => :blue,
        "NUS_SARSOP_Policy" => :red
    )
    
    # Calculate treatment counts for each policy
    treatment_data = Dict{String, Dict{String, Float64}}()
    
    for policy_name in ["Heuristic_Policy", "NUS_SARSOP_Policy"]
        # Filter data for this policy
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        
        # Initialize treatment counts for this policy
        treatment_data[policy_name] = Dict{String, Float64}()
        
        # Get all unique seeds
        seeds = unique(data_filtered.seed)
        
        # Collect treatment counts across all seeds
        treatment_counts = Dict{String, Vector{Int}}(
            "NoTreatment" => [],
            "Treatment" => [],
            "ThermalTreatment" => []
        )
        
        for seed in seeds
            data_seed = filter(row -> row.seed == seed, data_filtered)
            h = data_seed.history[1]
            actions = collect(action_hist(h))
            
            # Count treatments for this seed
            seed_treatments = Dict{String, Int}(
                "NoTreatment" => 0,
                "Treatment" => 0,
                "ThermalTreatment" => 0
            )
            
            for action in actions
                if action == NoTreatment
                    seed_treatments["NoTreatment"] += 1
                elseif action == Treatment
                    seed_treatments["Treatment"] += 1
                elseif action == ThermalTreatment
                    seed_treatments["ThermalTreatment"] += 1
                end
            end
            
            # Add to treatment counts
            for treatment_type in keys(treatment_counts)
                push!(treatment_counts[treatment_type], seed_treatments[treatment_type])
            end
        end
        
        # Calculate mean treatment counts for this policy
        for treatment_type in treatment_types
            treatment_data[policy_name][treatment_type] = mean(treatment_counts[treatment_type])
        end
    end
    
    # Create bar chart data
    x_positions = 1:length(treatment_types)
    bar_width = 0.35
    
    # Plot bars for each treatment type
    for (i, treatment_type) in enumerate(treatment_types)
        # Heuristic bar
        heuristic_value = treatment_data["Heuristic_Policy"][treatment_type]
        bar!(p, [i - bar_width/2], [heuristic_value], 
             label=i == 1 ? "Heuristic" : "", 
             color=policy_colors["Heuristic_Policy"], 
             alpha=0.7,
             bar_width=bar_width)
        
        # SARSOP bar
        sarsop_value = treatment_data["NUS_SARSOP_Policy"][treatment_type]
        bar!(p, [i + bar_width/2], [sarsop_value], 
             label=i == 1 ? "SARSOP" : "", 
             color=policy_colors["NUS_SARSOP_Policy"], 
             alpha=0.7,
             bar_width=bar_width)
    end
    
    # Set x-axis ticks
    xticks!(p, x_positions, treatment_labels)
    
    # mkpath(joinpath(config.figures_dir, "research_plots/treatment_plots"))
    # savefig(p, joinpath(config.figures_dir, "research_plots/treatment_plots/treatment_distribution_comparison.png"))
    savefig(p, "debug_plots/treatment_distribution_comparison.png")
    return p
end

using PGFPlotsX, Statistics

# function plot_heuristic_vs_sarsop_sealice_levels_over_time_latex(parallel_data, config)
#     ax = @pgf Axis(
#         Options(
#             :title => "Population Dynamics Over Time (North Sea)",
#             :xlabel => "Time Since Production Start (Weeks)",
#             :ylabel => "Avg. Adult Female Sea Lice per Fish",
#             :legend_pos => "north east",
#             :xmin => 0, :xmax => config.steps_per_episode,
#             :ymin => 0, :ymax => 0.6,
#             :grid => "both",
#             :width => "12cm",
#             :height => "6cm",
#         )
#     )

#     policy_styles = Dict(
#         "Heuristic_Policy" => "blue",
#         "NUS_SARSOP_Policy" => "red",
#     )

#     for (policy_name, color) in policy_styles
#         try
#             data_filtered = filter(row -> row.policy == policy_name, parallel_data)
#             seeds = unique(data_filtered.seed)
#             time_steps = 1:config.steps_per_episode

#             mean_sealice, ci_lower, ci_upper = Float64[], Float64[], Float64[]

#             for t in time_steps
#                 vals = Float64[]
#                 for seed in seeds
#                     data_seed = filter(row -> row.seed == seed, data_filtered)
#                     h = data_seed.history[1]
#                     states = collect(state_hist(h))
#                     if t <= length(states)
#                         push!(vals, states[t].SeaLiceLevel)
#                     end
#                 end
#                 if !isempty(vals)
#                     m = mean(vals); s = std(vals)
#                     se = s / sqrt(length(vals))
#                     ci = 1.96 * se
#                     push!(mean_sealice, m)
#                     push!(ci_lower, m - ci)
#                     push!(ci_upper, m + ci)
#                 else
#                     push!(mean_sealice, NaN)
#                     push!(ci_lower, NaN)
#                     push!(ci_upper, NaN)
#                 end
#             end

#             coords_mean  = collect(zip(time_steps, mean_sealice))
#             coords_lower = collect(zip(time_steps, ci_lower))
#             coords_upper = collect(zip(time_steps, ci_upper))

#             # Mean line
#             push!(ax, Plot(Options(:color => color, :thick => nothing, :mark => "none"), Coordinates(coords_mean)))

#             # Upper/lower paths
#             push!(ax, Plot(Options(:name_path => "upper_$policy_name", :draw => "none"), Coordinates(coords_upper)))
#             push!(ax, Plot(Options(:name_path => "lower_$policy_name", :draw => "none"), Coordinates(coords_lower)))

#             # Ribbon fill
#             push!(ax, @pgf("\\addplot[$(color)!30] fill between[of=upper_$policy_name and lower_$policy_name];"))

#         catch e
#             @warn "Could not plot $policy_name: $e"
#         end
#     end

#     # Regulatory line
#     push!(ax, @pgf("\\addplot[black, dashed, thick] coordinates {(1,0.5) ($(config.steps_per_episode),0.5)};"))
#     push!(ax, Legend(["Heuristic", "SARSOP", "Reg. Limit"]))

#     pgfsave("debug_plots/heuristic_vs_sarsop_sealice_time_latex.pdf", ax)
#     return ax
# end

# function plot_treatment_distribution_comparison_latex(parallel_data, config)
#     treatment_types  = ["NoTreatment", "Treatment", "ThermalTreatment"]
#     treatment_labels = ["None", "Chemical", "Thermal"]

#     policy_colors = Dict(
#         "Heuristic_Policy" => "blue",
#         "NUS_SARSOP_Policy" => "red"
#     )

#     treatment_data = Dict{String, Vector{Float64}}()
#     for policy_name in ["Heuristic_Policy", "NUS_SARSOP_Policy"]
#         data_filtered = filter(row -> row.policy == policy_name, parallel_data)
#         seeds = unique(data_filtered.seed)

#         counts = Dict(t => Float64[] for t in treatment_types)
#         for seed in seeds
#             data_seed = filter(row -> row.seed == seed, data_filtered)
#             h = data_seed.history[1]
#             actions = collect(action_hist(h))
#             c = Dict(t => 0 for t in treatment_types)
#             for a in actions
#                 if a == NoTreatment
#                     c["NoTreatment"] += 1
#                 elseif a == Treatment
#                     c["Treatment"] += 1
#                 elseif a == ThermalTreatment
#                     c["ThermalTreatment"] += 1
#                 end
#             end
#             for t in treatment_types
#                 push!(counts[t], c[t])
#             end
#         end
#         treatment_data[policy_name] = [mean(counts[t]) for t in treatment_types]
#     end

#     ax = @pgf Axis(
#         Options(
#             :title => "Treatment Distribution: Heuristic vs SARSOP",
#             :ybar => nothing,
#             :bar_width => "10pt",
#             :enlarge_x_limits => "0.2",
#             :xlabel => "Treatment Type",
#             :ylabel => "Avg. Number of Treatments",
#             :xtick => "data",
#             :xticklabels => "{" * join(treatment_labels, ",") * "}",
#             :legend_pos => "north east",
#             :grid => "both",
#         )
#     )

#     push!(ax, Plot(Options(:fill => policy_colors["Heuristic_Policy"]), Coordinates(enumerate(treatment_data["Heuristic_Policy"]))))
#     push!(ax, Plot(Options(:fill => policy_colors["NUS_SARSOP_Policy"]), Coordinates(enumerate(treatment_data["NUS_SARSOP_Policy"]))))
#     push!(ax, Legend(["Heuristic", "SARSOP"]))

#     pgfsave("debug_plots/treatment_distribution_comparison_latex.pdf", ax)
#     return ax
# end
############ Plot 1: Time-series (Heuristic vs SARSOP) with 95% CI, transparent ############

function plot_heuristic_vs_sarsop_sealice_levels_over_time_latex(parallel_data, config)
    ax = @pgf Axis(Options(
        :title => "Population Dynamics Over Time (West Region)",
        :xlabel => "Time Since Production Start (Weeks)",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :legend_pos => "north east",
        :xmin => 0, :xmax => config.steps_per_episode,
        :ymin => 0, :ymax => 0.6,
        :grid => "both",
        :width => "12cm",
        :height => "6cm",
        :title_style => "color=white",
        :xlabel_style => "color=white",
        :ylabel_style => "color=white",
        :tick_label_style => "color=white",
        :grid_style => "color=gray!30",
        "legend style" => "text=white,fill=none,draw=none",
        AXIS_TRANSPARENT...,
    ))

    policy_styles = Dict(
        "Heuristic_Policy" => "blue",
        "NUS_SARSOP_Policy" => "red",
    )

    # Keep this to match legend order later
    plotted_order = String[]

    for (policy_name, color) in policy_styles
        try
            data_filtered = filter(row -> row.policy == policy_name, parallel_data)
            seeds = unique(data_filtered.seed)
            time_steps = 1:config.steps_per_episode

            mean_sealice, ci_lower, ci_upper = Float64[], Float64[], Float64[]
            for t in time_steps
                vals = Float64[]
                for seed in seeds
                    data_seed = filter(row -> row.seed == seed, data_filtered)
                    h = data_seed.history[1]
                    states = collect(state_hist(h))
                    if t <= length(states)
                        push!(vals, states[t].SeaLiceLevel)
                    end
                end
                if !isempty(vals)
                    m = mean(vals); s = std(vals)
                    se = s / sqrt(length(vals))
                    ci = 1.96 * se
                    push!(mean_sealice, m)
                    push!(ci_lower, m - ci)
                    push!(ci_upper, m + ci)
                else
                    push!(mean_sealice, NaN)
                    push!(ci_lower, NaN)
                    push!(ci_upper, NaN)
                end
            end

            coords_mean  = collect(zip(time_steps, mean_sealice))
            coords_lower = collect(zip(time_steps, ci_lower))
            coords_upper = collect(zip(time_steps, ci_upper))

            # Mean line (note flag-style :thick must be written as => nothing)
            push!(ax, Plot(Options(:color => color, :thick => nothing, :mark => "none"),
                           Coordinates(coords_mean)))

            # Confidence band via fillbetween
            push!(ax, Plot(Options(:name_path => "upper_$policy_name", :draw => "none"),
                           Coordinates(coords_upper)))
            push!(ax, Plot(Options(:name_path => "lower_$policy_name", :draw => "none"),
                           Coordinates(coords_lower)))
            push!(ax, @pgf("\\addplot[$(color)!30] fill between[of=upper_$policy_name and lower_$policy_name];"))

            push!(plotted_order, policy_name)
        catch e
            @warn "Could not plot $policy_name: $e"
        end
    end

    # Regulatory line
    push!(ax, @pgf("\\addplot[white, dashed, thick] coordinates {(1,0.5) ($(config.steps_per_episode),0.5)};"))

    # Legend with correct colors (order: the two policies actually plotted, then Reg. Limit)
    # Add Heuristic (blue)
    push!(ax, @pgf("\\addlegendimage{blue, thick}"))
    push!(ax, @pgf("\\addlegendentry{Heuristic}"))
    # Add SARSOP (red)
    push!(ax, @pgf("\\addlegendimage{red, thick}"))
    push!(ax, @pgf("\\addlegendentry{SARSOP}"))
    # Add regulatory limit (white dashed)
    push!(ax, @pgf("\\addlegendimage{white, dashed, thick}"))
    push!(ax, @pgf("\\addlegendentry{Reg. Limit}"))

    PGFPlotsX.save("debug_plots/heuristic_vs_sarsop_sealice_time_latex.pdf", ax)
    save_transparent_png("debug_plots/heuristic_vs_sarsop_sealice_time_latex.pdf", ax)
    return ax
end


############ Plot 2: Treatment distribution (grouped bars), transparent ############

function plot_treatment_distribution_comparison_latex(parallel_data, config)
    treatment_types  = ["NoTreatment", "Treatment", "ThermalTreatment"]
    treatment_labels = ["None", "Chemical", "Thermal"]

    policy_colors = Dict(
        "Heuristic_Policy" => "blue",
        "NUS_SARSOP_Policy" => "green"
    )

    # Compute averages per policy
    treatment_data = Dict{String, Vector{Float64}}()
    for policy_name in ["Heuristic_Policy", "NUS_SARSOP_Policy"]
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        seeds = unique(data_filtered.seed)

        counts = Dict(t => Float64[] for t in treatment_types)
        for seed in seeds
            data_seed = filter(row -> row.seed == seed, data_filtered)
            h = data_seed.history[1]
            actions = collect(action_hist(h))
            c = Dict(t => 0 for t in treatment_types)
            for a in actions
                if a == NoTreatment
                    c["NoTreatment"] += 1
                elseif a == Treatment
                    c["Treatment"] += 1
                elseif a == ThermalTreatment
                    c["ThermalTreatment"] += 1
                end
            end
            for t in treatment_types
                push!(counts[t], c[t])
            end
        end
        treatment_data[policy_name] = [mean(counts[t]) for t in treatment_types]
    end

    ax = @pgf Axis(Options(
        :title => "Treatment Distribution: Heuristic vs SARSOP (West Region)",
        :ybar => nothing,                          # flag option must be Pair
        :bar_width => "10pt",
        :enlarge_x_limits => "0.2",
        :xlabel => "Treatment Type",
        :ylabel => "Avg. Number of Treatments",
        :xtick => "data",
        :xticklabels => "{" * join(treatment_labels, ",") * "}",
        :legend_pos => "north east",
        :grid => "both",
        :width => "12cm",
        :height => "6cm",
        :title_style => "color=white",
        :xlabel_style => "color=white",
        :ylabel_style => "color=white",
        :tick_label_style => "color=white",
        :grid_style => "color=gray!30",
        "legend style" => "text=white,fill=none,draw=none",
        AXIS_TRANSPARENT...,
    ))

    # PGFPlots groups bars automatically when multiple ybar series share the same x coords
    push!(ax, Plot(Options(:fill => policy_colors["Heuristic_Policy"]),
                   Coordinates(enumerate(treatment_data["Heuristic_Policy"]))))
    push!(ax, Plot(Options(:fill => policy_colors["NUS_SARSOP_Policy"]),
                   Coordinates(enumerate(treatment_data["NUS_SARSOP_Policy"]))))
    push!(ax, Legend(["Heuristic", "SARSOP"]))

    PGFPlotsX.save("debug_plots/treatment_distribution_comparison_latex.pdf", ax)
    save_transparent_png("debug_plots/treatment_distribution_comparison_latex.pdf", ax)
    return ax
end

# ----------------------------
# Policy Action Heatmap: Shows which actions SARSOP policy chooses
# based on sea temperature and current sea lice levels
# ----------------------------
function plot_sarsop_policy_action_heatmap(config, λ=0.6)
    
    # Load the SARSOP policy
    policy_path = joinpath(config.policies_dir, "NUS_SARSOP_Policy", "policy_pomdp_mdp_$(λ)_lambda.jld2")
    if !isfile(policy_path)
        error("Policy file not found: $policy_path")
    end
    
    @load policy_path policy pomdp mdp
    
    # Define temperature and sea lice level ranges
    temp_range = 8.0:0.5:24.0  # Sea temperature range (°C)
    sealice_range = 0.0:0.01:1.0  # Sea lice level range
    
    # Fixed values for sessile and motile lice (as requested)
    fixed_sessile = 0.25  # Constant sessile level
    fixed_motile = 0.25   # Constant motile level
    
    # Action colors
    action_colors = Dict(
        NoTreatment => "blue",
        Treatment => "green", 
        ThermalTreatment => "red"
    )
    
    # Create separate coordinate lists for each action
    no_treatment_coords = []
    treatment_coords = []
    thermal_coords = []
    
    for (i, sealice_level) in enumerate(sealice_range)
        for (j, temp) in enumerate(temp_range)
            # Predict next sea lice level using the current state
            pred_adult, pred_motile, pred_sessile = predict_next_abundances(
                sealice_level, fixed_motile, fixed_sessile, temp
            )
            
            # Create a belief state centered on the predicted level
            # We'll use a simple discretized belief for the policy
            if pomdp isa SeaLiceLogPOMDP
                pred_adult = log(max(pred_adult, 1e-6))
                # Create a belief vector for log space
                state_space = states(pomdp)
                belief = zeros(length(state_space))
                # Find closest state and set belief
                distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
                closest_idx = argmin(distances)
                belief[closest_idx] = 1.0
            else
                # Create a belief vector for raw space
                state_space = states(pomdp)
                belief = zeros(length(state_space))
                # Find closest state and set belief
                distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
                closest_idx = argmin(distances)
                belief[closest_idx] = 1.0
            end
            
            # Get action from policy
            try
                chosen_action = action(policy, belief)
                coord = (temp, sealice_level)
                
                if chosen_action == NoTreatment
                    push!(no_treatment_coords, coord)
                elseif chosen_action == Treatment
                    push!(treatment_coords, coord)
                elseif chosen_action == ThermalTreatment
                    push!(thermal_coords, coord)
                end
            catch e
                @warn "Could not get action for temp=$temp, sealice=$sealice_level: $e"
                push!(no_treatment_coords, (temp, sealice_level))  # Default to no treatment
            end
        end
    end
    
    # Create the plot
    ax = @pgf Axis(Options(
        :title => "Dominant policy action: SARSOP",
        :xlabel => "Sea Temperature (°C)",
        :ylabel => "Average Adult Female Sea Lice per Fish",
        :width => "12cm",
        :height => "8cm",
        :title_style => "color=black",
        :xlabel_style => "color=black",
        :ylabel_style => "color=black",
        :tick_label_style => "color=black",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => Options("fill" => "white", "draw" => "black", "text" => "black"),
    ))
    
    # Plot each action type as a scatter plot
    if !isempty(no_treatment_coords)
        push!(ax, Plot(Options(:color => "blue", :mark => "square", :mark_size => "1pt", :only_marks => nothing),
                       Coordinates(no_treatment_coords)))
    end
    
    if !isempty(treatment_coords)
        push!(ax, Plot(Options(:color => "red", :mark => "square", :mark_size => "1pt", :only_marks => nothing),
                       Coordinates(treatment_coords)))
    end
    
    if !isempty(thermal_coords)
        push!(ax, Plot(Options(:color => "green", :mark => "square", :mark_size => "1pt", :only_marks => nothing),
                       Coordinates(thermal_coords)))
    end
    
    # Add legend with correct colors
    push!(ax, @pgf("\\addlegendimage{blue, mark=square, mark size=3pt}"))
    push!(ax, @pgf("\\addlegendentry{No Treatment}"))
    push!(ax, @pgf("\\addlegendimage{green, mark=square, mark size=3pt}"))
    push!(ax, @pgf("\\addlegendentry{Chemical Treatment}"))
    push!(ax, @pgf("\\addlegendimage{red, mark=square, mark size=3pt}"))
    push!(ax, @pgf("\\addlegendentry{Thermal Treatment}"))
    
    PGFPlotsX.save("debug_plots/sarsop_policy_action_heatmap_latex.pdf", ax)
    save_transparent_png("debug_plots/sarsop_policy_action_heatmap_latex.pdf", ax)
    return ax
end

# ----------------------------
# Policy Action Heatmap: Shows which actions Heuristic policy chooses
# based on sea temperature and current sea lice levels
# ----------------------------
function plot_heuristic_policy_action_heatmap(config, λ=0.6)
    
    # Create heuristic policy
    heuristic_config = HeuristicConfig(
        raw_space_threshold=config.heuristic_threshold,
        belief_threshold_mechanical=config.heuristic_belief_threshold_mechanical,
        belief_threshold_thermal=config.heuristic_belief_threshold_thermal,
        rho=config.heuristic_rho
    )
    
    # Create POMDP for the heuristic policy
    if config.log_space
        pomdp = SeaLiceLogPOMDP(
            lambda=λ,
            reward_lambdas=config.reward_lambdas,
            costOfTreatment=config.costOfTreatment,
            growthRate=config.growthRate,
            rho=config.rho,
            discount_factor=config.discount_factor,
            discretization_step=config.discretization_step,
            adult_sd=abs(log(config.raw_space_sampling_sd)),
            regulation_limit=config.regulation_limit,
            full_observability_solver=config.full_observability_solver,
        )
    else
        pomdp = SeaLicePOMDP(
            lambda=λ,
            reward_lambdas=config.reward_lambdas,
            costOfTreatment=config.costOfTreatment,
            growthRate=config.growthRate,
            rho=config.rho,
            discount_factor=config.discount_factor,
            discretization_step=config.discretization_step,
            adult_sd=config.raw_space_sampling_sd,
            regulation_limit=config.regulation_limit,
            full_observability_solver=config.full_observability_solver,
        )
    end
    
    policy = HeuristicPolicy(pomdp, heuristic_config)
    
    # Define temperature and sea lice level ranges
    temp_range = 8.0:0.5:24.0  # Sea temperature range (°C)
    sealice_range = 0.0:0.01:1.0  # Sea lice level range
    
    # Fixed values for sessile and motile lice (as requested)
    fixed_sessile = 0.25  # Constant sessile level
    fixed_motile = 0.25   # Constant motile level
    
    # Action colors
    action_colors = Dict(
        NoTreatment => "blue",
        Treatment => "green", 
        ThermalTreatment => "red"
    )
    
    # Create separate coordinate lists for each action
    no_treatment_coords = []
    treatment_coords = []
    thermal_coords = []
    
    for (i, sealice_level) in enumerate(sealice_range)
        for (j, temp) in enumerate(temp_range)
            # Predict next sea lice level using the current state
            pred_adult, pred_motile, pred_sessile = predict_next_abundances(
                sealice_level, fixed_motile, fixed_sessile, temp
            )
            
            # Create a belief state centered on the predicted level
            # We'll use a simple discretized belief for the policy
            if pomdp isa SeaLiceLogPOMDP
                pred_adult = log(max(pred_adult, 1e-6))
                # Create a belief vector for log space
                state_space = states(pomdp)
                belief = zeros(length(state_space))
                # Find closest state and set belief
                distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
                closest_idx = argmin(distances)
                belief[closest_idx] = 1.0
            else
                # Create a belief vector for raw space
                state_space = states(pomdp)
                belief = zeros(length(state_space))
                # Find closest state and set belief
                distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
                closest_idx = argmin(distances)
                belief[closest_idx] = 1.0
            end
            
            # Get action from policy
            try
                chosen_action = action(policy, belief)
                coord = (temp, sealice_level)
                
                if chosen_action == NoTreatment
                    push!(no_treatment_coords, coord)
                elseif chosen_action == Treatment
                    push!(treatment_coords, coord)
                elseif chosen_action == ThermalTreatment
                    push!(thermal_coords, coord)
                end
            catch e
                @warn "Could not get action for temp=$temp, sealice=$sealice_level: $e"
                push!(no_treatment_coords, (temp, sealice_level))  # Default to no treatment
            end
        end
    end
    
    # Create the plot
    ax = @pgf Axis(Options(
        :title => "Dominant policy action: Heuristic",
        :xlabel => "Sea Temperature (°C)",
        :ylabel => "Average Adult Female Sea Lice per Fish",
        :width => "12cm",
        :height => "8cm",
        :title_style => "color=black",
        :xlabel_style => "color=black",
        :ylabel_style => "color=black",
        :tick_label_style => "color=black",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => Options("fill" => "white", "draw" => "black", "text" => "black"),
    ))
    
    # Plot each action type as a scatter plot
    if !isempty(no_treatment_coords)
        push!(ax, Plot(Options(:color => "blue", :mark => "square", :mark_size => "1pt", :only_marks => nothing),
                       Coordinates(no_treatment_coords)))
    end
    
    if !isempty(treatment_coords)
        push!(ax, Plot(Options(:color => "red", :mark => "square", :mark_size => "1pt", :only_marks => nothing),
                       Coordinates(treatment_coords)))
    end
    
    if !isempty(thermal_coords)
        push!(ax, Plot(Options(:color => "red", :mark => "square", :mark_size => "1pt", :only_marks => nothing),
                       Coordinates(thermal_coords)))
    end
    
    # Add legend with correct colors
    push!(ax, @pgf("\\addlegendimage{blue, mark=square, mark size=3pt}"))
    push!(ax, @pgf("\\addlegendentry{No Treatment}"))
    push!(ax, @pgf("\\addlegendimage{red, mark=square, mark size=3pt}"))
    push!(ax, @pgf("\\addlegendentry{Chemical Treatment}"))
    push!(ax, @pgf("\\addlegendentry{Thermal Treatment}"))
    
    PGFPlotsX.save("debug_plots/heuristic_policy_action_heatmap_latex.pdf", ax)
    save_transparent_png("debug_plots/heuristic_policy_action_heatmap_latex.pdf", ax)
    return ax
end

# ----------------------------
# Combined Policy Action Heatmap: Shows both SARSOP and Heuristic policies
# side by side for comparison using groupplots with shared axes
# ----------------------------
function plot_combined_policy_action_heatmaps(config, λ=0.6)
    
    # Generate data for both policies
    sarsop_data = generate_policy_action_data("SARSOP", config, λ)
    heuristic_data = generate_policy_action_data("Heuristic", config, λ)
    
    # Create the combined plot using groupplots with shared axes
    ax = @pgf GroupPlot(Options(
        :group_style => "{group size=2 by 1, horizontal sep=0.5cm}",
        :width => "12cm",
        :height => "8cm",
        :title => "Dominant policy actions: SARSOP and Heuristic",
        :title_style => "color=black",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => Options("fill" => "white", "draw" => "black", "text" => "black"),
    ))
    
    # SARSOP subplot (left) - only show ylabel on left plot
    push!(ax, @pgf("\\nextgroupplot[title=SARSOP, xlabel=Sea Temperature (°C), ylabel=Average Adult Female Sea Lice per Fish, xmin=8, xmax=24, ymin=0, ymax=1.0, title style={color=black}, xlabel style={color=black}, ylabel style={color=black}, tick label style={color=black}, axis background/.style={fill=white}, legend style={fill=white,draw=black,text=black}]"))
    
    # Plot SARSOP data
    if !isempty(sarsop_data.no_treatment_coords)
        push!(ax, @pgf("\\addplot[blue, mark=square, mark size=1pt, only marks] coordinates {$(join(["($(coord[1]), $(coord[2]))" for coord in sarsop_data.no_treatment_coords], " "))};"))
    end
    if !isempty(sarsop_data.treatment_coords)
        push!(ax, @pgf("\\addplot[green, mark=square, mark size=1pt, only marks] coordinates {$(join(["($(coord[1]), $(coord[2]))" for coord in sarsop_data.treatment_coords], " "))};"))
    end
    if !isempty(sarsop_data.thermal_coords)
        push!(ax, @pgf("\\addplot[red, mark=square, mark size=1pt, only marks] coordinates {$(join(["($(coord[1]), $(coord[2]))" for coord in sarsop_data.thermal_coords], " "))};"))
    end
    
    # Add SARSOP legend
    push!(ax, @pgf("\\addlegendentry{No Treatment}"))
    push!(ax, @pgf("\\addlegendentry{Chemical}"))
    push!(ax, @pgf("\\addlegendentry{Thermal}"))
    
    # Heuristic subplot (right) - no ylabel, shared y-axis
    push!(ax, @pgf("\\nextgroupplot[title=Heuristic, xlabel=Sea Temperature (°C), xmin=8, xmax=24, ymin=0, ymax=1.0, title style={color=black}, xlabel style={color=black}, tick label style={color=black}, axis background/.style={fill=white}, legend style={fill=white,draw=black,text=black}]"))
    
    # Plot Heuristic data
    if !isempty(heuristic_data.no_treatment_coords)
        push!(ax, @pgf("\\addplot[blue, mark=square, mark size=1pt, only marks] coordinates {$(join(["($(coord[1]), $(coord[2]))" for coord in heuristic_data.no_treatment_coords], " "))};"))
    end
    if !isempty(heuristic_data.treatment_coords)
        push!(ax, @pgf("\\addplot[red, mark=square, mark size=1pt, only marks] coordinates {$(join(["($(coord[1]), $(coord[2]))" for coord in heuristic_data.treatment_coords], " "))};"))
    end
    if !isempty(heuristic_data.thermal_coords)
        push!(ax, @pgf("\\addplot[red, mark=square, mark size=1pt, only marks] coordinates {$(join(["($(coord[1]), $(coord[2]))" for coord in heuristic_data.thermal_coords], " "))};"))
    end
    
    # Add Heuristic legend
    push!(ax, @pgf("\\addlegendentry{No Treatment}"))
    push!(ax, @pgf("\\addlegendentry{Chemical}"))
    push!(ax, @pgf("\\addlegendentry{Thermal}"))
    
    PGFPlotsX.save("debug_plots/combined_policy_action_heatmaps_latex.pdf", ax)
    save_transparent_png("debug_plots/combined_policy_action_heatmaps_latex.pdf", ax)
    return ax
end

# ----------------------------
# Helper function to generate policy action data
# ----------------------------
function generate_policy_action_data(policy_type, config, λ)
    
    # Define temperature and sea lice level ranges
    temp_range = 8.0:0.5:24.0  # Sea temperature range (°C)
    sealice_range = 0.0:0.01:1.0  # Sea lice level range
    
    # Fixed values for sessile and motile lice
    fixed_sessile = 0.25  # Constant sessile level
    fixed_motile = 0.25   # Constant motile level
    
    # Create separate coordinate lists for each action
    no_treatment_coords = []
    treatment_coords = []
    thermal_coords = []
    
    if policy_type == "SARSOP"
        # Load the SARSOP policy
        policy_path = joinpath(config.policies_dir, "NUS_SARSOP_Policy", "policy_pomdp_mdp_$(λ)_lambda.jld2")
        if !isfile(policy_path)
            error("Policy file not found: $policy_path")
        end
        @load policy_path policy pomdp mdp
    else  # Heuristic
        # Create heuristic policy
        heuristic_config = HeuristicConfig(
            raw_space_threshold=config.heuristic_threshold,
            belief_threshold_mechanical=config.heuristic_belief_threshold_mechanical,
            belief_threshold_thermal=config.heuristic_belief_threshold_thermal,
            rho=config.heuristic_rho
        )
        
        # Create POMDP for the heuristic policy
        if config.log_space
            pomdp = SeaLiceLogPOMDP(
                lambda=λ,
                reward_lambdas=config.reward_lambdas,
                costOfTreatment=config.costOfTreatment,
                growthRate=config.growthRate,
                rho=config.rho,
                discount_factor=config.discount_factor,
                discretization_step=config.discretization_step,
                adult_sd=abs(log(config.raw_space_sampling_sd)),
                regulation_limit=config.regulation_limit,
                full_observability_solver=config.full_observability_solver,
            )
        else
            pomdp = SeaLicePOMDP(
                lambda=λ,
                reward_lambdas=config.reward_lambdas,
                costOfTreatment=config.costOfTreatment,
                growthRate=config.growthRate,
                rho=config.rho,
                discount_factor=config.discount_factor,
                discretization_step=config.discretization_step,
                adult_sd=config.raw_space_sampling_sd,
                regulation_limit=config.regulation_limit,
                full_observability_solver=config.full_observability_solver,
            )
        end
        policy = HeuristicPolicy(pomdp, heuristic_config)
    end
    
    for (i, sealice_level) in enumerate(sealice_range)
        for (j, temp) in enumerate(temp_range)
            # Predict next sea lice level using the current state
            pred_adult, pred_motile, pred_sessile = predict_next_abundances(
                sealice_level, fixed_motile, fixed_sessile, temp
            )
            
            # Create a belief state centered on the predicted level
            if pomdp isa SeaLiceLogPOMDP
                pred_adult = log(max(pred_adult, 1e-6))
                # Create a belief vector for log space
                state_space = states(pomdp)
                belief = zeros(length(state_space))
                # Find closest state and set belief
                distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
                closest_idx = argmin(distances)
                belief[closest_idx] = 1.0
            else
                # Create a belief vector for raw space
                state_space = states(pomdp)
                belief = zeros(length(state_space))
                # Find closest state and set belief
                distances = [abs(s.SeaLiceLevel - pred_adult) for s in state_space]
                closest_idx = argmin(distances)
                belief[closest_idx] = 1.0
            end
            
            # Get action from policy
            try
                chosen_action = action(policy, belief)
                coord = (temp, sealice_level)
                
                if chosen_action == NoTreatment
                    push!(no_treatment_coords, coord)
                elseif chosen_action == Treatment
                    push!(treatment_coords, coord)
                elseif chosen_action == ThermalTreatment
                    push!(thermal_coords, coord)
                end
            catch e
                @warn "Could not get action for temp=$temp, sealice=$sealice_level: $e"
                push!(no_treatment_coords, (temp, sealice_level))  # Default to no treatment
            end
        end
    end
    
    return (no_treatment_coords=no_treatment_coords, 
            treatment_coords=treatment_coords, 
            thermal_coords=thermal_coords)
end

# ----------------------------
# Plot one simulation with all state variables over time, including
# temperature, sea lice levels, average number of fish, average fish weight
# ----------------------------
function plot_one_simulation_with_all_state_variables_over_time(parallel_data, config, policy_name)

    data_filtered = filter(row -> row.policy == policy_name, parallel_data)
    seeds = unique(data_filtered.seed)
    data_seed = filter(row -> row.seed == seeds[1], data_filtered)
    h = data_seed.history[1]
    states = collect(state_hist(h))
    actions = collect(action_hist(h))
    rewards = collect(reward_hist(h))

    p = plot(
        title="State Variables Over Time ($policy_name)",
        xlabel="Time",
        ylabel="Value",
        legend=:topright,
        grid=true,
    )
    
    # plot!(p, 1:length(states), [s.Temperature for s in states], label="Temperature")
    plot!(p, 1:length(states), [s.Adult for s in states], label="Adult Sea Lice")
    plot!(p, 1:length(states), [s.Motile for s in states], label="Motile Sea Lice")
    plot!(p, 1:length(states), [s.Sessile for s in states], label="Sessile Sea Lice")
    # plot!(p, 1:length(states), [s.NumberOfFish for s in states], label="Number of Fish")
    plot!(p, 1:length(states), [s.AvgFishWeight for s in states], label="Average Fish Weight")
    # plot!(p, 1:length(states), [s.Salinity for s in states], label="Salinity")
    
    savefig(p, "debug_plots/one_simulation_with_all_state_variables_over_time_$(policy_name).png")
    return p
end

# ----------------------------
# Combined Treatment Probability Over Time: Shows SARSOP and Heuristic policies
# side by side with treatment probabilities for each treatment type
# ----------------------------
function plot_combined_treatment_probability_over_time(parallel_data, config)
    
    # Create the combined plot using groupplots
    ax = @pgf GroupPlot(Options(
        :group_style => "{group size=2 by 1, horizontal sep=0.5cm}",
        :width => "14cm",
        :height => "8cm",
        :title => "Treatment Probability Over Time: SARSOP vs Heuristic Policies",
        :title_style => "color=white",
        "axis background/.style" => Options("fill" => "none"),
        "legend style" => Options("fill" => "none", "draw" => "none", "text" => "white"),
    ))
    
    # Define treatment types and their colors
    treatment_types = [NoTreatment, Treatment, ThermalTreatment]
    treatment_colors = ["teal", "blue", "violet"]
    treatment_names = ["No Treatment", "Chemical", "Thermal"]
    
    # Define policies to plot
    policies = ["NUS_SARSOP_Policy", "Heuristic_Policy"]
    policy_titles = ["SARSOP", "Heuristic"]
    
    # Process each policy
    for (policy_idx, policy_name) in enumerate(policies)
        
        # Filter data for this policy
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        
        # Get all unique seeds
        seeds = unique(data_filtered.seed)
        
        # Calculate treatment probabilities for each treatment type and time step
        time_steps = 1:config.steps_per_episode
        treatment_probs = Dict{Action, Vector{Float64}}()
        
        for treatment_type in treatment_types
            treatment_probs[treatment_type] = Float64[]
        end
        
        for t in time_steps
            # Extract treatment decisions at time step t from all episodes
            step_treatments = Dict{Action, Int}(
                NoTreatment => 0,
                Treatment => 0,
                ThermalTreatment => 0
            )
            
            total_episodes = 0
            
            for seed in seeds
                data_seed = filter(row -> row.seed == seed, data_filtered)
                h = data_seed.history[1]
                actions = collect(action_hist(h))
                
                if t <= length(actions)
                    action = actions[t]
                    step_treatments[action] += 1
                    total_episodes += 1
                end
            end
            
            # Calculate probabilities for each treatment type
            for treatment_type in treatment_types
                if total_episodes > 0
                    prob = step_treatments[treatment_type] / total_episodes
                    push!(treatment_probs[treatment_type], prob)
                else
                    push!(treatment_probs[treatment_type], 0.0)
                end
            end
        end
        
        # Create subplot for this policy
        if policy_idx == 1
            # SARSOP subplot (left) - show ylabel
            push!(ax, @pgf("\\nextgroupplot[title=$(policy_titles[policy_idx]), xlabel=Time (Weeks), ylabel=Treatment Probability, xmin=1, xmax=$(config.steps_per_episode), ymin=0, ymax=1.0, title style={color=white}, xlabel style={color=white}, ylabel style={color=white}, tick label style={color=white}, axis background/.style={fill=none}, legend style={fill=none,draw=none,text=white}]"))
        else
            # Heuristic subplot (right) - no ylabel, shared y-axis
            push!(ax, @pgf("\\nextgroupplot[title=$(policy_titles[policy_idx]), xlabel=Time (Weeks), xmin=1, xmax=$(config.steps_per_episode), ymin=0, ymax=1.0, title style={color=white}, xlabel style={color=white}, tick label style={color=white}, axis background/.style={fill=none}, legend style={fill=none,draw=none,text=white}]"))
        end
        
        # Plot each treatment type
        for (treatment_idx, treatment_type) in enumerate(treatment_types)
            color = treatment_colors[treatment_idx]
            name = treatment_names[treatment_idx]
            
            # Create coordinate string for this treatment type
            coords = join(["($(t), $(treatment_probs[treatment_type][t]))" for t in 1:length(time_steps)], " ")
            
            push!(ax, @pgf("\\addplot[$(color), mark=none, line width=1.5pt] coordinates {$(coords)};"))
            
            # Add legend only for the first subplot to avoid duplication
            if policy_idx == 1
                push!(ax, @pgf("\\addlegendentry{$(name)}"))
            end
        end
    end
    
    PGFPlotsX.save("debug_plots/combined_treatment_probability_over_time_latex.pdf", ax)
    save_transparent_png("debug_plots/combined_treatment_probability_over_time_latex.pdf", ax)
    return ax
end

# ----------------------------
# Plot 2: Time series of belief means and variances using PGFPlotsX
# Creates side-by-side plots showing belief trajectories for Adult, Motile, and Sessile
# ----------------------------
function plot_beliefs_over_time(data, algo_name, config, lambda)

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
    actions = action_hist(history)
    actions = [a == Treatment ? "M" : a == ThermalTreatment ? "Th" : "" for a in actions]

    # Only plot Adult female sea lice (index 1)
    i = 1  # Adult index
    color = "cyan"
    
    # Create the plot using single axis (not groupplots)
    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :title => "Kalman Filter Belief Trajectory of Adult Sea Lice",
        :title_style => "color=white",
        :xlabel => "Time Since Production Start (Weeks)",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xlabel_style => "color=white",
        :ylabel_style => "color=white",
        :tick_label_style => "color=white",
        :xmin => 0,
        :ymin => 0,
        "axis background/.style" => Options("fill" => "none"),
        "legend style" => Options("fill" => "none", "draw" => "none", "text" => "white"),
    ))

    # Time steps
    time_steps = 1:size(belief_means, 1)
    
    # Plot belief mean with confidence interval
    belief_upper = belief_means[:, i] .+ sqrt.(belief_variances_array[:, i])
    belief_lower = belief_means[:, i] .- sqrt.(belief_variances_array[:, i])
    
    # Filter out NaN and infinite values
    valid_indices = .!isnan.(belief_means[:, i]) .&& .!isnan.(belief_upper) .&& .!isnan.(belief_lower) .&& 
                   .!isinf.(belief_means[:, i]) .&& .!isinf.(belief_upper) .&& .!isinf.(belief_lower)
    
    if sum(valid_indices) > 0
        valid_time_steps = time_steps[valid_indices]
        valid_means = belief_means[valid_indices, i]
        valid_upper = belief_upper[valid_indices]
        valid_lower = belief_lower[valid_indices]
        
        # Create coordinate strings with valid data only
        mean_coords = join(["($(t), $(valid_means[j]))" for (j, t) in enumerate(valid_time_steps)], " ")
        upper_coords = join(["($(t), $(valid_upper[j]))" for (j, t) in enumerate(valid_time_steps)], " ")
        lower_coords = join(["($(t), $(valid_lower[j]))" for (j, t) in enumerate(valid_time_steps)], " ")
        
        # Add the confidence interval fill
        push!(ax, @pgf("\\addplot[name path=upper, $(color), mark=none, line width=0.5pt] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower, $(color), mark=none, line width=0.5pt] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[$(color), fill opacity=0.3] fill between[of=upper and lower];"))
        
        # Add the mean line
        push!(ax, @pgf("\\addplot[$(color), mark=none, line width=1pt] coordinates {$(mean_coords)};"))
    end
    
    # Add true values (filter out NaN/inf)
    valid_states = .!isnan.(states_df[:, i]) .&& .!isinf.(states_df[:, i])
    if sum(valid_states) > 0
        valid_state_times = findall(valid_states)
        valid_state_values = states_df[valid_states, i]
        true_coords = join(["($(valid_state_times[j]), $(valid_state_values[j]))" for j in 1:length(valid_state_times)], " ")
        push!(ax, @pgf("\\addplot[$(color), mark=x, mark size=2pt, only marks] coordinates {$(true_coords)};"))
    end
    
    # Add observations (filter out NaN/inf)
    valid_obs = .!isnan.(observations_df[:, i]) .&& .!isinf.(observations_df[:, i])
    if sum(valid_obs) > 0
        valid_obs_times = findall(valid_obs)
        valid_obs_values = observations_df[valid_obs, i]
        obs_coords = join(["($(valid_obs_times[j]), $(valid_obs_values[j]))" for j in 1:length(valid_obs_times)], " ")
        push!(ax, @pgf("\\addplot[$(color), mark=o, mark size=2pt, only marks] coordinates {$(obs_coords)};"))
    end
    
    # Add legend
    push!(ax, @pgf("\\addlegendentry{Belief mean}"))
    push!(ax, @pgf("\\addlegendentry{True value}"))
    push!(ax, @pgf("\\addlegendentry{Observation}"))

    # Save the plot
    PGFPlotsX.save("debug_plots/belief_trajectories_$(algo_name)_lambda_$(lambda)_latex.pdf", ax)
    save_transparent_png("debug_plots/belief_trajectories_$(algo_name)_lambda_$(lambda)_latex.pdf", ax)
    
    return ax
end