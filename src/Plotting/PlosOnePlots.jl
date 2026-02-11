using Plots
using JLD2
using GaussianFilters
using Statistics
using DataFrames
using PGFPlotsX
using POMDPTools
using Dates
using PGFPlotsX: @pgf, Axis, Plot, Coordinates, GroupPlot, Options

# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usetikzlibrary{intersections}")
# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

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

# Consistent palette + labeling for the Plos One figures
const PLOS_POLICY_STYLE_ORDERED = [
    ("NeverTreat_Policy", (; label = "NeverTreat",   line = "gray!70!black",    fill = "gray!20!white")),
    ("AlwaysTreat_Policy",(; label = "AlwaysTreat",  line = "red!80!black",     fill = "red!20!white")),
    ("Random_Policy",     (; label = "Random",       line = "orange!85!black",  fill = "orange!25!white")),
    ("Heuristic_Policy", (; label = "Heuristic",     line = "teal!75!black",    fill = "teal!25!white")),
    ("QMDP_Policy",       (; label = "QMDP",         line = "magenta!70!black", fill = "magenta!25!white")),
    ("NUS_SARSOP_Policy", (; label = "NUS SARSOP",   line = "blue!80!black",    fill = "blue!25!white")),
    ("Native_SARSOP_Policy", (; label = "SARSOP",   line = "cyan!70!black",    fill = "cyan!25!white")),
    ("VI_Policy",         (; label = "VI",           line = "violet!80!black",  fill = "violet!25!white")),
]
const PLOS_POLICY_STYLE_LOOKUP = Dict(name => style for (name, style) in PLOS_POLICY_STYLE_ORDERED)

const PLOS_ACTION_STYLE_ORDERED = [
    (NoTreatment,         (; label = "No Treatment",        color = "blue!70!black",    marker = "o",  mark_opts = "{fill=white}")),
    (MechanicalTreatment, (; label = "Mechanical",          color = "teal!75!black",    marker = "triangle*", mark_opts = "{solid}")),
    (ChemicalTreatment,   (; label = "Chemical",            color = "orange!85!black",  marker = "square*",   mark_opts = "{solid}")),
    (ThermalTreatment,    (; label = "Thermal",             color = "red!80!black",     marker = "diamond*",  mark_opts = "{solid}")),
]
const PLOS_ACTION_STYLE_LOOKUP = Dict(action => style for (action, style) in PLOS_ACTION_STYLE_ORDERED)

const PLOS_STAGE_STYLE_ORDERED = [
    (:adult,     (; label = "Adult (true)",        line = "blue!80!black",   fill = "blue!25!white")),
    (:sessile,   (; label = "Sessile",             line = "purple!70!black", fill = "purple!20!white")),
    (:motile,    (; label = "Motile",              line = "teal!70!black",   fill = "teal!20!white")),
    (:predicted, (; label = "Belief (predicted)",  line = "black!70",        fill = "black!10")),
]
const PLOS_STAGE_STYLE_LOOKUP = Dict(key => style for (key, style) in PLOS_STAGE_STYLE_ORDERED)

# Unwrap EnrichedBelief to GaussianBelief for GaussianFilters.unpack()
_unwrap_beliefs(beliefs) =
    !isempty(beliefs) && first(beliefs) isa EnrichedBelief ?
    [b.gaussian_belief for b in beliefs] : beliefs

const PLOS_FONT = "\\small"
const PLOS_LABEL_STYLE = "color=black, font=\\small"
const PLOS_TICK_STYLE = "color=black, font=\\small"
const PLOS_TITLE_STYLE = "color=black, font=\\small"
const PLOS_FILL_OPACITY = 0.5

_std_with_guard(v) = length(v) > 1 ? std(v; corrected=true) : 0.0

function _clamp_values!(vec, ymin, ymax)
    if ymin !== nothing
        for idx in eachindex(vec)
            vec[idx] = max(vec[idx], ymin)
        end
    end
    if ymax !== nothing
        for idx in eachindex(vec)
            vec[idx] = min(vec[idx], ymax)
        end
    end
    return vec
end

function _add_reg_limit!(ax, xmax, y; label="Reg. limit (0.5)", use_legend::Bool=false)
    xmax_val = float(xmax)
    if use_legend
        push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(1,$(y)) ($(xmax_val),$(y))};"))
        push!(ax, @pgf("\\addlegendentry{$(label)}"))
    else
        push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt, forget plot] coordinates {(1,$(y)) ($(xmax_val),$(y))};"))
        label_x = min(0.5, xmax_val)
        label_y = y + 0.04
        push!(ax, @pgf("""\\node[anchor=west, font=\\scriptsize, text=black!70] at (axis cs:$(label_x), $(label_y)) {$label};"""))
    end
end

const DEFAULT_EXCLUDE_POLICIES = Set(["NeverTreat_Policy", "AlwaysTreat_Policy"])

function _selected_policy_styles(policies_to_plot; exclude=DEFAULT_EXCLUDE_POLICIES)
    if policies_to_plot === nothing
        base = PLOS_POLICY_STYLE_ORDERED
    else
        wanted = Set(policies_to_plot)
        base = [(name, style) for (name, style) in PLOS_POLICY_STYLE_ORDERED if name in wanted]
        if isempty(base)
            @warn "No matching policies found for $(collect(wanted)); defaulting to all policies"
            base = PLOS_POLICY_STYLE_ORDERED
        else
            found = Set(first.(base))
            missing = setdiff(wanted, found)
            if !isempty(missing)
                @warn "Policies without plotting styles ignored: $(collect(missing))"
            end
        end
    end
    if !isempty(exclude)
        base = [(name, style) for (name, style) in base if !(name in exclude)]
    end
    return base
end

const PLOS_LEGEND_Y = 1.23

function plos_top_legend(; columns=nothing)
    args = [
        "fill" => "white",
        "draw" => "black!40",
        "text" => "black",
        "font" => PLOS_FONT,
        "at" => "{(0.5,$(PLOS_LEGEND_Y))}",
        "anchor" => "south",
        "row sep" => "1pt",
        "column sep" => "0.5cm",
    ]
    if columns !== nothing
        push!(args, "legend columns" => string(columns))
    end
    return Options(args...)
end

const DEFAULT_PLOS_START_DATE = Date(2000, 9, 1)
const DEFAULT_PRODUCTION_START_WEEK = 37

function _plos_start_date(config)
    sim_cfg = config.simulation_config
    if hasproperty(sim_cfg, :start_date)
        start_date = getproperty(sim_cfg, :start_date)
        start_date isa Date && return start_date
    end
    year_val = hasproperty(sim_cfg, :start_year) ? getproperty(sim_cfg, :start_year) : Dates.year(DEFAULT_PLOS_START_DATE)
    if hasproperty(sim_cfg, :start_month) || hasproperty(sim_cfg, :start_day)
        month_val = hasproperty(sim_cfg, :start_month) ? getproperty(sim_cfg, :start_month) : Dates.month(DEFAULT_PLOS_START_DATE)
        day_val = hasproperty(sim_cfg, :start_day) ? getproperty(sim_cfg, :start_day) : Dates.day(DEFAULT_PLOS_START_DATE)
        return Date(year_val, month_val, day_val)
    end
    prod_week = hasproperty(sim_cfg, :production_start_week) ? getproperty(sim_cfg, :production_start_week) : DEFAULT_PRODUCTION_START_WEEK
    prod_week = max(prod_week, 1)
    return Date(year_val, 1, 1) + Week(prod_week - 1)
end

function plos_time_ticks(config)
    steps = config.simulation_config.steps_per_episode
    start_date = _plos_start_date(config)
    ticks = Int[]
    labels = String[]
    prev_date = start_date
    push!(ticks, 1)
    push!(labels, monthabbr(start_date))
    show = true
    for week in 2:steps
        d = start_date + Week(week - 1)
        if Dates.month(d) != Dates.month(prev_date) || Dates.year(d) != Dates.year(prev_date)
            push!(ticks, week)
            label = monthabbr(d)
            if show
                push!(labels, label)
            else
                push!(labels, "")
            end
            show = !show
        end
        prev_date = d
    end
    return ticks, labels
end

# Backend is set in AquaOpt.__init__()

# ----------------------------
# Plot 1: Time series of belief means and variances using PGFPlotsX
# Creates side-by-side plots showing belief trajectories for Adult, Motile, and Sessile
# ----------------------------
function plos_one_plot_kalman_filter_belief_trajectory(data, algo_name, config; num_sigmas::Int=3)

    # Filter the data to only include the algorithm
    data = filter(row -> row.policy == algo_name, data)

    # Extract first belief history for given solver
    history = data.history[1]

    # Extract beliefs
    beliefs = _unwrap_beliefs(belief_hist(history))
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

    # Only plot Adult female sea lice (index 1)
    i = 1  # Adult index

    # Softer palette to match other plots / improve accessibility
    belief_line_color = "blue!80!black"
    belief_fill_color = "blue!30!white"
    true_color = "black!70"
    obs_color = "orange!85!black"
    
    ticks, labels = plos_time_ticks(config)

    # Create the plot using single axis (not groupplots)
    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :title_style => PLOS_TITLE_STYLE,
        :xlabel => "Time of Year",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 0,
        :ymin => 0,
        :enlarge_x_limits => "false",
        :enlarge_y_limits => "false",
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=4),
    ))

    _add_reg_limit!(ax, config.simulation_config.steps_per_episode, 0.5)

    # Time steps
    time_steps = 1:size(belief_means, 1)
    
    # Plot belief mean with ±num_sigmasσ confidence band
    sigma = sqrt.(belief_variances_array[:, i])
    belief_upper = belief_means[:, i] .+ num_sigmas .* sigma
    belief_lower = belief_means[:, i] .- num_sigmas .* sigma
    
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
        push!(ax, @pgf("\\addplot[name path=upper, draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower, draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(belief_fill_color), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper and lower];"))
        push!(ax, @pgf("\\addlegendimage{area legend, draw=$(belief_line_color), fill=$(belief_fill_color), fill opacity=$(PLOS_FILL_OPACITY)}"))
        push!(ax, @pgf("\\addlegendentry{Belief \$\\pm $(num_sigmas)\\sigma\$}"))
        
        # Add the mean line
        push!(ax, @pgf("\\addplot[draw=$(belief_line_color), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        push!(ax, @pgf("\\addlegendentry{Belief mean}"))
    else
        push!(ax, @pgf("\\addlegendimage{area legend, draw=$(belief_line_color), fill=$(belief_fill_color), fill opacity=$(PLOS_FILL_OPACITY)}"))
        push!(ax, @pgf("\\addlegendentry{Belief \$\\pm $(num_sigmas)\\sigma\$}"))
        push!(ax, @pgf("\\addlegendimage{draw=$(belief_line_color), line width=1.2pt}"))
        push!(ax, @pgf("\\addlegendentry{Belief mean}"))
    end
    
    # Add true values (filter out NaN/inf)
    valid_states = .!isnan.(states_df[:, i]) .&& .!isinf.(states_df[:, i])
    if sum(valid_states) > 0
        valid_state_times = findall(valid_states)
        valid_state_values = states_df[valid_states, i]
        true_coords = join(["($(valid_state_times[j]), $(valid_state_values[j]))" for j in 1:length(valid_state_times)], " ")
        push!(ax, @pgf("\\addplot[only marks, mark=*, mark size=1.8pt, mark options={draw=$(true_color), fill=$(true_color)}] coordinates {$(true_coords)};"))
        push!(ax, @pgf("\\addlegendentry{True value}"))
    else
        # Add a legend entry without plotting dummy data
        push!(ax, @pgf("\\addlegendimage{only marks, mark=*, mark size=1.8pt, mark options={draw=$(true_color), fill=$(true_color)}}"))
        push!(ax, @pgf("\\addlegendentry{True value}"))
    end
    
    # Add observations (filter out NaN/inf)
    valid_obs = .!isnan.(observations_df[:, i]) .&& .!isinf.(observations_df[:, i])
    if sum(valid_obs) > 0
        valid_obs_times = findall(valid_obs)
        valid_obs_values = observations_df[valid_obs, i]
        obs_coords = join(["($(valid_obs_times[j]), $(valid_obs_values[j]))" for j in 1:length(valid_obs_times)], " ")
        push!(ax, @pgf("\\addplot[only marks, mark=o, mark size=2pt, mark options={draw=$(obs_color), fill=white}] coordinates {$(obs_coords)};"))
        push!(ax, @pgf("\\addlegendentry{Observation}"))
    else
        # Add a legend entry without plotting dummy data
        push!(ax, @pgf("\\addlegendimage{only marks, mark=o, mark size=2pt, mark options={draw=$(obs_color), fill=white}}"))
        push!(ax, @pgf("\\addlegendentry{Observation}"))
    end

    # Save the plot
    mkpath("Quick_Access")
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sarsop_kalman_filter_belief_trajectory_$(num_sigmas)sigmas.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sarsop_kalman_filter_belief_trajectory_$(num_sigmas)sigmas.tex"), ax; include_preamble=false)
    return ax
end

# ----------------------------
# Plot: Sea Lice Levels Over Time - Policy Comparison
# Compares effectiveness of different aquaculture management policies in controlling 
# adult female sea lice populations over time with optional 95% confidence interval ribbons.
# Shows compliance with regulatory limit (0.5 lice/fish) across multiple policies.
# Parameters: show_ci=true/false to toggle confidence interval ribbons
# ----------------------------
function plos_one_sealice_levels_over_time(parallel_data, config;
        show_ci=true, policies_to_plot=nothing, show_legend=true)
    ticks, labels = plos_time_ticks(config)
    axis_options = Any[
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 1,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 0.6,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_legend
        push!(axis_options, "legend style" => plos_top_legend(columns=8))
    end
    ax = @pgf Axis(Options(axis_options...))

    for (policy_name, style) in _selected_policy_styles(policies_to_plot)
        try
            data_filtered = filter(row -> row.policy == policy_name, parallel_data)
            isempty(data_filtered) && continue

            seeds = unique(data_filtered.seed)
            isempty(seeds) && continue

            time_steps = 1:config.simulation_config.steps_per_episode
            mean_sealice = zeros(Float64, length(time_steps))
            ci_lower = similar(mean_sealice)
            ci_upper = similar(mean_sealice)

            for (idx, t) in enumerate(time_steps)
                step_sealice = Float64[]
                for seed in seeds
                    data_seed = filter(row -> row.seed == seed, data_filtered)
                    isempty(data_seed) && continue
                    states = collect(state_hist(data_seed.history[1]))
                    if t <= length(states)
                        push!(step_sealice, states[t].SeaLiceLevel)
                    end
                end

                if isempty(step_sealice)
                    mean_sealice[idx] = NaN
                    ci_lower[idx] = NaN
                    ci_upper[idx] = NaN
                    continue
                end

                mean_level = mean(step_sealice)
                std_level = _std_with_guard(step_sealice)
                se_level = std_level / sqrt(length(step_sealice))
                margin = 1.96 * se_level

                mean_sealice[idx] = mean_level
                ci_lower[idx] = mean_level - margin
                ci_upper[idx] = mean_level + margin
            end

            valid_indices = .!isnan.(mean_sealice) .&& .!isnan.(ci_lower) .&& .!isnan.(ci_upper)
            if any(valid_indices)
                valid_time = time_steps[valid_indices]
                valid_mean = mean_sealice[valid_indices]
                valid_lower = ci_lower[valid_indices]
                valid_upper = ci_upper[valid_indices]
                mean_coords = join(["($(valid_time[j]), $(valid_mean[j]))" for j in 1:length(valid_time)], " ")

                safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")

                if show_ci
                    upper_coords = join(["($(valid_time[j]), $(valid_upper[j]))" for j in 1:length(valid_time)], " ")
                    lower_coords = join(["($(valid_time[j]), $(valid_lower[j]))" for j in 1:length(valid_time)], " ")
                    push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
                    push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
                    push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
                end

                push!(ax, @pgf("\\addplot[mark=none, line width=1.4pt, color=$(style.line)] coordinates {$(mean_coords)};"))
                if show_legend
                    push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
                end
            end
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end

    _add_reg_limit!(ax, config.simulation_config.steps_per_episode, 0.5; use_legend=show_legend)

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sealice_levels_over_time.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sealice_levels_over_time.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "north_sealice_levels_over_time.pdf"), ax)
    return ax
end


function plos_one_episode_sealice_levels_over_time(
    parallel_data,
    config;
    episode_id::Int = 1,
    savefig::Bool = true,
    policies_to_plot = nothing,
    show_legend::Bool = true,
)
    # Setup
    ticks, labels = plos_time_ticks(config)
    policy_styles = collect(_selected_policy_styles(policies_to_plot))
    isempty(policy_styles) && return nothing

    # Create multi-panel figure
    n_panels = length(policy_styles)
    middle_panel = ceil(Int, n_panels / 2)
    axis_height = clamp(5.2 - 0.35 * (n_panels - 1), 3.2, 5.2)
    gp = @pgf GroupPlot({
        group_style = {
            "group size" => "1 by $n_panels",
            "vertical sep" => "40pt",
            "xlabels at" => "edge bottom",
            "xticklabels at" => "edge bottom",
        },
        width => "18cm",
        height => "$(axis_height)cm",
        xlabel => "Time of Year",
        xlabel_style => PLOS_LABEL_STYLE,
    })

    # Plot each policy in its own panel
    for (panel_idx, (policy_name, style)) in enumerate(policy_styles)
        is_first = panel_idx == 1
        is_middle = panel_idx == middle_panel
        # Create axis
        ax = _create_episode_panel_axis(ticks, labels, config, style;
            add_ylabel=is_middle, add_legend=(show_legend && is_first))

        # Load and plot episode data
        _plot_episode_data!(ax, parallel_data, policy_name, style, episode_id, config)

        # Add treatment legend to first panel
        if is_first && show_legend
            _add_episode_treatment_legend!(ax)
        end

        push!(gp, ax)
    end

    # Save output files
    if savefig
        _save_episode_figure(gp, config, episode_id)
    end

    return gp
end

function _create_episode_panel_axis(ticks, labels, config, style;
        add_ylabel::Bool=false, add_legend::Bool=false)
    axis_opts = Any[
        :xmin => 1,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 1,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
        "tick label style" => PLOS_TICK_STYLE,
        "title" => style.label,
        "title style" => PLOS_TITLE_STYLE,
    ]

    if add_ylabel
        push!(axis_opts, :ylabel => "Adult Female\\\\Sea Lice per Fish")
        push!(axis_opts, :ylabel_style => "{" * PLOS_LABEL_STYLE * ", align=center}")
    end

    if add_legend
        push!(axis_opts, "legend style" => plos_top_legend(columns=3))
    end

    return @pgf Axis(Options(axis_opts...))
end

function _plot_episode_data!(ax, parallel_data, policy_name, style, episode_id, config)
    # Filter for this policy
    data_filtered = filter(row -> row.policy == policy_name, parallel_data)
    seeds = unique(data_filtered.seed)

    # Validate data
    if isempty(seeds) || episode_id > length(seeds)
        isempty(seeds) && @warn "No seeds found for $policy_name"
        episode_id > length(seeds) && @warn "Episode $episode_id not available for $policy_name (only $(length(seeds)) episodes)"
        return
    end

    # Extract episode
    selected_seed = seeds[episode_id]
    episode_df = filter(row -> row.seed == selected_seed, data_filtered)
    if isempty(episode_df)
        @warn "No data for seed $selected_seed in $policy_name"
        return
    end

    try
        # Extract history
        history = episode_df.history[1]
        states = collect(state_hist(history))
        actions = collect(action_hist(history))

        # Plot trajectory (no per-panel legend — policy name is the panel title)
        _add_trajectory!(ax, states, style, false)

        # Plot treatment markers (no per-marker legend — handled by first panel)
        _add_treatment_markers!(ax, states, actions)

        # Add regulatory limit
        _add_reg_limit!(ax, config.simulation_config.steps_per_episode, 0.5)
    catch e
        @warn "Error plotting episode for $policy_name: $e"
    end
end

function _add_trajectory!(ax, states, style, add_legend)
    time_steps = 1:length(states)
    levels = [st.SeaLiceLevel for st in states]
    coords = join(["($(time_steps[i]), $(levels[i]))" for i in eachindex(levels)], " ")

    push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.25pt] coordinates {$coords};"))

    if add_legend
        push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
        return true
    end
    return false
end

function _add_treatment_markers!(ax, states, actions)
    treatment_steps = findall(a -> a != NoTreatment, actions)
    isempty(treatment_steps) && return

    levels = [st.SeaLiceLevel for st in states]
    action_coords = Dict{Any, Vector{Tuple{Float64, Float64}}}()

    for t in treatment_steps
        act = actions[t]
        coords_vec = get!(action_coords, act, Tuple{Float64,Float64}[])
        push!(coords_vec, (t, levels[t]))
    end

    for (act, coords_vec) in action_coords
        action_style = get(PLOS_ACTION_STYLE_LOOKUP, act,
            (; color="black!70", marker="*", mark_opts="{solid}", label="Treatment"))
        coords_str = join(["($(x), $(y))" for (x, y) in coords_vec], " ")
        push!(ax, @pgf("\\addplot[only marks, mark=$(action_style.marker), mark size=2.6pt, color=$(action_style.color), mark options=$(action_style.mark_opts), forget plot] coordinates {$coords_str};"))
    end
end

function _add_episode_treatment_legend!(ax)
    for (act, style) in PLOS_ACTION_STYLE_ORDERED
        act == NoTreatment && continue
        push!(ax, raw"\addlegendimage{only marks, mark=" * style.marker *
            ", mark size=2.6pt, color=" * style.color *
            ", mark options=" * style.mark_opts * "}")
        push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end
end

function _save_episode_figure(gp, config, episode_id)
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")

    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "sealice_episode_$(episode_id).pdf"), gp)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "sealice_episode_$(episode_id).tex"), gp; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "sealice_episode_$(episode_id).pdf"), gp)
end


# ----------------------------
# Shared helpers for reward metric time-series plots
# ----------------------------
function _extract_metric_caches(data_filtered, seeds)
    caches = NamedTuple[]
    for seed in seeds
        data_seed = filter(row -> row.seed == seed, data_filtered)
        isempty(data_seed) && continue
        history = data_seed.history[1]
        states = collect(state_hist(history))
        actions = collect(action_hist(history))
        rewards = Float64.(collect(reward_hist(history)))
        push!(caches, (; states, actions, rewards))
    end
    return caches
end

function _plos_one_metric_plot(parallel_data, config, compute_step_value;
        ylabel, file_suffix, ymin=nothing, ymax=nothing,
        policies_to_plot=nothing, show_legend::Bool=true)

    ticks, labels = plos_time_ticks(config)
    axis_pairs = Any[
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => ylabel,
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => config.simulation_config.steps_per_episode,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_legend
        push!(axis_pairs, "legend style" => plos_top_legend(columns=7))
    end
    if ymin !== nothing
        push!(axis_pairs, :ymin => ymin)
    end
    if ymax !== nothing
        push!(axis_pairs, :ymax => ymax)
    end
    axis_ymin = ymin
    axis_ymax = ymax
    ax = @pgf Axis(Options(axis_pairs...))

    time_steps = 1:config.simulation_config.steps_per_episode
    for (policy_name, style) in _selected_policy_styles(policies_to_plot)
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue
        seeds = unique(data_filtered.seed)
        isempty(seeds) && continue
        caches = _extract_metric_caches(data_filtered, seeds)
        isempty(caches) && continue

        mean_values = fill(NaN, length(time_steps))
        lower_values = fill(NaN, length(time_steps))
        upper_values = fill(NaN, length(time_steps))

        for (idx, t) in enumerate(time_steps)
            step_values = Float64[]
            for cache in caches
                value = compute_step_value(cache, t, config)
                if value !== nothing && !isnan(value)
                    push!(step_values, value)
                end
            end
            if !isempty(step_values)
                n = length(step_values)
                mean_val = mean(step_values)
                std_val = _std_with_guard(step_values)
                se = n > 0 ? std_val / sqrt(n) : 0.0
                margin = 1.96 * se
                mean_values[idx] = mean_val
                lower_values[idx] = mean_val - margin
                upper_values[idx] = mean_val + margin
            end
        end

        valid_mask = .!isnan.(mean_values) .&& .!isnan.(lower_values) .&& .!isnan.(upper_values)
        valid_indices = findall(valid_mask)
        isempty(valid_indices) && continue

        valid_times = [time_steps[i] for i in valid_indices]
        valid_mean = mean_values[valid_indices]
        valid_lower = lower_values[valid_indices]
        valid_upper = upper_values[valid_indices]
        _clamp_values!(valid_mean, axis_ymin, axis_ymax)
        _clamp_values!(valid_lower, axis_ymin, axis_ymax)
        _clamp_values!(valid_upper, axis_ymin, axis_ymax)

        mean_coords = join(["($(valid_times[j]), $(valid_mean[j]))" for j in eachindex(valid_times)], " ")
        upper_coords = join(["($(valid_times[j]), $(valid_upper[j]))" for j in eachindex(valid_times)], " ")
        lower_coords = join(["($(valid_times[j]), $(valid_lower[j]))" for j in eachindex(valid_times)], " ")

        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.4pt] coordinates {$(mean_coords)};"))
        if show_legend
            push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
        end
    end

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", file_suffix), ax)
    tex_suffix = replace(file_suffix, ".pdf" => ".tex")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", tex_suffix), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", file_suffix), ax)
    return ax
end

reward_step_value(cache, t, _) = t <= length(cache.rewards) ? cache.rewards[t] : nothing

function biomass_loss_step_value(cache, t, config)
    if t <= 1 || t > length(cache.states)
        return nothing
    end
    sim_params = SeaLiceSimPOMDP(location=config.solver_config.location)
    n_pairs = min(length(cache.states) - 1, length(cache.actions))
    cumulative_loss = 0.0
    for τ in 1:min(t-1, n_pairs)
        cumulative_loss += _expected_biomass_shortfall(sim_params, cache.states[τ], cache.states[τ+1])
    end
    return cumulative_loss
end

function regulatory_penalty_step_value(cache, t, config)
    if t <= length(cache.states)
        limit = config.solver_config.regulation_limit
        return cache.states[t].Adult > limit ? 1.0 : 0.0
    end
    return nothing
end

function fish_disease_step_value(cache, t, _)
    if t <= length(cache.states) && t <= length(cache.actions)
        return get_fish_disease(cache.actions[t]) + 100.0 * cache.states[t].SeaLiceLevel
    end
    return nothing
end

function treatment_cost_step_value(cache, t, _)
    if t <= length(cache.actions)
        return get_treatment_cost(cache.actions[t])
    end
    return nothing
end

function cumulative_treatment_cost_value(cache, t, _)
    if t < 1 || t > length(cache.actions)
        return nothing
    end
    total = 0.0
    for τ in 1:t
        total += get_treatment_cost(cache.actions[τ])
    end
    return total
end

function cumulative_regulatory_penalty_value(cache, t, config)
    if t < 1 || t > length(cache.states)
        return nothing
    end
    limit = config.solver_config.regulation_limit
    count = 0.0
    for τ in 1:t
        if τ <= length(cache.states) && cache.states[τ].Adult > limit
            count += 1.0
        end
    end
    return count
end

function cumulative_welfare_cost_value(cache, t, config)
    if t < 1
        return nothing
    end
    n = min(t, length(cache.actions), length(cache.states))
    welfare_MNOK = config.solver_config.welfare_cost_MNOK
    chronic_MNOK = config.solver_config.chronic_lice_cost_MNOK
    cooldown_mult = 1.5
    total = 0.0
    for τ in 1:n
        # Component 4: treatment stress (fish health)
        base_stress = get_fish_disease(cache.actions[τ])
        stress_mult = (cache.states[τ].Cooldown == 1) ? cooldown_mult : 1.0
        fish_health = base_stress * stress_mult * welfare_MNOK
        # Component 5: chronic sea lice burden
        adult = cache.states[τ].Adult
        sea_lice = adult * (1.0 + 0.2 * max(0, adult - 0.5)) * chronic_MNOK
        total += fish_health + sea_lice
    end
    return total
end

function plos_one_reward_over_time(parallel_data, config; policies_to_plot=nothing, show_legend::Bool=true)
    _plos_one_metric_plot(
        parallel_data, config, reward_step_value;
        ylabel = "Reward per Step",
        file_suffix = "north_reward_over_time.pdf",
        policies_to_plot = policies_to_plot,
        show_legend = show_legend,
    )
end

function plos_one_biomass_loss_over_time(parallel_data, config; policies_to_plot=nothing, show_legend::Bool=true)
    _plos_one_metric_plot(
        parallel_data, config, biomass_loss_step_value;
        ylabel = "Cumulative Biomass Loss (tons)",
        file_suffix = "north_biomass_loss_over_time.pdf",
        ymin = 0.0,
        policies_to_plot = policies_to_plot,
        show_legend = show_legend,
    )
end

function plos_one_regulatory_penalty_over_time(parallel_data, config; policies_to_plot=nothing, show_legend::Bool=true)
    _plos_one_metric_plot(
        parallel_data, config, regulatory_penalty_step_value;
        ylabel = "Penalty Probability",
        file_suffix = "north_regulatory_penalty_over_time.pdf",
        ymin = 0.0,
        ymax = 0.1,
        policies_to_plot = policies_to_plot,
        show_legend = show_legend,
    )
end

function plos_one_fish_disease_over_time(parallel_data, config; policies_to_plot=nothing, show_legend::Bool=true)
    _plos_one_metric_plot(
        parallel_data, config, fish_disease_step_value;
        ylabel = "Fish Disease Penalty",
        file_suffix = "north_fish_disease_over_time.pdf",
        ymin = 0.0,
        policies_to_plot = policies_to_plot,
        show_legend = show_legend,
    )
end

function plos_one_treatment_cost_over_time(parallel_data, config; policies_to_plot=nothing, show_legend::Bool=true)
    _plos_one_metric_plot(
        parallel_data, config, treatment_cost_step_value;
        ylabel = "Treatment Cost per Step",
        file_suffix = "north_treatment_cost_over_time.pdf",
        ymin = 0.0,
        policies_to_plot = policies_to_plot,
        show_legend = show_legend,
    )
end


# ----------------------------
# Treatment Probability Over Time: Shows all policies overlaid in a single plot
# Each policy shows the probability of treating (any action that is not NoTreatment)
# ----------------------------
function plos_one_combined_treatment_probability_over_time(parallel_data, config;
        policies_to_plot=nothing, show_legend::Bool=true)
    
    # Create a single plot using PGFPlotsX (same style as other plots)
    ticks, labels = plos_time_ticks(config)
    axis_options = Any[
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => "Treatment Probability",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 1.0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_legend
        push!(axis_options, "legend style" => plos_top_legend(columns=7))
    end
    ax = @pgf Axis(Options(axis_options...))
    
    # Process each policy
    for (policy_name, style) in _selected_policy_styles(policies_to_plot)
        try
            # Filter data for this policy
            data_filtered = filter(row -> row.policy == policy_name, parallel_data)
            
            if isempty(data_filtered)
                @warn "No data found for policy: $policy_name"
                continue
            end
            
            # Get all unique seeds
            seeds = unique(data_filtered.seed)
            
            # Calculate treatment probabilities over time
            time_steps = 1:config.simulation_config.steps_per_episode
            treatment_probs = zeros(Float64, length(time_steps))
            
            for (idx, t) in enumerate(time_steps)
                # Count treatment actions (any action that is not NoTreatment)
                treatment_count = 0
                total_episodes = 0
                
                for seed in seeds
                    data_seed = filter(row -> row.seed == seed, data_filtered)
                    h = data_seed.history[1]
                    actions = collect(action_hist(h))
                    
                    if t <= length(actions)
                        action = actions[t]
                        if action != NoTreatment
                            treatment_count += 1
                        end
                        total_episodes += 1
                    end
                end
                
                # Calculate treatment probability
                if total_episodes > 0
                    treatment_probs[idx] = treatment_count / total_episodes
                end
            end
            
            # Create coordinate string for this policy
            coords = join(["($(step), $(treatment_probs[idx]))" for (idx, step) in enumerate(time_steps)], " ")
            
            # Add the line plot
            push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.4pt] coordinates {$(coords)};"))
            if show_legend
                push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
            end
            
        catch e
            @warn "Could not process policy $policy_name: $e"
        end
    end
    
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_probability_over_time.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_probability_over_time.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "north_treatment_probability_over_time.pdf"), ax)
    return ax
end

# ----------------------------
# Policy Action Heatmap: Shows which actions SARSOP policy chooses
# based on sea temperature and current sea lice levels
# ----------------------------
function plos_one_sarsop_dominant_action(parallel_data, config, algo_name::String)

    # Load the SARSOP policy
    policy_path = joinpath(config.policies_dir, "policies_pomdp_mdp.jld2")
    if !isfile(policy_path)
        error("Policy file not found: $policy_path")
    end

    @load policy_path all_policies
    policy_bundle = all_policies[algo_name]
    policy = policy_bundle.policy
    pomdp = policy_bundle.pomdp
    
    # Define temperature and sea lice level ranges
    temp_range = 8.0:0.5:24.0  # Sea temperature range (°C)
    sealice_range = 0.0:0.01:1.0  # Sea lice level range
    
    # Fixed values for sessile and motile lice (as requested)
    fixed_sessile = 0.25  # Constant sessile level
    fixed_motile = 0.25   # Constant motile level
    
    action_coords = Dict(action => Vector{Tuple{Float64, Float64}}() for (action, _) in PLOS_ACTION_STYLE_ORDERED)
    
    for (i, sealice_level) in enumerate(sealice_range)
        for (j, temp) in enumerate(temp_range)
            # Predict next sea lice level using the current state
            pred_adult, pred_motile, pred_sessile = predict_next_abundances(
                sealice_level, fixed_motile, fixed_sessile, temp, config.solver_config.location, config.solver_config.reproduction_rate
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
                
                push!(get!(action_coords, chosen_action, Vector{Tuple{Float64, Float64}}()), coord)
            catch e
                @warn "Could not get action for temp=$temp, sealice=$sealice_level: $e"
                push!(action_coords[NoTreatment], (temp, sealice_level))
            end
        end
    end
    
    # Create the plot
    ax = @pgf Axis(Options(
        :xlabel => "Sea Temperature (°C)",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xmin => first(temp_range),
        :xmax => last(temp_range),
        :ymin => first(sealice_range),
        :ymax => last(sealice_range),
        :width => "14cm",
        :height => "8cm",
        :title_style => PLOS_TITLE_STYLE,
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=2),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.3",
    ))
    
    for (action, style) in PLOS_ACTION_STYLE_ORDERED
        coords = get(action_coords, action, Vector{Tuple{Float64, Float64}}())
        isempty(coords) && continue
        push!(ax,
            Plot(
                Options(
                    :only_marks => nothing,
                    :mark => style.marker,
                    :mark_size => "2.4pt",
                    :color => style.color,
                    "mark options" => style.mark_opts
                ),
                Coordinates(coords)
            )
        )
        push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end
    
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "norway_sarsop_dominant_action.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "norway_sarsop_dominant_action.tex"), ax; include_preamble=false)
    save_transparent_png(joinpath(config.figures_dir, "Plos_One_Plots", "norway_sarsop_dominant_action.pdf"), ax)
    return ax
end


# ----------------------------
# Policy Decision Map: Filled heatmap showing which action the POMDP policy
# chooses at each (lice level, week of year) combination.
# x = adult female sea lice per fish, y = annual week (1–52)
# Colour = action (sequential: NoTreatment → Chemical → Mechanical → Thermal)
# Overlays the season-dependent regulatory limit as a stepped line.
# ----------------------------
const DECISION_MAP_COLORMAP = raw"{actions}{rgb255=(49,104,178) rgb255=(72,179,154) rgb255=(245,185,66) rgb255=(204,51,51)}"
const DECISION_MAP_ACTION_LABELS = ["No Treatment", "Chemical", "Mechanical", "Thermal"]
const DECISION_MAP_ACTION_IDX = Dict(
    NoTreatment => 0, ChemicalTreatment => 1,
    MechanicalTreatment => 2, ThermalTreatment => 3,
)
const DECISION_MAP_SEASON_BOUNDS = [
    (13, 25, "Spring"), (26, 38, "Summer"),
    (39, 51, "Autumn"), (1, 12, "Winter"),
]

function _build_decision_map_axis(config, algo_name::String;
        title::String="",
        include_ylabel::Bool=true,
        include_season_labels::Bool=true,
        include_colorbar::Bool=true,
        axis_width::String="14cm",
        axis_height::String="10cm")

    # Load policy and POMDP
    policy_path = joinpath(config.policies_dir, "policies_pomdp_mdp.jld2")
    isfile(policy_path) || error("Policy file not found: $policy_path")
    @load policy_path all_policies
    policy_bundle = all_policies[algo_name]
    pol = policy_bundle.policy
    pomdp = policy_bundle.pomdp

    # Grid ranges
    lice_display = 0.0:0.02:1.0
    week_range = 1:52
    is_log = pomdp isa SeaLiceLogPOMDP

    # Fixed state dimensions: mid-range biomass, no cooldown
    bi = 3; ci = 1
    n_lice_states = pomdp.n_lice
    n_season = pomdp.n_season
    n_biomass = pomdp.n_biomass
    n_total = n_lice_states * n_season * n_biomass * pomdp.n_cooldown

    # Build grid data for surf plot
    x_vec = Float64[]; y_vec = Float64[]; z_vec = Float64[]

    for week in week_range
        season = week_to_season(week)
        for lice_level in lice_display
            model_val = is_log ? log(max(lice_level, 1e-3)) : lice_level
            li = clamp(encode(pomdp.lice_lindisc, model_val), 1, n_lice_states)
            bvec = zeros(n_total)
            idx = clamp(li + n_lice_states * ((season - 1) + n_season * ((bi - 1) + n_biomass * (ci - 1))), 1, n_total)
            bvec[idx] = 1.0
            chosen_action = try; action(pol, bvec); catch; NoTreatment; end
            push!(x_vec, lice_level)
            push!(y_vec, Float64(week))
            push!(z_vec, Float64(DECISION_MAP_ACTION_IDX[chosen_action]))
        end
    end

    n_lice_cols = length(lice_display)

    opts = Any[
        :xlabel => "Adult Female Lice per Fish",
        :xmin => first(lice_display),
        :xmax => last(lice_display),
        :ymin => 1, :ymax => 52,
        :width => axis_width, :height => axis_height,
        :xlabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        "axis background/.style" => Options("fill" => "white"),
        "view" => "{0}{90}",
        "colormap" => DECISION_MAP_COLORMAP,
        "point meta min" => 0, "point meta max" => 3,
    ]
    if include_ylabel
        push!(opts, :ylabel => "Week of Year")
        push!(opts, :ylabel_style => PLOS_LABEL_STYLE)
    end
    if include_colorbar
        push!(opts, "colorbar" => nothing)
        push!(opts, "colorbar style" => Options(
            "ytick" => "{0, 1, 2, 3}",
            "yticklabels" => "{" * join(DECISION_MAP_ACTION_LABELS, ",") * "}",
            "tick label style" => PLOS_TICK_STYLE,
        ))
    end
    if !isempty(title)
        push!(opts, :title => title)
        push!(opts, :title_style => PLOS_TITLE_STYLE)
    end

    ax = @pgf Axis(Options(opts...))

    push!(ax, Plot3(
        Options("surf" => nothing, "shader" => "flat corner", "mesh/cols" => n_lice_cols),
        Table(x_vec, y_vec, z_vec)
    ))

    if include_season_labels
        for (w_start, w_end, label) in DECISION_MAP_SEASON_BOUNDS
            mid_week = (w_start + w_end) / 2
            push!(ax, @pgf("""\\node[anchor=west, font=\\scriptsize, text=black!70]
                at (axis cs:$(last(lice_display) * 1.02), $(mid_week)) {$(label)};"""))
        end
    end

    return ax
end

function plos_one_policy_decision_map(parallel_data, config, algo_name::String)
    ax = _build_decision_map_axis(config, algo_name)
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "policy_decision_map.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "policy_decision_map.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "policy_decision_map.pdf"), ax)
    return ax
end

function plos_one_lambda_decision_maps(configs, labels, algo_name::String;
        output_dir::String="Quick_Access")
    n = length(configs)
    @assert n == length(labels) "configs and labels must have same length"

    axes = Axis[]
    for (i, (cfg, lbl)) in enumerate(zip(configs, labels))
        ax = _build_decision_map_axis(cfg, algo_name;
            title=lbl,
            include_ylabel=(i == 1),
            include_season_labels=(i == 1),
            include_colorbar=(i == n),
            axis_width="8cm",
            axis_height="8cm",
        )
        push!(axes, ax)
    end

    gp = @pgf GroupPlot({
        group_style = {
            "group size" => "$n by 1",
            "horizontal sep" => "8pt",
            "xlabels at" => "edge bottom",
            "y descriptions at" => "edge left",
        },
    }, axes...)

    mkpath(output_dir)
    PGFPlotsX.save(joinpath(output_dir, "lambda_decision_maps.pdf"), gp)
    PGFPlotsX.save(joinpath(output_dir, "lambda_decision_maps.tex"), gp; include_preamble=false)
    return gp
end


# ----------------------------
# Kalman Filter Trajectory with num_sigmas σ Confidence Band
# Shows ground truth, noisy observations, KF estimate, and num_sigmas σ uncertainty
# ----------------------------
function plot_kalman_filter_trajectory_with_uncertainty(data, algo_name, config, num_sigmas=3)
    
    # Filter the data to only include the algorithm
    data = filter(row -> row.policy == algo_name, data)
    
    # Extract first belief history for given solver
    history = data.history[1]
    
    # Extract beliefs
    beliefs = _unwrap_beliefs(belief_hist(history))
    belief_means, belief_covariances = unpack(beliefs)
    
    # Extract belief variances (diagonal of covariance matrices)
    belief_variances = [diag(belief_covariances[i, :, :]) for i in 1:size(belief_covariances, 1)]
    belief_variances_array = hcat(belief_variances...)'
    
    # Extract states (ground truth)
    states = state_hist(history)
    states_df = DataFrame(
        Adult = [s.Adult for s in states],
        Motile = [s.Motile for s in states],
        Sessile = [s.Sessile for s in states],
        Temperature = [s.Temperature for s in states]
    )
    
    # Extract observations (noisy measurements)
    observations = observation_hist(history)
    observations_df = DataFrame(
        Adult = [o.Adult for o in observations],
        Motile = [o.Motile for o in observations],
        Sessile = [o.Sessile for o in observations],
        Temperature = [o.Temperature for o in observations]
    )
    
    # Only plot Adult female sea lice (index 1)
    i = 1  # Adult index
    
    ticks, labels = plos_time_ticks(config)

    # Create the plot using single axis
    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :title => "Kalman Filter Estimation Error with ($num_sigmas)σ Uncertainty Band",
        :title_style => "color=black",
        :xlabel => "Time of Year",
        :ylabel => "Estimation Error (True - KF Mean)",
        :xlabel_style => "color=black",
        :ylabel_style => "color=black",
        :tick_label_style => "color=black",
        :xmin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=4),
    ))
    
    # Time steps
    time_steps = 1:size(belief_means, 1)
    
    # Calculate estimation error (true - KF mean)
    estimation_error = states_df[:, i] .- belief_means[:, i]
    
    # Calculate num_sigmas * σ confidence band for the error
    error_upper_num_sigmas = num_sigmas .* sqrt.(belief_variances_array[:, i])
    error_lower_num_sigmas = -num_sigmas .* sqrt.(belief_variances_array[:, i])
    
    # Filter out NaN and infinite values
    valid_indices = .!isnan.(estimation_error) .&& .!isnan.(error_upper_num_sigmas) .&& .!isnan.(error_lower_num_sigmas) .&& 
                   .!isinf.(estimation_error) .&& .!isinf.(error_upper_num_sigmas) .&& .!isinf.(error_lower_num_sigmas)
    
    if sum(valid_indices) > 0
        valid_time_steps = time_steps[valid_indices]
        valid_errors = estimation_error[valid_indices]
        valid_upper = error_upper_num_sigmas[valid_indices]
        valid_lower = error_lower_num_sigmas[valid_indices]

        # Create coordinate strings with valid data only
        error_coords = join(["($(t), $(valid_errors[j]))" for (j, t) in enumerate(valid_time_steps)], " ")
        upper_coords = join(["($(t), $(valid_upper[j]))" for (j, t) in enumerate(valid_time_steps)], " ")
        lower_coords = join(["($(t), $(valid_lower[j]))" for (j, t) in enumerate(valid_time_steps)], " ")

        # Add the 3σ confidence band fill
        push!(ax, @pgf("\\addplot[name path=upper3sigma, blue, mark=none, line width=0.5pt, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower3sigma, blue, mark=none, line width=0.5pt, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[blue, fill opacity=0.2] fill between[of=upper3sigma and lower3sigma];"))
        push!(ax, @pgf("\\addlegendentry{3σ Uncertainty Band}"))

        # Add the estimation error line (should be close to zero for good KF performance)
        push!(ax, @pgf("\\addplot[red, mark=none, line width=1.5pt] coordinates {$(error_coords)};"))
        push!(ax, @pgf("\\addlegendentry{Estimation Error}"))
    end

    # Add zero reference line
    push!(ax, @pgf("\\addplot[black, mark=none, line width=1pt, dashed] coordinates {(0, 0) ($(size(belief_means, 1)), 0)};"))
    push!(ax, @pgf("\\addlegendentry{Zero Reference}"))
    
    # Save the plot
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "kalman_filter_trajectory_($num_sigmas)sigma_$(algo_name)_latex.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "kalman_filter_trajectory_($num_sigmas)sigma_$(algo_name)_latex.tex"), ax; include_preamble=false)
    return ax
end


# --- small helpers -----------------------------------------------------------
coords(xs, ys) = join(["($(xs[k]), $(ys[k]))" for k in eachindex(xs)], " ")

function _valid_mask(vs...)
    m = trues(length(first(vs)))
    for v in vs
        m .&= .!isnan.(v) .& .!isinf.(v)
    end
    m
end

# ---------------------------------------------------------------------------
# Improved: two-panel figure (top: levels; bottom: residuals)
# ---------------------------------------------------------------------------
function plot_kalman_filter_belief_trajectory_two_panel(data, algo_name, config)
    # Filter rows
    data = filter(row -> row.policy == algo_name, data)
    @assert nrow(data) > 0 "No rows matching policy=$(algo_name)."

    # Extract first history bundle
    history = data.history[1]

    # Beliefs
    beliefs = _unwrap_beliefs(belief_hist(history))
    belief_means, belief_covariances = unpack(beliefs)

    # Diagonal variances -> std
    belief_vars = [diag(belief_covariances[i, :, :]) for i in 1:size(belief_covariances, 1)]
    belief_vars_mat = hcat(belief_vars...)'          # T × S
    belief_std = sqrt.(belief_vars_mat)

    # States & observations
    states = state_hist(history)
    observations = observation_hist(history)
    actions_raw = collect(action_hist(history))

    states_df = DataFrame(
        Adult = [s.Adult for s in states],
        Motile = [s.Motile for s in states],
        Sessile = [s.Sessile for s in states],
        Temperature = [s.Temperature for s in states],
    )
    obs_df = DataFrame(
        Adult = [o.Adult for o in observations],
        Motile = [o.Motile for o in observations],
        Sessile = [o.Sessile for o in observations],
        Temperature = [o.Temperature for o in observations],
    )

    # Compact action tags for top-of-plot markers
    action_tags = [action_short_label(a) for a in actions_raw]

    # Only plot Adult (index 1)
    i = 1
    t = collect(1:size(belief_means, 1))

    μ  = belief_means[:, i]
    σ  = belief_std[:, i]
    hi = μ .+ 3 .* σ
    lo = μ .- 3 .* σ

    x_state_mask = _valid_mask(states_df[:, i])
    x_obs_mask   = _valid_mask(obs_df[:, i])
    x_bel_mask   = _valid_mask(μ, hi, lo)
    tμ, μv, hiv, lov = t[x_bel_mask], μ[x_bel_mask], hi[x_bel_mask], lo[x_bel_mask]
    ts, sv = findall(x_state_mask), states_df[x_state_mask, i]
    tobs, ov = findall(x_obs_mask),   obs_df[x_obs_mask, i]

    # Residuals (belief - true)
    # Align by index; drop any step where either is invalid
    x_res_mask = _valid_mask(μ, states_df[:, i])
    tr = t[x_res_mask]
    μr = μ[x_res_mask] .- states_df[x_res_mask, i]
    sr = 3 .* σ[x_res_mask]

    # Annotate treatment times (indices with "M" or "Th")
    treat_idx = [k for k in eachindex(action_tags) if !isempty(action_tags[k])]
    treat_lbl = action_tags[treat_idx]

    ticks, labels = plos_time_ticks(config)

    # Figure: groupplot with two rows, shared x
    gp = @pgf GroupPlot(
        {
            group_style = {
                "group size"         = "1 by 2",
                vertical_sep         = "12pt",
                "ylabels at"         = "edge left",
                "xticklabels at"     = "edge bottom",
            },
            width                  = "18cm",
            height                 = "5.2cm",
        }
    )

    # --- TOP: levels ---------------------------------------------------------
    ax1 = @pgf Axis(Options(
        "title" => "Kalman Belief vs. True and Observed (Adult Female Lice)",
        :xtick => ticks,
        :xticklabels => labels,
        "xlabel" => "Time of Year",
        "ylabel" => "Avg. adult female lice / fish",
        "xmin" => 0,
        "ymin" => 0,
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
        "legend style" => plos_top_legend(columns=4),
        "tick label style" => "{/pgf/number format/fixed}",
        "clip marker paths" => true,
    ))

    # 3σ band
    push!(ax1, @pgf("\\addplot[name path=upper, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tμ, hiv))};"))
    push!(ax1, @pgf("\\addplot[name path=lower, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tμ, lov))};"))
    push!(ax1, @pgf("\\addplot[fill opacity=0.25] fill between[of=upper and lower];"))
    push!(ax1, @pgf(raw"\addlegendentry{Belief $\pm 3\sigma$}"))

    # Mean line
    push!(ax1, @pgf("\\addplot[mark=none, thick] coordinates {$(coords(tμ, μv))};"))
    push!(ax1, @pgf("\\addlegendentry{Belief mean}"))

    # True state (connected line for readability)
    if !isempty(ts)
        push!(ax1, @pgf("\\addplot[mark=*, mark size=1.1pt, only marks] coordinates {$(coords(ts, sv))};"))
        push!(ax1, @pgf("\\addlegendentry{True state}"))
    end

    # Observations (hollow)
    if !isempty(tobs)
        push!(ax1, @pgf("\\addplot[mark=o, mark size=1.5pt, only marks] coordinates {$(coords(tobs, ov))};"))
        push!(ax1, @pgf("\\addlegendentry{Observation}"))
    end

    # Regulatory limit line (e.g., 0.5)
    reg_limit = 0.5
    max_time = maximum(t)
    push!(ax1, @pgf("\\addplot[densely dashed] coordinates {(0,$(reg_limit)) ($(max_time),$(reg_limit))};"))
    push!(ax1, @pgf("\\addlegendentry{Reg. limit}"))

    # Treatment annotations (labels at top)
    if !isempty(treat_idx)
        # Place tiny text labels near the top using a fixed y-coordinate
        max_y = maximum([maximum(μ), maximum(sv), maximum(ov)])
        label_y = max_y * 1.05  # 5% above the maximum value
        for k in eachindex(treat_idx)
            push!(ax1, @pgf("""\\node[anchor=south, font=\\scriptsize]
                at (axis cs:$(treat_idx[k]), $(label_y)) {$(treat_lbl[k])};"""))
        end
    end

    push!(gp, ax1)

    # --- BOTTOM: residuals ---------------------------------------------------
    ax2 = @pgf Axis(Options(
        :xtick => ticks,
        :xticklabels => labels,
        "xlabel" => "Time of Year",
        "ylabel" => "Residual (belief − true)",
        "grid"   => "both",
        "major grid style" => "dashed, opacity=0.35",
        "legend style" => plos_top_legend(columns=4),
        "tick label style" => "{/pgf/number format/fixed}",
        "xmin" => 0,
    ))

    if !isempty(tr)
        # residual band (same 3σ width)
        push!(ax2, @pgf("\\addplot[name path=rupper, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tr, μr .+ sr))};"))
        push!(ax2, @pgf("\\addplot[name path=rlower, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tr, μr .- sr))};"))
        push!(ax2, @pgf("\\addplot[fill opacity=0.25] fill between[of=rupper and rlower];"))
        push!(ax2, @pgf(raw"\addlegendentry{Residual $\pm 3\sigma$}"))

        # residual mean
        push!(ax2, @pgf("\\addplot[mark=none, thick] coordinates {$(coords(tr, μr))};"))
        push!(ax2, @pgf("\\addlegendentry{Residual mean}"))
    end

    # Zero reference line
    push!(ax2, @pgf("\\addplot[black!50, densely dashed] coordinates {(0,0) ($(max_time),0)};"))
    push!(ax2, @pgf("\\addlegendentry{Zero reference}"))

    # Repeat treatment markers faintly along top for context
    if !isempty(treat_idx)
        # Calculate a reasonable y-position for the bottom panel
        max_residual = maximum(abs.(μr))
        label_y_bottom = max_residual * 0.8  # 80% of max residual
        for k in eachindex(treat_idx)
            push!(ax2, @pgf("""\\node[anchor=south, font=\\scriptsize, text opacity=0.7]
                at (axis cs:$(treat_idx[k]), $(label_y_bottom)) {$(treat_lbl[k])};"""))
        end
    end

    push!(gp, ax2)

    # Save
    mkpath("Quick_Access")
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    out1 = joinpath(config.figures_dir, "Plos_One_Plots", "2_panel_kalman_filter_belief_trajectory_$(algo_name)_latex.pdf")
    out1_tex = joinpath(config.figures_dir, "Plos_One_Plots", "2_panel_kalman_filter_belief_trajectory_$(algo_name)_latex.tex")
    out2 = joinpath("Quick_Access", "2_panel_kalman_filter_belief_trajectory_$(algo_name)_latex.pdf")
    PGFPlotsX.save(out1, gp)
    PGFPlotsX.save(out1_tex, gp; include_preamble=false)
    PGFPlotsX.save(out2, gp)
    return gp
end


# ----------------------------
# Shows Adult, Sessile, Motile, and Predicted sea lice levels over time with 95% CI bands
# ----------------------------
function plos_one_algo_sealice_levels_over_time(data, config, algo_name)

    data_policy = filter(row -> row.policy == algo_name, data)
    histories = collect(data_policy.history)

    # Calculate mean and 95% CI band for each time step for all sea lice stages
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

        for episode_history in histories
            states = collect(state_hist(episode_history))
            observations = collect(observation_hist(episode_history))

            if t <= length(states) && t <= length(observations)
                push!(step_adult, states[t].Adult)
                push!(step_sessile, states[t].Sessile)
                push!(step_motile, states[t].Motile)
                push!(step_predicted, observations[t].SeaLiceLevel)
            end
        end

        # Calculate mean and 95% CI band for each stage
        for (step_data, mean_vec, ci_lower_vec, ci_upper_vec) in [
            (step_adult, mean_adult, ci_lower_adult, ci_upper_adult),
            (step_sessile, mean_sessile, ci_lower_sessile, ci_upper_sessile),
            (step_motile, mean_motile, ci_lower_motile, ci_upper_motile),
            (step_predicted, mean_predicted, ci_lower_predicted, ci_upper_predicted)
        ]
            if !isempty(step_data)
                mean_level = mean(step_data)
                std_level = _std_with_guard(step_data)
                n_episodes = length(step_data)
                se_level = n_episodes > 0 ? std_level / sqrt(n_episodes) : 0.0
                ci_margin = 1.96 * se_level

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

    ticks, labels = plos_time_ticks(config)
    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => "Avg. Lice per Fish",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 1,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=5),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ))

    stage_series = [
        (; key = :adult,     mean = mean_adult,     lower = ci_lower_adult,     upper = ci_upper_adult),
        (; key = :sessile,   mean = mean_sessile,   lower = ci_lower_sessile,   upper = ci_upper_sessile),
        (; key = :motile,    mean = mean_motile,    lower = ci_lower_motile,    upper = ci_upper_motile),
        (; key = :predicted, mean = mean_predicted, lower = ci_lower_predicted, upper = ci_upper_predicted),
    ]

    for series in stage_series
        style = PLOS_STAGE_STYLE_LOOKUP[series.key]
        valid_indices = .!isnan.(series.mean) .&& .!isnan.(series.lower) .&& .!isnan.(series.upper)
        if any(valid_indices)
            valid_time = time_steps[valid_indices]
            valid_mean = series.mean[valid_indices]
            valid_lower = series.lower[valid_indices]
            valid_upper = series.upper[valid_indices]

            mean_coords = join(["($(valid_time[j]), $(valid_mean[j]))" for j in 1:length(valid_time)], " ")
            upper_coords = join(["($(valid_time[j]), $(valid_upper[j]))" for j in 1:length(valid_time)], " ")
            lower_coords = join(["($(valid_time[j]), $(valid_lower[j]))" for j in 1:length(valid_time)], " ")

            safe_name = replace(style.label, r"[^A-Za-z0-9]" => "")

            push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
            push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
            push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))

            push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.4pt] coordinates {$(mean_coords)};"))
            push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
        end
    end

    ymax_candidates = Float64[]
    for series in stage_series
        valid_upper = series.upper[.!isnan.(series.upper)]
        isempty(valid_upper) && continue
        push!(ymax_candidates, maximum(valid_upper))
    end
    ymax_val = isempty(ymax_candidates) ? nothing : maximum(ymax_candidates)

    _add_reg_limit!(ax, config.simulation_config.steps_per_episode, 0.5; use_legend=true)

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "$(algo_name)_sealice_levels.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "$(algo_name)_sealice_levels.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "$(algo_name)_sealice_levels.pdf"), ax)

    return ax
end

# ----------------------------
# Treatment Distribution Comparison: Shows treatment counts for all policies
# Displays No Treatment, Chemical Treatment, and Thermal Treatment counts
# ----------------------------
function plos_one_treatment_distribution_comparison(parallel_data, config)
    treatment_types  = ["NoTreatment", "MechanicalTreatment", "ChemicalTreatment", "ThermalTreatment"]
    treatment_labels = ["No Treatment", "Mechanical", "Chemical", "Thermal"]

    # Define all policies with colors matching other plots
    policy_info = [
        ("Heuristic_Policy", "teal", "Heuristic"),
        ("NUS_SARSOP_Policy", "blue", "NUS SARSOP"),
        ("Native_SARSOP_Policy", "cyan", "SARSOP"),
        ("VI_Policy", "purple", "VI"),
        ("QMDP_Policy", "violet", "QMDP"),
        ("Random_Policy", "orange", "Random"),
        ("NeverTreat_Policy", "gray", "NeverTreat"),
        ("AlwaysTreat_Policy", "red", "AlwaysTreat")
    ]

    # Compute averages per policy
    treatment_data = Dict{String, Vector{Float64}}()
    for (policy_name, _, _) in policy_info
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)

        if isempty(data_filtered)
            continue
        end

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
                elseif a == MechanicalTreatment
                    c["MechanicalTreatment"] += 1
                elseif a == ChemicalTreatment
                    c["ChemicalTreatment"] += 1
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

    # Create the plot using PGFPlotsX (same style as other plots)
    ax = @pgf Axis(Options(
        :ybar => nothing,
        :bar_width => "8pt",
        :enlarge_x_limits => "0.15",
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Treatment Type",
        :ylabel => "Average Number of Treatments",
        :xlabel_style => "color=black",
        :ylabel_style => "color=black",
        :tick_label_style => "color=black",
        :xtick => "data",
        :xticklabels => "{" * join(treatment_labels, ",") * "}",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=2),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ))

    # Add bars for each policy
    for (policy_name, color, label) in policy_info
        if haskey(treatment_data, policy_name)
            push!(ax, Plot(Options(:fill => color, :draw => "black", :line_width => "0.3pt"),
                          Coordinates(enumerate(treatment_data[policy_name]))))
            push!(ax, LegendEntry(label))
        end
    end

    # Save the plot
    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_distribution.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_distribution.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "north_treatment_distribution.pdf"), ax)

    return ax
end


# ----------------------------
# Helper: add metric lines + CI bands to an existing PGFPlotsX Axis
# Used by both standalone metric plots and the combined 6-panel figure.
# ----------------------------
function _add_metric_lines!(ax, parallel_data, config, compute_step_value;
        policies_to_plot=nothing, show_legend::Bool=true,
        ymin=nothing, ymax=nothing)

    time_steps = 1:config.simulation_config.steps_per_episode
    for (policy_name, style) in _selected_policy_styles(policies_to_plot)
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue
        seeds = unique(data_filtered.seed)
        isempty(seeds) && continue
        caches = _extract_metric_caches(data_filtered, seeds)
        isempty(caches) && continue

        mean_values = fill(NaN, length(time_steps))
        lower_values = fill(NaN, length(time_steps))
        upper_values = fill(NaN, length(time_steps))

        for (idx, t) in enumerate(time_steps)
            step_values = Float64[]
            for cache in caches
                value = compute_step_value(cache, t, config)
                if value !== nothing && !isnan(value)
                    push!(step_values, value)
                end
            end
            if !isempty(step_values)
                n = length(step_values)
                mean_val = mean(step_values)
                std_val = _std_with_guard(step_values)
                se = n > 0 ? std_val / sqrt(n) : 0.0
                margin = 1.96 * se
                mean_values[idx] = mean_val
                lower_values[idx] = mean_val - margin
                upper_values[idx] = mean_val + margin
            end
        end

        valid_mask = .!isnan.(mean_values) .&& .!isnan.(lower_values) .&& .!isnan.(upper_values)
        valid_indices = findall(valid_mask)
        isempty(valid_indices) && continue

        valid_times = [time_steps[i] for i in valid_indices]
        valid_mean = mean_values[valid_indices]
        valid_lower = lower_values[valid_indices]
        valid_upper = upper_values[valid_indices]
        _clamp_values!(valid_mean, ymin, ymax)
        _clamp_values!(valid_lower, ymin, ymax)
        _clamp_values!(valid_upper, ymin, ymax)

        mean_coords = join(["($(valid_times[j]), $(valid_mean[j]))" for j in eachindex(valid_times)], " ")
        upper_coords = join(["($(valid_times[j]), $(valid_upper[j]))" for j in eachindex(valid_times)], " ")
        lower_coords = join(["($(valid_times[j]), $(valid_lower[j]))" for j in eachindex(valid_times)], " ")

        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.4pt] coordinates {$(mean_coords)};"))
        if show_legend
            push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
        end
    end
end


# ----------------------------
# Combined 6-panel time-series figure (6×1 GroupPlot)
# Panels: (a) Reward, (b) Sea lice levels, (c) Treatment cost,
#          (d) Regulatory penalty, (e) Fish disease, (f) Biomass loss
# ----------------------------
function plos_one_combined_metrics_panel(parallel_data, config;
        policies_to_plot=nothing, show_legend::Bool=true)

    ticks, labels = plos_time_ticks(config)

    panels = [
        (; label="(a)", ylabel="Reward\\\\per Step",
           compute=reward_step_value, ymin=nothing, ymax=nothing, reg_limit=false),
        (; label="(b)", ylabel="Adult Female Lice\\\\per Fish",
           compute=((cache, t, cfg) -> begin
               t <= length(cache.states) ? cache.states[t].SeaLiceLevel : nothing
           end),
           ymin=0.0, ymax=0.6, reg_limit=true),
        (; label="(c)", ylabel="Cum. Treatment\\\\Cost (MNOK)",
           compute=cumulative_treatment_cost_value, ymin=0.0, ymax=nothing, reg_limit=false),
        (; label="(d)", ylabel="Cum. Regulatory\\\\Violations",
           compute=cumulative_regulatory_penalty_value, ymin=0.0, ymax=nothing, reg_limit=false),
        (; label="(e)", ylabel="Cum. Welfare\\\\Cost (MNOK)",
           compute=cumulative_welfare_cost_value, ymin=0.0, ymax=nothing, reg_limit=false),
        (; label="(f)", ylabel="Cum. Biomass\\\\Loss (tons)",
           compute=biomass_loss_step_value, ymin=0.0, ymax=nothing, reg_limit=false),
    ]

    n_panels = length(panels)
    gp = @pgf GroupPlot({
        group_style = {
            "group size" => "1 by $n_panels",
            "vertical sep" => "16pt",
            "xlabels at" => "edge bottom",
            "xticklabels at" => "edge bottom",
        },
        width => "18cm",
        height => "6cm",
        "yticklabel style" => "{text width=3em, align=right}",
    })

    for (panel_idx, panel) in enumerate(panels)
        is_last = panel_idx == n_panels
        is_first = panel_idx == 1

        axis_opts = Any[
            :ylabel => panel.ylabel,
            :ylabel_style => "{" * PLOS_LABEL_STYLE * ", align=center}",
            :tick_label_style => PLOS_TICK_STYLE,
            :xmin => 1,
            :xmax => config.simulation_config.steps_per_episode,
            :xtick => ticks,
            :xticklabels => is_last ? labels : String[],
            "axis background/.style" => Options("fill" => "white"),
            "grid" => "both",
            "major grid style" => "dashed, opacity=0.35",
            "scaled y ticks" => "false",
            "yticklabel style" => "{/pgf/number format/fixed, /pgf/number format/precision=2}",
        ]
        if is_last
            push!(axis_opts, :xlabel => "Time of Year")
            push!(axis_opts, :xlabel_style => PLOS_LABEL_STYLE)
        end
        if is_first && show_legend
            push!(axis_opts, "legend style" => plos_top_legend(columns=5))
        end
        if panel.ymin !== nothing
            push!(axis_opts, :ymin => panel.ymin)
        end
        if panel.ymax !== nothing
            push!(axis_opts, :ymax => panel.ymax)
        end

        ax = @pgf Axis(Options(axis_opts...))

        _add_metric_lines!(ax, parallel_data, config, panel.compute;
            policies_to_plot=policies_to_plot,
            show_legend=(is_first && show_legend),
            ymin=panel.ymin, ymax=panel.ymax)

        if panel.reg_limit
            _add_reg_limit!(ax, config.simulation_config.steps_per_episode, 0.5)
        end

        push!(gp, ax)
    end

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "combined_metrics_panel.pdf"), gp)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "combined_metrics_panel.tex"), gp; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "combined_metrics_panel.pdf"), gp)
    return gp
end


# ----------------------------
# Treatment Action Distribution: Grouped bar chart with one group per policy,
# bars colored by treatment type (Mechanical, Chemical, Thermal).
# ----------------------------
function plos_one_treatment_action_distribution(parallel_data, config;
        policies_to_plot=nothing)

    policy_styles = collect(_selected_policy_styles(policies_to_plot))

    # Compute average treatment counts per policy (exclude NoTreatment)
    treatment_actions = [
        (MechanicalTreatment, "Mechanical", "teal!75!black"),
        (ChemicalTreatment,   "Chemical",   "orange!85!black"),
        (ThermalTreatment,    "Thermal",    "red!80!black"),
    ]

    # Collect data: policy_index => [mean_mech, mean_chem, mean_therm]
    policy_labels = String[]
    policy_means = Vector{Vector{Float64}}()  # one per treatment type

    for _ in treatment_actions
        push!(policy_means, Float64[])
    end

    for (policy_name, style) in policy_styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue

        seeds = unique(data_filtered.seed)
        push!(policy_labels, style.label)

        for (ti, (action, _, _)) in enumerate(treatment_actions)
            counts = Float64[]
            for seed in seeds
                data_seed = filter(row -> row.seed == seed, data_filtered)
                isempty(data_seed) && continue
                actions = collect(action_hist(data_seed.history[1]))
                push!(counts, count(==(action), actions))
            end
            push!(policy_means[ti], isempty(counts) ? 0.0 : mean(counts))
        end
    end

    n_policies = length(policy_labels)
    n_policies == 0 && return nothing

    ax = @pgf Axis(Options(
        :ybar => nothing,
        :bar_width => "6pt",
        :enlarge_x_limits => "0.1",
        :width => "18cm",
        :height => "8cm",
        :ylabel => "Average Number of Treatments per Cycle",
        :ylabel_style => PLOS_LABEL_STYLE,
        :xlabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xtick => "data",
        :xticklabels => "{" * join(policy_labels, ",") * "}",
        "x tick label style" => "{rotate=30, anchor=east, font=\\small}",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=3),
        "grid" => "major",
        "major grid style" => "dashed, opacity=0.35",
        :ymin => 0,
    ))

    for (ti, (_, label, color)) in enumerate(treatment_actions)
        push!(ax, Plot(
            Options(:fill => color, :draw => "black!60", :line_width => "0.3pt"),
            Coordinates(collect(zip(0:n_policies-1, policy_means[ti])))
        ))
        push!(ax, LegendEntry(label))
    end

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "treatment_action_distribution.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "treatment_action_distribution.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "treatment_action_distribution.pdf"), ax)
    return ax
end


# ----------------------------
# Economic Comparison: Net profit focused bar chart with cost components.
# Shows net profit as the primary bar per policy, plus cost breakdown bars.
# ----------------------------
function plos_one_economic_comparison(parallel_data, config;
        policies_to_plot=nothing)

    policy_styles = collect(_selected_policy_styles(policies_to_plot))

    # Check that financial columns exist
    col_names = Set(Symbol.(names(parallel_data)))
    if :net_profit_MNOK ∉ col_names
        @warn "Financial columns not found in data; skipping economic comparison plot. Run with high_fidelity_sim=true."
        return nothing
    end

    financial_metrics = [
        (:total_treatment_cost_MNOK,  "Treatment Cost",   "teal!75!black"),
        (:total_regulatory_cost_MNOK, "Regulatory Cost",  "orange!85!black"),
        (:total_biomass_loss_MNOK,    "Biomass Loss",     "red!70!black"),
        (:total_welfare_cost_MNOK,    "Welfare Cost",     "violet!70!black"),
    ]

    policy_labels = String[]
    metric_means = [Float64[] for _ in financial_metrics]
    metric_cis = [Float64[] for _ in financial_metrics]

    for (policy_name, style) in policy_styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue

        push!(policy_labels, style.label)

        for (mi, (col, _, _)) in enumerate(financial_metrics)
            values = data_filtered[:, col]
            mc = mean_and_ci(values)
            push!(metric_means[mi], mc.mean)
            push!(metric_cis[mi], mc.ci)
        end
    end

    n_policies = length(policy_labels)
    n_policies == 0 && return nothing

    ax = @pgf Axis(Options(
        :ybar => nothing,
        :bar_width => "5pt",
        :enlarge_x_limits => "0.08",
        :width => "18cm",
        :height => "8cm",
        :ylabel => "MNOK per Production Cycle",
        :ylabel_style => PLOS_LABEL_STYLE,
        :xlabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xtick => "data",
        :xticklabels => "{" * join(policy_labels, ",") * "}",
        "x tick label style" => "{rotate=30, anchor=east, font=\\small}",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=3),
        "grid" => "major",
        "major grid style" => "dashed, opacity=0.35",
    ))

    for (mi, (_, label, color)) in enumerate(financial_metrics)
        push!(ax, Plot(
            Options(
                :fill => color,
                :draw => "black!60",
                :line_width => "0.3pt",
                "error bars/y dir" => "both",
                "error bars/y explicit" => nothing,
            ),
            Coordinates(
                collect(0:n_policies-1),
                metric_means[mi];
                yerror=metric_cis[mi],
            )
        ))
        push!(ax, LegendEntry(label))
    end

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "economic_comparison.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "economic_comparison.tex"), ax; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "economic_comparison.pdf"), ax)
    return ax
end


# ----------------------------
# Combined Bar Panel: Treatment action distribution (top) + Economic comparison (bottom)
# with shared x-axis (policy names).
# ----------------------------
function plos_one_combined_bar_panel(parallel_data, config;
        policies_to_plot=nothing)

    policy_styles = collect(_selected_policy_styles(policies_to_plot))

    # --- Compute treatment counts per policy ---
    treatment_actions = [
        (MechanicalTreatment, "Mechanical", "teal!75!black"),
        (ChemicalTreatment,   "Chemical",   "orange!85!black"),
        (ThermalTreatment,    "Thermal",    "red!80!black"),
    ]

    policy_labels = String[]
    treatment_means = [Float64[] for _ in treatment_actions]
    treatment_cis = [Float64[] for _ in treatment_actions]

    for (policy_name, style) in policy_styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue

        seeds = unique(data_filtered.seed)
        push!(policy_labels, style.label)

        for (ti, (act, _, _)) in enumerate(treatment_actions)
            counts = Float64[]
            for seed in seeds
                data_seed = filter(row -> row.seed == seed, data_filtered)
                isempty(data_seed) && continue
                actions = collect(action_hist(data_seed.history[1]))
                push!(counts, count(==(act), actions))
            end
            mc = mean_and_ci(counts)
            push!(treatment_means[ti], mc.mean)
            push!(treatment_cis[ti], mc.ci)
        end
    end

    n_policies = length(policy_labels)
    n_policies == 0 && return nothing

    # --- Compute financial metrics per policy ---
    col_names = Set(Symbol.(names(parallel_data)))
    has_financial = :net_profit_MNOK ∈ col_names

    financial_metrics = [
        (:total_treatment_cost_MNOK,  "Treatment Cost",   "blue!60!black"),
        (:total_regulatory_cost_MNOK, "Regulatory Cost",  "yellow!70!black"),
        (:total_biomass_loss_MNOK,    "Biomass Loss",     "purple!60!black"),
        (:total_welfare_cost_MNOK,    "Welfare Cost",     "brown!60!black"),
    ]

    metric_means = [Float64[] for _ in financial_metrics]
    metric_cis = [Float64[] for _ in financial_metrics]

    if has_financial
        for (policy_name, style) in policy_styles
            data_filtered = filter(row -> row.policy == policy_name, parallel_data)
            isempty(data_filtered) && continue

            for (mi, (col, _, _)) in enumerate(financial_metrics)
                values = data_filtered[:, col]
                mc = mean_and_ci(values)
                push!(metric_means[mi], mc.mean)
                push!(metric_cis[mi], mc.ci)
            end
        end
    end

    xticklabel_str = "{" * join(policy_labels, ",") * "}"

    # --- Build GroupPlot ---
    gp = @pgf GroupPlot({
        group_style = {
            "group size" => "1 by 2",
            "vertical sep" => "60pt",
            "xlabels at" => "edge bottom",
            "xticklabels at" => "edge bottom",
        },
        width => "18cm",
        height => "7cm",
    })

    # --- Top panel: Treatment action distribution ---
    ax1 = @pgf Axis(Options(
        :ybar => nothing,
        :bar_width => "6pt",
        :enlarge_x_limits => "0.1",
        :ylabel => "Avg. Treatments\\\\per Cycle",
        :ylabel_style => "{" * PLOS_LABEL_STYLE * ", align=center}",
        :tick_label_style => PLOS_TICK_STYLE,
        :xtick => "data",
        :xticklabels => xticklabel_str,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => Options(
            "fill" => "white", "draw" => "black!40", "text" => "black",
            "font" => PLOS_FONT,
            "at" => "{(0.5,1.08)}", "anchor" => "south",
            "row sep" => "1pt", "column sep" => "0.5cm",
            "legend columns" => "3",
        ),
        "grid" => "major",
        "major grid style" => "dashed, opacity=0.35",
        :ymin => 0,
    ))

    # Treatment bars + legend entries (with error bars)
    for (ti, (_, label, color)) in enumerate(treatment_actions)
        push!(ax1, Plot(
            Options(
                :fill => color, :draw => "black!60", :line_width => "0.3pt",
                "error bars/y dir" => "both",
                "error bars/y explicit" => nothing,
            ),
            Coordinates(
                collect(0:n_policies-1),
                treatment_means[ti];
                yerror=treatment_cis[ti],
            )
        ))
        push!(ax1, LegendEntry(label))
    end
    push!(gp, ax1)

    # --- Bottom panel: Economic comparison (legend between panels) ---
    ax2 = @pgf Axis(Options(
        :ybar => nothing,
        :bar_width => "5pt",
        :enlarge_x_limits => "0.1",
        :ylabel => "Cost\\\\(MNOK)",
        :ylabel_style => "{" * PLOS_LABEL_STYLE * ", align=center}",
        :xlabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xtick => "data",
        :xticklabels => xticklabel_str,
        "x tick label style" => "{rotate=0, anchor=north, font=\\small}",
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => Options(
            "fill" => "white", "draw" => "black!40", "text" => "black",
            "font" => PLOS_FONT,
            "at" => "{(0.5,1.15)}", "anchor" => "south",
            "row sep" => "1pt", "column sep" => "0.5cm",
            "legend columns" => "4",
        ),
        "grid" => "major",
        "major grid style" => "dashed, opacity=0.35",
        :ymin => 0,
    ))

    if has_financial
        for (mi, (_, label, color)) in enumerate(financial_metrics)
            push!(ax2, Plot(
                Options(
                    :fill => color,
                    :draw => "black!60",
                    :line_width => "0.3pt",
                    "error bars/y dir" => "both",
                    "error bars/y explicit" => nothing,
                ),
                Coordinates(
                    collect(0:n_policies-1),
                    metric_means[mi];
                    yerror=metric_cis[mi],
                )
            ))
            push!(ax2, LegendEntry(label))
        end
    end
    push!(gp, ax2)

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "combined_bar_panel.pdf"), gp)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "combined_bar_panel.tex"), gp; include_preamble=false)
    PGFPlotsX.save(joinpath("Quick_Access", "combined_bar_panel.pdf"), gp)
    return gp
end
