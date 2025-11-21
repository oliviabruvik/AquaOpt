using Plots
using JLD2
using GaussianFilters
using Statistics
using DataFrames
using PGFPlotsX
using POMDPTools
using Dates

# Consistent palette + labeling for the Plos One figures
const PLOS_POLICY_STYLE_ORDERED = [
    ("NeverTreat_Policy", (; label = "NeverTreat",   line = "gray!70!black",    fill = "gray!20!white")),
    ("AlwaysTreat_Policy",(; label = "AlwaysTreat",  line = "red!80!black",     fill = "red!20!white")),
    ("Random_Policy",     (; label = "Random",       line = "orange!85!black",  fill = "orange!25!white")),
    ("Heuristic_Policy", (; label = "Heuristic",     line = "teal!75!black",    fill = "teal!25!white")),
    ("QMDP_Policy",       (; label = "QMDP",         line = "magenta!70!black", fill = "magenta!25!white")),
    ("NUS_SARSOP_Policy", (; label = "SARSOP",       line = "blue!80!black",    fill = "blue!25!white")),
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

const PLOS_FONT = "\\small"
const PLOS_LABEL_STYLE = "color=black, font=\\small"
const PLOS_TICK_STYLE = "color=black, font=\\small"
const PLOS_TITLE_STYLE = "color=black, font=\\small"
const PLOS_FILL_OPACITY = 0.5

_std_with_guard(v) = length(v) > 1 ? std(v; corrected=true) : 0.0

function _selected_policy_styles(policies_to_plot)
    if policies_to_plot === nothing
        return PLOS_POLICY_STYLE_ORDERED
    end
    wanted = Set(policies_to_plot)
    filtered = [(name, style) for (name, style) in PLOS_POLICY_STYLE_ORDERED if name in wanted]
    if isempty(filtered)
        @warn "No matching policies found for $(collect(wanted)); defaulting to all policies"
        return PLOS_POLICY_STYLE_ORDERED
    end
    found = Set(first.(filtered))
    missing = setdiff(wanted, found)
    if !isempty(missing)
        @warn "Policies without plotting styles ignored: $(collect(missing))"
    end
    return filtered
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
function plos_one_plot_kalman_filter_belief_trajectory(data, algo_name, config, lambda)

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
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=4),
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
        push!(ax, @pgf("\\addplot[name path=upper, draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower, draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(belief_fill_color), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper and lower];"))
        push!(ax, @pgf("\\addlegendimage{area legend, draw=$(belief_line_color), fill=$(belief_fill_color), fill opacity=$(PLOS_FILL_OPACITY)}"))
        push!(ax, @pgf("\\addlegendentry{Belief ±1σ}"))
        
        # Add the mean line
        push!(ax, @pgf("\\addplot[draw=$(belief_line_color), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        push!(ax, @pgf("\\addlegendentry{Belief mean}"))
    else
        push!(ax, @pgf("\\addlegendimage{area legend, draw=$(belief_line_color), fill=$(belief_fill_color), fill opacity=$(PLOS_FILL_OPACITY)}"))
        push!(ax, @pgf("\\addlegendentry{Belief ±1σ}"))
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
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sarsop_kalman_filter_belief_trajectory.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sarsop_kalman_filter_belief_trajectory.tex"), ax)
    return ax
end

# ----------------------------
# Plot: Sea Lice Levels Over Time - Policy Comparison
# Compares effectiveness of different aquaculture management policies in controlling 
# adult female sea lice populations over time with optional 95% confidence intervals.
# Shows compliance with regulatory limit (0.5 lice/fish) across multiple policies.
# Parameters: show_ci=true/false to toggle confidence interval ribbons
# ----------------------------
function plos_one_sealice_levels_over_time(parallel_data, config; show_ci=true, policies_to_plot=nothing)
    ticks, labels = plos_time_ticks(config)
    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => "Avg. Adult Female Sea Lice per Fish",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 0.6,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=7),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ))

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
                push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
            end
        catch e
            @warn "Could not load results for $policy_name: $e"
        end
    end

    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(config.simulation_config.steps_per_episode),0.5)};"))
    push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sealice_levels_over_time.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_sealice_levels_over_time.tex"), ax)
    PGFPlotsX.save(joinpath("Quick_Access", "north_sealice_levels_over_time.pdf"), ax)
    return ax
end


function plos_one_episode_sealice_levels_over_time(
    parallel_data,
    config;
    episode_id::Int = 1,
    savefig::Bool = true,
    policies_to_plot = nothing
)
    ticks, labels = plos_time_ticks(config)
    policy_styles = collect(_selected_policy_styles(policies_to_plot))
    legend_cols = max(length(policy_styles), 1)

    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => "Adult Female Sea Lice per Fish",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :ymax => 1,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=legend_cols),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ))

    for (policy_name, style) in policy_styles
        try
            data_filtered = filter(row -> row.policy == policy_name, parallel_data)
            seeds = unique(data_filtered.seed)
            isempty(seeds) && continue
            if episode_id > length(seeds)
                @warn "Policy $policy_name has only $(length(seeds)) episodes, skipping"
                continue
            end

            selected_seed = seeds[episode_id]
            episode_df = filter(row -> row.seed == selected_seed, data_filtered)
            isempty(episode_df) && continue

            history = episode_df.history[1]
            states = collect(state_hist(history))
            actions = collect(action_hist(history))

            time_steps = 1:length(states)
            levels = [st.SeaLiceLevel for st in states]
            coords = join(["($(time_steps[i]), $(levels[i]))" for i in eachindex(levels)], " ")

            push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.35pt] coordinates {$coords};"))
            push!(ax, @pgf("\\addlegendentry{$(style.label)}"))

            treatment_steps = findall(a -> a != NoTreatment, actions)
            max_level = isempty(levels) ? 0.0 : maximum(levels)
            label_y = min(max_level + 0.12, 1.15)
            for t in treatment_steps
                push!(ax, @pgf("\\addplot+[only marks, mark=*, mark size=3pt, draw=black!70, fill=$(style.line), line width=0.35pt] coordinates {($(t), $(levels[t]))};"))
                tag = action_short_label(actions[t])
                isempty(tag) && continue
                push!(ax, @pgf("""\\node[anchor=south, font=\\scriptsize, text=black!75]
                    at (axis cs:$(t), $(label_y)) {$(tag)};"""))
            end
        catch e
            @warn "Error plotting $policy_name: $e"
        end
    end

    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(config.simulation_config.steps_per_episode),0.5)};"))
    push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))
    push!(ax, @pgf("\\addlegendimage{only marks, mark=*, mark size=3pt, mark options={draw=black!70, fill=white}}"))
    push!(ax, @pgf("\\addlegendentry{Treatment event}"))

    if savefig
        mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
        mkpath("Quick_Access")
        PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "sealice_episode_$(episode_id).pdf"), ax)
        PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "sealice_episode_$(episode_id).tex"), ax)
        PGFPlotsX.save(joinpath("Quick_Access", "sealice_episode_$(episode_id).pdf"), ax)
    end
    
    return ax
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
        initial_biomass = isempty(states) ? 0.0 : states[1].AvgFishWeight * states[1].NumberOfFish
        push!(caches, (; states, actions, rewards, initial_biomass))
    end
    return caches
end

function _plos_one_metric_plot(parallel_data, config, compute_step_value;
        ylabel, file_suffix, ymin=nothing, ymax=nothing, policies_to_plot=nothing)

    ticks, labels = plos_time_ticks(config)
    axis_pairs = [
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
        "legend style" => plos_top_legend(columns=7),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if ymin !== nothing
        push!(axis_pairs, :ymin => ymin)
    end
    if ymax !== nothing
        push!(axis_pairs, :ymax => ymax)
    end
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

        mean_coords = join(["($(valid_times[j]), $(valid_mean[j]))" for j in eachindex(valid_times)], " ")
        upper_coords = join(["($(valid_times[j]), $(valid_upper[j]))" for j in eachindex(valid_times)], " ")
        lower_coords = join(["($(valid_times[j]), $(valid_lower[j]))" for j in eachindex(valid_times)], " ")

        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.4pt] coordinates {$(mean_coords)};"))
        push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", file_suffix), ax)
    tex_suffix = replace(file_suffix, ".pdf" => ".tex")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", tex_suffix), ax)
    PGFPlotsX.save(joinpath("Quick_Access", file_suffix), ax)
    return ax
end

reward_step_value(cache, t, _) = t <= length(cache.rewards) ? cache.rewards[t] : nothing

function biomass_loss_step_value(cache, t, _)
    if t <= length(cache.states)
        state = cache.states[t]
        current_biomass = state.AvgFishWeight * state.NumberOfFish
        return max(cache.initial_biomass - current_biomass, 0.0) / 1000.0
    end
    return nothing
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

function plos_one_reward_over_time(parallel_data, config; policies_to_plot=nothing)
    _plos_one_metric_plot(
        parallel_data, config, reward_step_value;
        ylabel = "Reward per Step",
        file_suffix = "north_reward_over_time.pdf",
        policies_to_plot = policies_to_plot
    )
end

function plos_one_biomass_loss_over_time(parallel_data, config; policies_to_plot=nothing)
    _plos_one_metric_plot(
        parallel_data, config, biomass_loss_step_value;
        ylabel = "Cumulative Biomass Loss (tons)",
        file_suffix = "north_biomass_loss_over_time.pdf",
        ymin = 0.0,
        ymax = 0.1,
        policies_to_plot = policies_to_plot
    )
end

function plos_one_regulatory_penalty_over_time(parallel_data, config; policies_to_plot=nothing)
    _plos_one_metric_plot(
        parallel_data, config, regulatory_penalty_step_value;
        ylabel = "Penalty Probability",
        file_suffix = "north_regulatory_penalty_over_time.pdf",
        ymin = 0.0,
        ymax = 0.1,
        policies_to_plot = policies_to_plot
    )
end

function plos_one_fish_disease_over_time(parallel_data, config; policies_to_plot=nothing)
    _plos_one_metric_plot(
        parallel_data, config, fish_disease_step_value;
        ylabel = "Fish Disease Penalty",
        file_suffix = "north_fish_disease_over_time.pdf",
        ymin = 0.0,
        policies_to_plot = policies_to_plot
    )
end

function plos_one_treatment_cost_over_time(parallel_data, config; policies_to_plot=nothing)
    _plos_one_metric_plot(
        parallel_data, config, treatment_cost_step_value;
        ylabel = "Treatment Cost per Step",
        file_suffix = "north_treatment_cost_over_time.pdf",
        ymin = 0.0,
        policies_to_plot = policies_to_plot
    )
end


# ----------------------------
# Treatment Probability Over Time: Shows all policies overlaid in a single plot
# Each policy shows the probability of treating (any action that is not NoTreatment)
# ----------------------------
function plos_one_combined_treatment_probability_over_time(parallel_data, config; policies_to_plot=nothing)
    
    # Create a single plot using PGFPlotsX (same style as other plots)
    ticks, labels = plos_time_ticks(config)
    ax = @pgf Axis(Options(
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
        "legend style" => plos_top_legend(columns=7),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ))
    
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
            push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
            
        catch e
            @warn "Could not process policy $policy_name: $e"
        end
    end
    
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_probability_over_time.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_probability_over_time.tex"), ax)
    PGFPlotsX.save(joinpath("Quick_Access", "north_treatment_probability_over_time.pdf"), ax)
    return ax
end

# ----------------------------
# Policy Action Heatmap: Shows which actions SARSOP policy chooses
# based on sea temperature and current sea lice levels
# ----------------------------
function plos_one_sarsop_dominant_action(parallel_data, config, λ=0.6)
    
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
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "norway_sarsop_dominant_action.tex"), ax)
    save_transparent_png(joinpath(config.figures_dir, "Plos_One_Plots", "norway_sarsop_dominant_action.pdf"), ax)
    return ax
end



# ----------------------------
# Kalman Filter Trajectory with 3σ Confidence Band
# Shows ground truth, noisy observations, KF estimate, and 3σ uncertainty
# ----------------------------
function plot_kalman_filter_trajectory_with_uncertainty(data, algo_name, config, lambda)
    
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
        :title => "Kalman Filter Estimation Error with 3σ Uncertainty Band",
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
    
    # Calculate 3σ confidence band for the error
    error_upper_3sigma = 3 .* sqrt.(belief_variances_array[:, i])
    error_lower_3sigma = -3 .* sqrt.(belief_variances_array[:, i])
    
    # Filter out NaN and infinite values
    valid_indices = .!isnan.(estimation_error) .&& .!isnan.(error_upper_3sigma) .&& .!isnan.(error_lower_3sigma) .&& 
                   .!isinf.(estimation_error) .&& .!isinf.(error_upper_3sigma) .&& .!isinf.(error_lower_3sigma)
    
    if sum(valid_indices) > 0
        valid_time_steps = time_steps[valid_indices]
        valid_errors = estimation_error[valid_indices]
        valid_upper = error_upper_3sigma[valid_indices]
        valid_lower = error_lower_3sigma[valid_indices]

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
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "kalman_filter_trajectory_3sigma_$(algo_name)_lambda_$(lambda)_latex.pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "kalman_filter_trajectory_3sigma_$(algo_name)_lambda_$(lambda)_latex.tex"), ax)
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
function plot_kalman_filter_belief_trajectory_two_panel(data, algo_name, config, λ)
    # Filter rows
    data = filter(row -> row.policy == algo_name && row.lambda == λ, data)
    @assert nrow(data) > 0 "No rows matching policy=$(algo_name) and lambda=$(λ)."

    # Extract first history bundle
    history = data.history[1]

    # Beliefs
    beliefs = belief_hist(history)
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
    hi = μ .+ σ
    lo = μ .- σ

    x_state_mask = _valid_mask(states_df[:, i])
    x_obs_mask   = _valid_mask(obs_df[:, i])
    x_bel_mask   = _valid_mask(μ, hi, lo)
    tμ, μv, hiv, lov = t[x_bel_mask], μ[x_bel_mask], hi[x_bel_mask], lo[x_bel_mask]
    ts, sv = findall(x_state_mask), states_df[x_state_mask, i]
    tobs, ov = findall(x_obs_mask),   obs_df[x_obs_mask, i]

    # Residuals (belief - true)
    # Align by index; drop any step where either is invalid
    x_res_mask = _valid_mask(μ, states_df[:, i])
    tr, μr, sr = t[x_res_mask], μ[x_res_mask] .- states_df[x_res_mask, i], σ[x_res_mask]

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

    # 1σ band
    push!(ax1, @pgf("\\addplot[name path=upper, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tμ, hiv))};"))
    push!(ax1, @pgf("\\addplot[name path=lower, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tμ, lov))};"))
    push!(ax1, @pgf("\\addplot[fill opacity=0.25] fill between[of=upper and lower];"))
    push!(ax1, @pgf("\\addlegendentry{Belief ±1σ}"))

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
        # residual band (same σ)
        push!(ax2, @pgf("\\addplot[name path=rupper, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tr, μr .+ sr))};"))
        push!(ax2, @pgf("\\addplot[name path=rlower, mark=none, line width=0.4pt, forget plot] coordinates {$(coords(tr, μr .- sr))};"))
        push!(ax2, @pgf("\\addplot[fill opacity=0.25] fill between[of=rupper and rlower];"))
        push!(ax2, @pgf("\\addlegendentry{Residual ±1σ}"))

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
    out1 = joinpath(config.figures_dir, "Plos_One_Plots", "2_panel_kalman_filter_belief_trajectory_$(algo_name)_lambda_$(λ)_latex.pdf")
    out1_tex = joinpath(config.figures_dir, "Plos_One_Plots", "2_panel_kalman_filter_belief_trajectory_$(algo_name)_lambda_$(λ)_latex.tex")
    out2 = joinpath("Quick_Access", "2_panel_kalman_filter_belief_trajectory_$(algo_name)_lambda_$(λ)_latex.pdf")
    PGFPlotsX.save(out1, gp)
    PGFPlotsX.save(out1_tex, gp)
    PGFPlotsX.save(out2, gp)
    return gp
end


# ----------------------------
# Shows Adult, Sessile, Motile, and Predicted sea lice levels over time with 95% CI
# ----------------------------
function plos_one_algo_sealice_levels_over_time(config, algo_name, lambda_value)

    policy_name = algo_name

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
                std_level = _std_with_guard(step_data)
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

    ticks, labels = plos_time_ticks(config)
    ax = @pgf Axis(Options(
        :width => "18cm",
        :height => "6cm",
        :xlabel => "Time of Year",
        :ylabel => "Avg. Lice per Fish",
        :xlabel_style => PLOS_LABEL_STYLE,
        :ylabel_style => PLOS_LABEL_STYLE,
        :tick_label_style => PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "legend style" => plos_top_legend(columns=7),
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

    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(config.simulation_config.steps_per_episode),0.5)};"))
    push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))

    mkpath(joinpath(config.figures_dir, "Plos_One_Plots"))
    mkpath("Quick_Access")
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "$(algo_name)_sealice_levels_lambda_$(lambda_value).pdf"), ax)
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "$(algo_name)_sealice_levels_lambda_$(lambda_value).tex"), ax)
    PGFPlotsX.save(joinpath("Quick_Access", "$(algo_name)_sealice_levels_lambda_$(lambda_value).pdf"), ax)

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
        ("NUS_SARSOP_Policy", "blue", "SARSOP"),
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
    PGFPlotsX.save(joinpath(config.figures_dir, "Plos_One_Plots", "north_treatment_distribution.tex"), ax)
    PGFPlotsX.save(joinpath("Quick_Access", "north_treatment_distribution.pdf"), ax)

    return ax
end
