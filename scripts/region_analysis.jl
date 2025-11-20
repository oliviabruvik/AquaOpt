#!/usr/bin/env julia

using AquaOpt
using DataFrames
using JLD2
using PGFPlotsX
using PGFPlotsX: Axis, GroupPlot, Options, Plot, @pgf
using Statistics
using POMDPTools: state_hist, action_hist, reward_hist, observation_hist

struct RegionInput
    name::String
    experiment_path::String
end

struct RegionData
    name::String
    config::ExperimentConfig
    parallel_data::DataFrame
end

function usage()
    println("Usage: julia --project scripts/region_analysis.jl --west PATH --north PATH --south PATH [--output-dir DIR]")
end

function parse_args(args)
    west = nothing
    north = nothing
    south = nothing
    output_dir::Union{Nothing,String} = nothing
    i = 1
    while i <= length(args)
        arg = strip(args[i])
        if isempty(arg)
            i += 1
            continue
        end
        if arg == "--west"
            i += 1
            west = args[i]
        elseif arg == "--north"
            i += 1
            north = args[i]
        elseif arg == "--south"
            i += 1
            south = args[i]
        elseif arg in ("-o", "--output-dir")
            i += 1
            output_dir = args[i]
        elseif arg in ("-h", "--help")
            usage()
            exit()
        else
            usage()
            error("Unknown argument: $arg")
        end
        i += 1
    end
    if any(isnothing, (west, north, south))
        usage()
        error("Please provide --west, --north, and --south experiment directories.")
    end
    output_dir = isnothing(output_dir) ? "region_outputs" : output_dir
    return (
        [RegionInput("West", west),
         RegionInput("North", north),
         RegionInput("South", south)],
        output_dir
    )
end

function adjust_config_paths!(config, experiment_root::String)
    config = deepcopy(config)
    config.experiment_dir = experiment_root
    config.policies_dir = joinpath(experiment_root, "policies")
    config.simulations_dir = joinpath(experiment_root, "simulation_histories")
    config.results_dir = joinpath(experiment_root, "avg_results")
    config.figures_dir = joinpath(experiment_root, "figures")
    return config
end

function load_experiment_config(experiment_root::String)
    cfg_path = joinpath(experiment_root, "config", "experiment_config.jld2")
    isfile(cfg_path) || error("Could not find config file at $cfg_path")
    @load cfg_path config
    return adjust_config_paths!(config, experiment_root)
end

function ensure_dataframe(data)
    df = data isa DataFrame ? data : DataFrame(data)
    rename!(df, Symbol.(names(df)))
    return df
end

function load_parallel_data(experiment_root::String)
    data_path = joinpath(experiment_root, "simulation_histories", "all_policies_simulation_data.jld2")
    isfile(data_path) || error("Could not find simulation data at $data_path")
    @load data_path data
    return ensure_dataframe(data)
end

function load_region(region::RegionInput)
    path = abspath(region.experiment_path)
    config = load_experiment_config(path)
    data = load_parallel_data(path)
    return RegionData(region.name, config, data)
end

function compute_sealice_stats(parallel_data, config)
    stats = Dict{String, Tuple{Vector{Int}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    styles = AquaOpt.PLOS_POLICY_STYLE_ORDERED
    time_steps = 1:config.simulation_config.steps_per_episode
    for (policy_name, _) in styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue
        seeds = unique(data_filtered.seed)
        isempty(seeds) && continue
        mean_vals = fill(NaN, length(time_steps))
        lower_vals = similar(mean_vals)
        upper_vals = similar(mean_vals)
        for (idx, t) in enumerate(time_steps)
            samples = Float64[]
            for seed in seeds
                seed_df = filter(row -> row.seed == seed, data_filtered)
                isempty(seed_df) && continue
                states = collect(state_hist(seed_df.history[1]))
                if t <= length(states)
                    push!(samples, states[t].SeaLiceLevel)
                end
            end
            if !isempty(samples)
                mean_level = mean(samples)
                std_level = length(samples) > 1 ? std(samples) : 0.0
                se_level = std_level / sqrt(length(samples))
                margin = 1.96 * se_level
                mean_vals[idx] = mean_level
                lower_vals[idx] = mean_level - margin
                upper_vals[idx] = mean_level + margin
            end
        end
        valid = .!isnan.(mean_vals) .&& .!isnan.(lower_vals) .&& .!isnan.(upper_vals)
        if any(valid)
            stats[policy_name] = (collect(time_steps[valid]),
                                  mean_vals[valid],
                                  lower_vals[valid],
                                  upper_vals[valid])
        end
    end
    return stats
end

function extract_metric_caches(data_filtered, seeds)
    caches = NamedTuple[]
    for seed in seeds
        seed_df = filter(row -> row.seed == seed, data_filtered)
        isempty(seed_df) && continue
        history = seed_df.history[1]
        states = collect(state_hist(history))
        actions = collect(action_hist(history))
        rewards = collect(reward_hist(history))
        initial_biomass = isempty(states) ? 0.0 : states[1].AvgFishWeight * states[1].NumberOfFish
        push!(caches, (; states, actions, rewards, initial_biomass))
    end
    return caches
end

function compute_treatment_cost_stats(parallel_data, config)
    stats = Dict{String, Tuple{Vector{Int}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}()
    styles = AquaOpt.PLOS_POLICY_STYLE_ORDERED
    time_steps = 1:config.simulation_config.steps_per_episode
    for (policy_name, _) in styles
        data_filtered = filter(row -> row.policy == policy_name, parallel_data)
        isempty(data_filtered) && continue
        seeds = unique(data_filtered.seed)
        isempty(seeds) && continue
        caches = extract_metric_caches(data_filtered, seeds)
        isempty(caches) && continue
        means = fill(NaN, length(time_steps))
        lowers = similar(means)
        uppers = similar(means)
        for (idx, t) in enumerate(time_steps)
            values = Float64[]
            for cache in caches
                if t <= length(cache.actions)
                    push!(values, get_treatment_cost(cache.actions[t]))
                end
            end
            if !isempty(values)
                mean_val = mean(values)
                std_val = length(values) > 1 ? std(values) : 0.0
                se = std_val / sqrt(length(values))
                margin = 1.96 * se
                means[idx] = mean_val
                lowers[idx] = mean_val - margin
                uppers[idx] = mean_val + margin
            end
        end
        valid = .!isnan.(means) .&& .!isnan.(lowers) .&& .!isnan.(uppers)
        if any(valid)
            stats[policy_name] = (collect(time_steps[valid]),
                                  means[valid],
                                  lowers[valid],
                                  uppers[valid])
        end
    end
    return stats
end

function compute_sarsop_stage_stats(region::RegionData, lambda_value::Float64)
    policy = "NUS_SARSOP_Policy"
    histories_dir = joinpath(region.config.simulations_dir, policy, "$(policy)_histories.jld2")
    @load histories_dir histories
    histories_lambda = histories[lambda_value]
    steps = region.config.simulation_config.steps_per_episode
    stages = Dict(
        :adult => (Float64[], Float64[], Float64[]),
        :sessile => (Float64[], Float64[], Float64[]),
        :motile => (Float64[], Float64[], Float64[]),
        :predicted => (Float64[], Float64[], Float64[])
    )
    for t in 1:steps
        samples = Dict(
            :adult => Float64[],
            :sessile => Float64[],
            :motile => Float64[],
            :predicted => Float64[],
        )
        for episode in histories_lambda
            states = collect(state_hist(episode))
            observations = collect(observation_hist(episode))
            if t <= length(states)
                push!(samples[:adult], states[t].Adult)
                push!(samples[:sessile], states[t].Sessile)
                push!(samples[:motile], states[t].Motile)
            end
            if t <= length(observations)
                push!(samples[:predicted], observations[t].SeaLiceLevel)
            end
        end
        for (stage, data) in samples
            mean_vec, lower_vec, upper_vec = stages[stage]
            if !isempty(data)
                mean_val = mean(data)
                std_val = length(data) > 1 ? std(data) : 0.0
                se = std_val / sqrt(length(data))
                margin = 1.96 * se
                push!(mean_vec, mean_val)
                push!(lower_vec, mean_val - margin)
                push!(upper_vec, mean_val + margin)
            else
                push!(mean_vec, NaN)
                push!(lower_vec, NaN)
                push!(upper_vec, NaN)
            end
        end
    end
    return stages
end

function time_ticks(config)
    return AquaOpt.plos_time_ticks(config)
end

function region_axis(region::RegionData, stats; ylabel::String, show_xlabel::Bool, show_legend::Bool)
    ticks, labels = time_ticks(region.config)
    option_pairs = [
        :width => "15cm",
        :height => "4.5cm",
        :title => region.name,
        :ylabel => ylabel,
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_xlabel
        push!(option_pairs, :xlabel => "Time of Year")
    end
    if show_legend
        push!(option_pairs, "legend style" => AquaOpt.plos_top_legend(columns=length(AquaOpt.PLOS_POLICY_STYLE_ORDERED)))
    else
        push!(option_pairs, :legend => false)
    end
    ax = Axis(Options(option_pairs...))
    for (policy_name, style) in AquaOpt.PLOS_POLICY_STYLE_ORDERED
        haskey(stats, policy_name) || continue
        times, mean_vals, lower_vals, upper_vals = stats[policy_name]
        mean_coords = join(["($(times[j]), $(mean_vals[j]))" for j in eachindex(times)], " ")
        upper_coords = join(["($(times[j]), $(upper_vals[j]))" for j in eachindex(times)], " ")
        lower_coords = join(["($(times[j]), $(lower_vals[j]))" for j in eachindex(times)], " ")
        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(AquaOpt.PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        show_legend && push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end
    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(region.config.simulation_config.steps_per_episode),0.5)};"))
    show_legend && push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))
    return ax
end

function treatment_cost_axis(region::RegionData, stats; show_xlabel::Bool, show_legend::Bool)
    ticks, labels = time_ticks(region.config)
    option_pairs = [
        :width => "15cm",
        :height => "4.5cm",
        :title => region.name,
        :ylabel => "Treatment Cost per Step",
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_xlabel
        push!(option_pairs, :xlabel => "Time of Year")
    end
    if show_legend
        push!(option_pairs, "legend style" => AquaOpt.plos_top_legend(columns=length(AquaOpt.PLOS_POLICY_STYLE_ORDERED)))
    else
        push!(option_pairs, :legend => false)
    end
    ax = Axis(Options(option_pairs...))
    for (policy_name, style) in AquaOpt.PLOS_POLICY_STYLE_ORDERED
        haskey(stats, policy_name) || continue
        times, mean_vals, lower_vals, upper_vals = stats[policy_name]
        mean_coords = join(["($(times[j]), $(mean_vals[j]))" for j in eachindex(times)], " ")
        upper_coords = join(["($(times[j]), $(upper_vals[j]))" for j in eachindex(times)], " ")
        lower_coords = join(["($(times[j]), $(lower_vals[j]))" for j in eachindex(times)], " ")
        safe_name = replace(policy_name, r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(ax, @pgf("\\addplot[forget plot, fill=$(style.fill), fill opacity=$(AquaOpt.PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];"))
        push!(ax, @pgf("\\addplot[color=$(style.line), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        show_legend && push!(ax, @pgf("\\addlegendentry{$(style.label)}"))
    end
    return ax
end

function sarsop_axis(region::RegionData, stats; ylabel::String, show_xlabel::Bool, show_legend::Bool)
    ticks, labels = time_ticks(region.config)
    option_pairs = [
        :width => "15cm",
        :height => "4.5cm",
        :title => region.name,
        :ylabel => ylabel,
        :xlabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :ylabel_style => AquaOpt.PLOS_LABEL_STYLE,
        :tick_label_style => AquaOpt.PLOS_TICK_STYLE,
        :xmin => 0,
        :xmax => region.config.simulation_config.steps_per_episode,
        :ymin => 0,
        :xtick => ticks,
        :xticklabels => labels,
        "axis background/.style" => Options("fill" => "white"),
        "grid" => "both",
        "major grid style" => "dashed, opacity=0.35",
    ]
    if show_xlabel
        push!(option_pairs, :xlabel => "Time of Year")
    end
    legend_entries = [
        (:adult, "Adult (true)"),
        (:sessile, "Sessile"),
        (:motile, "Motile"),
        (:predicted, "Belief (predicted)")
    ]
    if show_legend
        push!(option_pairs, "legend style" => AquaOpt.plos_top_legend(columns=length(legend_entries)))
    else
        push!(option_pairs, :legend => false)
    end
    ax = Axis(Options(option_pairs...))
    colors = Dict(
        :adult => "blue!80!black",
        :sessile => "purple!70!black",
        :motile => "teal!70!black",
        :predicted => "black!70"
    )
    for (key, label) in legend_entries
        mean_vals, lower_vals, upper_vals = stats[key]
        times = collect(1:length(mean_vals))
        valid = .!isnan.(mean_vals) .&& .!isnan.(lower_vals) .&& .!isnan.(upper_vals)
        any(valid) || continue
        t = times[valid]
        μ = mean_vals[valid]
        lo = lower_vals[valid]
        hi = upper_vals[valid]
        mean_coords = join(["($(t[i]), $(μ[i]))" for i in eachindex(t)], " ")
        upper_coords = join(["($(t[i]), $(hi[i]))" for i in eachindex(t)], " ")
        lower_coords = join(["($(t[i]), $(lo[i]))" for i in eachindex(t)], " ")
        safe_name = replace(String(label), r"[^A-Za-z0-9]" => "")
        push!(ax, @pgf("\\addplot[name path=upper$(safe_name), draw=none, forget plot] coordinates {$(upper_coords)};"))
        push!(ax, @pgf("\\addplot[name path=lower$(safe_name), draw=none, forget plot] coordinates {$(lower_coords)};"))
        push!(
            ax,
            @pgf("\\addplot[forget plot, fill=$(colors[key])!40!white, fill opacity=$(AquaOpt.PLOS_FILL_OPACITY)] fill between[of=upper$(safe_name) and lower$(safe_name)];")
        )
        push!(ax, @pgf("\\addplot[color=$(colors[key]), mark=none, line width=1.2pt] coordinates {$(mean_coords)};"))
        show_legend && push!(ax, @pgf("\\addlegendentry{$label}"))
    end
    push!(ax, @pgf("\\addplot[black!70, densely dashed, line width=1pt] coordinates {(0,0.5) ($(region.config.simulation_config.steps_per_episode),0.5)};"))
    show_legend && push!(ax, @pgf("\\addlegendentry{Reg. limit (0.5)}"))
    return ax
end

function build_group_plot(axes::Vector{Axis})
    gp = GroupPlot(Options(
        "group style" => Options(
            "group size" => "1 by $(length(axes))",
            "vertical sep" => "12pt"
        ),
        :width => "18cm"
    ))
    for ax in axes
        push!(gp, ax)
    end
    return gp
end

function save_output(gp, out_pdf::String; save_tex::Bool=false)
    mkpath(dirname(out_pdf))
    PGFPlotsX.save(out_pdf, gp)
    if save_tex
        PGFPlotsX.save(replace(out_pdf, ".pdf" => ".tex"), gp)
    end
end

function main()
    regions_input, out_dir = parse_args(ARGS)
    regions = [load_region(r) for r in regions_input]

    sealice_stats = [compute_sealice_stats(r.parallel_data, r.config) for r in regions]
    treatment_stats = [compute_treatment_cost_stats(r.parallel_data, r.config) for r in regions]
    sarsop_stats = [compute_sarsop_stage_stats(r, 0.6) for r in regions]

    axes_sealice = Axis[]
    axes_cost = Axis[]
    axes_sarsop = Axis[]

    for (idx, region) in enumerate(regions)
        push!(axes_sealice, region_axis(region, sealice_stats[idx];
            ylabel = idx == 1 ? "Adult Female Sea Lice per Fish" : "",
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))

        push!(axes_cost, treatment_cost_axis(region, treatment_stats[idx];
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))

        push!(axes_sarsop, sarsop_axis(region, sarsop_stats[idx];
            ylabel = idx == 1 ? "Avg. Lice per Fish" : "",
            show_xlabel = idx == length(regions),
            show_legend = idx == 1))
    end

    mkpath(out_dir)
    save_output(build_group_plot(axes_sealice), joinpath(out_dir, "region_sealice_levels_over_time.pdf"))
    save_output(build_group_plot(axes_cost), joinpath(out_dir, "region_treatment_cost_over_time.pdf"), save_tex=true)
    save_output(build_group_plot(axes_sarsop), joinpath(out_dir, "region_sarsop_sealice_stages_lambda_0.6.pdf"))

    println("Region plots saved under $(abspath(out_dir)).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
