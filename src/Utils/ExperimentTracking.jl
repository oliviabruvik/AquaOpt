using Dates

# ----------------------------
# Save experiment configuration
# ----------------------------
function save_experiment_config(config::ExperimentConfig)
    config_dir = joinpath(config.experiment_dir, "config")
    mkpath(config_dir)

    # Save jld2 file
    @save joinpath(config_dir, "experiment_config.jld2") config

    # Save txt file
    open(joinpath(config_dir, "experiment_config.txt"), "w") do io
        println(io, "timestamp: $(Dates.now())")
        for field in fieldnames(typeof(config))
            value = getfield(config, field)
            println(io, "$field: $value")
        end
    end
end