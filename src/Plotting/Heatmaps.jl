using Plots
using JLD2
plotlyjs()  # Set the backend to PlotlyJS

# ----------------------------
# Plot 9: Treatment Decision Heatmap
# ----------------------------
function plot_treatment_heatmap(algorithm, config)

    # Get lambda values from config
    lambda_values = config.lambda_values
    
    # Initialize matrix to store treatment decisions
    # Rows = states, Columns = lambda values (transposed from before)
    treatment_matrix = nothing
    y_vals = nothing
    
    policies_dir = joinpath(config.policies_dir, algorithm.solver_name)
    
    for (i, λ) in enumerate(lambda_values)
        # Load policy, pomdp, and mdp for this lambda
        policy_file_path = joinpath(policies_dir, "policy_pomdp_mdp_$(λ)_lambda.jld2")
        
        if !isfile(policy_file_path)
            @warn "Policy file not found at $policy_file_path for λ=$λ"
            continue
        end
        
        @load policy_file_path policy pomdp mdp

        # Create state space grid
        if config.log_space
            states = [SeaLiceLogState(x) for x in pomdp.log_sea_lice_range]
            if y_vals === nothing
                y_vals = [exp(s.SeaLiceLevel) for s in states]  # Convert back to original space for y-axis
            end
        else
            states = [SeaLiceState(x) for x in pomdp.sea_lice_range]
            if y_vals === nothing
                y_vals = [s.SeaLiceLevel for s in states]
            end
        end

        function getAction(s, policy)
            if typeof(policy) <: ValueIterationPolicy
                return action(policy, s)
            else
                # For other POMDP policies, create a deterministic belief vector
                # where the current state has probability 1
                state_space = states
                n_states = length(state_space)
                bvec = zeros(n_states)
                
                # Find the index of the current state
                state_idx = findfirst(st -> abs(st.SeaLiceLevel - s.SeaLiceLevel) < 1e-6, state_space)
                if state_idx !== nothing
                    bvec[state_idx] = 1.0
                else
                    # If exact match not found, find closest state
                    distances = [abs(st.SeaLiceLevel - s.SeaLiceLevel) for st in state_space]
                    state_idx = argmin(distances)
                    bvec[state_idx] = 1.0
                end
                
                return action(policy, bvec)
            end
        end

        ## Get a hundred actions to see percentage of treatment decisions
        function getPercentageOfTreatmentDecisions(policy, s)
            actions = [getAction(s, policy) for _ in 1:100]
            return sum(actions .== Treatment) / 100
        end

        percentage_of_treatment_decisions = [getPercentageOfTreatmentDecisions(policy, s) for s in states]
        
        # Initialize matrix on first iteration
        # Now: Rows = states, Columns = lambda values
        if treatment_matrix === nothing
            treatment_matrix = zeros(length(states), length(lambda_values))
        end
        
        # Store decisions for this lambda (column i, all states)
        treatment_matrix[:, i] = percentage_of_treatment_decisions
    end
    
    if treatment_matrix === nothing
        @warn "No policies found to create heatmap"
        return nothing
    end

    p = heatmap(
        lambda_values,      # x-axis: lambda values
        y_vals,            # y-axis: sea lice levels
        treatment_matrix,  # matrix: states × lambdas
        title="Treatment Decision Heatmap: λ vs Sea Lice Level",
        xlabel="λ (Cost-Benefit Trade-off Parameter)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        colorbar_title="Treatment Decision",
        c=:RdYlBu,  # Red-Yellow-Blue colormap (red=treat, blue=no treat)
        aspect_ratio=:auto
    )

    # Save plot
    plot_dir = joinpath(config.figures_dir, "treatment_heatmaps", algorithm.solver_name)
    mkpath(plot_dir)
    savefig(p, joinpath(plot_dir, "treatment_heatmap_lambda_vs_state.png"))
    return p
end

# ----------------------------
# Plot 10: Simulation-Based Treatment Decision Heatmap
# ----------------------------
function plot_simulation_treatment_heatmap(algorithm, config; use_observations=false, n_bins=50)
    
    # Get lambda values from config
    lambda_values = config.lambda_values
    
    # Initialize matrix to store treatment frequencies
    # Rows = sea lice level bins, Columns = lambda values (transposed from before)
    treatment_freq_matrix = nothing
    bin_centers = nothing
    
    histories_dir = joinpath(config.simulations_dir, algorithm.solver_name)
    histories_filename = "$(algorithm.solver_name)_histories"
    @load joinpath(histories_dir, "$(histories_filename).jld2") histories
    
    for (i, λ) in enumerate(lambda_values)
        try
            # Load simulation histories for this lambda
            histories_lambda = histories[histories.lambda .== λ, :]

            if isempty(histories_lambda)
                @warn "No histories found for λ=$λ"
                continue
            end

            # Get action, state, and reward histories
            action_hists = histories_lambda.action_hists[1]
            state_hists = histories_lambda.state_hists[1]
            measurement_hists = histories_lambda.measurement_hists[1]
            
            # Extract data from all episodes
            all_states = []
            all_actions = []

            for episode in 1:length(state_hists)
                state_hist = state_hists[episode]
                action_hist = action_hists[episode]
                
                # Convert states to sea lice levels
                if use_observations
                    # Use measurements instead of states if requested
                    measurement_hist = measurement_hists[episode]
                    sea_lice_levels = if config.log_space
                        [exp(o.SeaLiceLevel) for o in measurement_hist]
                    else
                        [o.SeaLiceLevel for o in measurement_hist]
                    end
                else
                    # Use actual states
                    sea_lice_levels = if config.log_space
                        [exp(s.SeaLiceLevel) for s in state_hist]
                    else
                        [s.SeaLiceLevel for s in state_hist]
                    end
                end
                
                append!(all_states, sea_lice_levels)
                append!(all_actions, action_hist)
            end
            
            # Create bins for sea lice levels on first iteration
            if bin_centers === nothing
                min_level = minimum(all_states)
                max_level = maximum(all_states)
                # Add small buffer to ensure all data points fit
                min_level = max(0.0, min_level - 0.1)
                max_level = max_level + 0.1
                
                bin_edges = range(min_level, stop=max_level, length=n_bins+1)
                bin_centers = [(bin_edges[j] + bin_edges[j+1]) / 2 for j in 1:n_bins]
                
                # Initialize matrix: Rows = bins, Columns = lambda values
                treatment_freq_matrix = zeros(n_bins, length(lambda_values))
            end
            
            # Bin the data and calculate treatment frequencies
            bin_edges = range(minimum(bin_centers) - (bin_centers[2] - bin_centers[1])/2, 
                            stop=maximum(bin_centers) + (bin_centers[2] - bin_centers[1])/2, 
                            length=n_bins+1)
            
            for bin_idx in 1:n_bins
                # Find states in this bin
                in_bin = (all_states .>= bin_edges[bin_idx]) .& (all_states .< bin_edges[bin_idx+1])
                
                if sum(in_bin) > 0
                    # Calculate treatment frequency for this bin
                    actions_in_bin = all_actions[in_bin]
                    treatment_freq = sum(a == Treatment for a in actions_in_bin) / length(actions_in_bin)
                    treatment_freq_matrix[bin_idx, i] = treatment_freq  # Row = bin, Column = lambda
                else
                    # No data in this bin
                    treatment_freq_matrix[bin_idx, i] = NaN
                end
            end
            
        catch e
            @warn "Failed to process simulation histories for λ=$λ: $e"
            continue
        end
    end
    
    if treatment_freq_matrix === nothing
        @warn "No simulation histories could be loaded to create heatmap"
        return nothing
    end
    
    # Replace NaN values with a neutral color (0.5)
    treatment_freq_matrix[isnan.(treatment_freq_matrix)] .= 0.5

    data_type = use_observations ? "Observations" : "States"
    p = heatmap(
        lambda_values,           # x-axis: lambda values  
        bin_centers,            # y-axis: sea lice level bins
        treatment_freq_matrix,  # matrix: bins × lambdas
        title="Simulation Treatment Frequency: λ vs Sea Lice Level ($data_type)",
        xlabel="λ (Cost-Benefit Trade-off Parameter)",
        ylabel="Sea Lice Level (Avg. Adult Female Lice per Fish)",
        colorbar_title="Treatment Frequency",
        c=:RdYlBu,  # Red-Yellow-Blue colormap
        aspect_ratio=:auto,
        clims=(0, 1)  # Ensure color scale goes from 0 to 1
    )

    # Save plot
    plot_dir = joinpath(config.figures_dir, "simulation_treatment_heatmaps", algorithm.solver_name)
    mkpath(plot_dir)
    filename_suffix = use_observations ? "observations" : "states"
    savefig(p, joinpath(plot_dir, "simulation_treatment_heatmap_$(filename_suffix).png"))
    return p
end