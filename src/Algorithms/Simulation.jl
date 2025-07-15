using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters
using Statistics
using Base.Sys

include("../../src/Utils/Utils.jl")

# ----------------------------
# Simulation & Evaluation
# ----------------------------
function run_simulation(policy, mdp, pomdp, config, algorithm)

    # Store all histories
    belief_hists = []
    r_total_hists = []
    action_hists = []
    state_hists = []
    measurement_hists = []
    reward_hists = []

    # Create simulator POMDP based on whether we're in log space
    sim_pomdp = if typeof(pomdp) <: SeaLiceLogMDP
        SeaLiceLogSimMDP(
            lambda=pomdp.lambda,
            costOfTreatment=pomdp.costOfTreatment,
            growthRate=pomdp.growthRate,
            rho=pomdp.rho,
            discount_factor=pomdp.discount_factor,
            skew=pomdp.skew
        )
    else
        SeaLiceSimMDP(
            lambda=pomdp.lambda,
            costOfTreatment=pomdp.costOfTreatment,
            growthRate=pomdp.growthRate,
            rho=pomdp.rho,
            discount_factor=pomdp.discount_factor,
            skew=pomdp.skew
        )
    end

    # Create simulator
    sim = RolloutSimulator(max_steps=config.steps_per_episode)
    updaterStruct = KFUpdater(sim_pomdp, process_noise=config.process_noise, observation_noise=config.observation_noise)
    
    # Use discrete updater for skew normal distribution, kalman filter otherwise
    if sim_pomdp.skew
        updater = DiscreteUpdater(sim_pomdp)
    else
        updater = config.ekf_filter ? updaterStruct.ekf : updaterStruct.ukf
    end

    # Run simulation for each episode
    for episode in 1:config.num_episodes

        # Set verbose to true for the first episode
        if episode == 1
            verbose = config.verbose
            step_through = config.step_through
        else
            verbose = false
            step_through = false
        end

        if verbose
            # Print all config values
            println("Config values:")
            println("  num_episodes: $(config.num_episodes)")
            println("  steps_per_episode: $(config.steps_per_episode)")
            println("  log_space: $(config.log_space)")
            println("  skew: $(config.skew)")
            println("  experiment_name: $(config.experiment_name)")
            println("  verbose: $(config.verbose)")
            println("  step_through: $(config.step_through)")
            println("  process_noise: $(config.process_noise)")
            println("  observation_noise: $(config.observation_noise)")
            println("  ekf_filter: $(config.ekf_filter)")
            println("  lambda: $(sim_pomdp.lambda)")
            println("  costOfTreatment: $(sim_pomdp.costOfTreatment)")
            println("  growthRate: $(sim_pomdp.growthRate)")
            println("  rho: $(sim_pomdp.rho)")
            println("  discount_factor: $(sim_pomdp.discount_factor)")
            println("  skew: $(sim_pomdp.skew)")
        end

        # Get initial state
        s = rand(initialstate(sim_pomdp))

        # Get initial belief from initial mean and sampling sd
        initial_belief = if typeof(sim_pomdp) <: SeaLiceLogSimMDP
            Normal(sim_pomdp.log_lice_initial_mean, sim_pomdp.sampling_sd)
        else
            Normal(sim_pomdp.sea_lice_initial_mean, sim_pomdp.sampling_sd)
        end

        r_total, action_hist, state_hist, measurement_hist, reward_hist, belief_hist = simulate_helper(sim, sim_pomdp, policy, updater, initial_belief, s, verbose, step_through)

        push!(r_total_hists, r_total)
        push!(action_hists, action_hist)
        push!(state_hists, state_hist)
        push!(measurement_hists, measurement_hist)
        push!(reward_hists, reward_hist)
        push!(belief_hists, belief_hist)
    end

    # Return averages
    return r_total_hists, action_hists, state_hists, measurement_hists, reward_hists, belief_hists
end

# ----------------------------
# Simulation Helper Function
# ----------------------------
function simulate_helper(sim::RolloutSimulator, sim_pomdp::POMDP, policy::Policy, updater::Any, initial_belief, s, verbose::Bool=false, step_through::Bool=false)
    
    # Store histories
    action_hist = []
    state_hist = []
    measurement_hist = []
    reward_hist = []
    belief_hist = []
    disc = 1.0
    r_total = 0.0

    # Get initial belief
    # b = if sim_pomdp.skew
    #     # norm_distr = SkewNormal(initial_belief.μ[1], initial_belief.Σ[1,1], 2.0)
    #     # bvec = [pdf(initial_belief, s.SeaLiceLevel) for s in states(sim_pomdp)]
    #     # bvec = normalize(bvec, 1)
    #     initialize_belief(updater, initial_belief)
    # else
    #     initialize_belief(updater, initial_belief)
    # end

    b = initialize_belief(updater, initial_belief)

    step = 1

    while disc > sim.eps && !isterminal(sim_pomdp, s) && step <= sim.max_steps

        # Clamp gaussian distribution to the range of the sea lice range
        b.μ[1] = if typeof(sim_pomdp) <: SeaLiceLogSimMDP
            clamp(b.μ[1], sim_pomdp.log_lice_bounds[1], sim_pomdp.log_lice_bounds[2])
        else
            clamp(b.μ[1], sim_pomdp.sea_lice_bounds[1], sim_pomdp.sea_lice_bounds[2])
        end

        # Calculate b as beliefvec from normal distribution
        if sim_pomdp.skew
            norm_distr = SkewNormal(b.μ[1], b.Σ[1,1], 2.0)
        else
            norm_distr = Normal(b.μ[1], b.Σ[1,1])
        end

        # Generate a belief vector from the normal distribution for POMDP policies
        if typeof(policy) <: ValueIterationPolicy
            a = action(policy, s)
        else
            # Discretize alpha vectors (representation of utility over belief states per action)
            state_space = states(policy.pomdp)
            bvec = discretize_distribution(norm_distr, state_space, sim_pomdp.skew)
            a = action(policy, bvec)
        end

        sp, o, r = @gen(:sp,:o,:r)(sim_pomdp, s, a, sim.rng)

        r_total += disc * r

        if sim_pomdp.skew
            b = update(updater, b, a, o)
        else
            b = runKalmanFilter(updater, b, a, o)
        end

        if verbose

            # Convert to raw space if log space
            if typeof(s) <: SeaLiceLogState
                sea_lice_level = round(exp(s.SeaLiceLevel), digits=2)
                measurement = round(exp(o.SeaLiceLevel), digits=2)
                next_state = round(exp(sp.SeaLiceLevel), digits=2)
                belief = round(exp(b.μ[1]), digits=2)
                belief_sd = round(exp(b.Σ[1,1]), digits=2)
                reward = round(r, digits=2)
            else
                sea_lice_level = round(s.SeaLiceLevel, digits=2)
                measurement = round(o.SeaLiceLevel, digits=2)
                next_state = round(sp.SeaLiceLevel, digits=2)
                belief = round(b.μ[1], digits=2)
                belief_sd = round(b.Σ[1,1], digits=2)
                reward = round(r, digits=2)
            end
    
            println(" \n\n Step: $step for lambda: $(sim_pomdp.lambda) and algorithm: $(typeof(policy))")
            println("Current sea lice level: $sea_lice_level")
            println("Belief: $belief, Belief SD: $belief_sd")
            println("Action: $a")
            println("Measurement: $measurement")
            println("Reward: $reward")
            println("Next state: $next_state")

            # Create and display the belief distribution plot
            if step_through
                plot_belief_distribution(s, b, o, a, step, sim_pomdp, policy)
                println("Press enter to continue...")
                readline()
            end
        end

        s = sp
        disc *= discount(sim_pomdp)
        step += 1

        # Update histories
        push!(action_hist, a)
        push!(state_hist, s)
        push!(measurement_hist, o)
        push!(reward_hist, r)
        push!(belief_hist, b)
    
    end

    return r_total, action_hist, state_hist, measurement_hist, reward_hist, belief_hist
end

# ----------------------------
# Calculate Averages
# ----------------------------
function calculate_averages(config, pomdp, action_hists, state_hists, reward_hists)

    total_steps = config.num_episodes * config.steps_per_episode
    total_cost, total_sealice, total_reward = 0.0, 0.0, 0.0

    for i in 1:config.num_episodes
        total_cost += sum(a == Treatment for a in action_hists[i]) * pomdp.costOfTreatment
        # Handle both regular and log space states
        total_sealice += if typeof(state_hists[i][1]) <: SeaLiceLogState
            sum(exp(s.SeaLiceLevel) for s in state_hists[i])
        else
            sum(s.SeaLiceLevel for s in state_hists[i])
        end
        total_reward += sum(reward_hists[i])
    end

    return total_reward / total_steps, total_cost / total_steps, total_sealice / total_steps
end

# ----------------------------
# Plot Belief Distribution
# ----------------------------
function plot_belief_distribution(s, b, o, a, step, sim_pomdp, policy)
    """
    Create a comprehensive plot showing the Gaussian belief distribution and key variables.
    """
    
    # Determine if we're in log space and convert values accordingly
    is_log_space = typeof(s) <: SeaLiceLogState
    
    # Get current state and belief values
    if is_log_space
        current_state = exp(s.SeaLiceLevel)
        belief_mean = exp(b.μ[1])
        belief_std = sqrt(exp(b.Σ[1,1]))
        measurement = exp(o.SeaLiceLevel)
    else
        current_state = s.SeaLiceLevel
        belief_mean = b.μ[1]
        belief_std = sqrt(b.Σ[1,1])
        measurement = o.SeaLiceLevel
    end
    
    # Create range for plotting the Gaussian
    x_range = range(0, belief_mean + 4*belief_std, length=200)
    
    # Create the Gaussian distribution
    if sim_pomdp.skew
        belief_dist = SkewNormal(belief_mean, belief_std, 2.0)
    else
        belief_dist = Normal(belief_mean, belief_std)
    end
    
    # Calculate PDF values
    y_values = pdf.(belief_dist, x_range)
    
    # Get solver type from the policy
    solver_type = typeof(policy).name.name
    
    # Get action text
    action_text = a == Treatment ? "TREATMENT" : "NO TREATMENT"
    
    # Create the main plot
    p = plot(
        x_range, y_values,
        title="$solver_type with λ=$(sim_pomdp.lambda) (Step $step)",
        xlabel="Sea Lice Level ($action_text)",
        ylabel="Probability Density",
        label="Belief Distribution",
        linewidth=2,
        color=:blue,
        grid=true,
        legend=:topright,
        size=(800, 600)
    )
    
    # Add action as subtitle using annotation
    # annotate!(mean(x_range), maximum(y_values) * 1.05, text("Action: $action_text", 12, :black, :center))
    
    # Add vertical lines for key values
    vline!([current_state], label="True State", color=:red, linestyle=:dash, linewidth=2)
    vline!([belief_mean], label="Belief Mean", color=:green, linestyle=:dash, linewidth=2)
    vline!([measurement], label="Measurement", color=:orange, linestyle=:dash, linewidth=2)
    
    # Add confidence intervals
    ci_68_lower = belief_mean - 1.0 * belief_std  # 68% confidence interval
    ci_68_upper = belief_mean + 1.0 * belief_std
    ci_95_lower = belief_mean - 1.96 * belief_std  # 95% confidence interval
    ci_95_upper = belief_mean + 1.96 * belief_std
    
    # Shade confidence intervals
    plot!(x_range, y_values, fillrange=0, fillalpha=0.1, color=:blue, label="")
    plot!([ci_68_lower, ci_68_upper], [maximum(y_values), maximum(y_values)], fillrange=0, fillalpha=0.2, color=:blue, label="68% CI")
    plot!([ci_95_lower, ci_95_upper], [maximum(y_values), maximum(y_values)], fillrange=0, fillalpha=0.1, color=:blue, label="95% CI")
    
    # Set y-axis to start from 0
    ylims!(0, 1.0)
    
    # Create directory for plots if it doesn't exist
    plot_dir = "debug_plots"
    mkpath(plot_dir)
    
    # Save the plot to a file
    plot_filename = joinpath(plot_dir, "step.png")
    savefig(p, plot_filename)
    
    println("Plot saved to: $plot_filename")
    
    # Create/update HTML file for live viewing
    create_live_viewer(step, plot_dir)
    
    return p
end

# ----------------------------
# Create Live Viewer HTML
# ----------------------------
function create_live_viewer(current_step, plot_dir)
    """
    Create an HTML file that automatically refreshes to show the latest plot.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simulation Step Viewer</title>
        <meta http-equiv="refresh" content="1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #333; 
                text-align: center;
                margin-bottom: 20px;
            }
            .plot-container {
                text-align: center;
                margin: 20px 0;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .info {
                background-color: #e8f4fd;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #2196F3;
            }
            .step-info {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                text-align: center;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sea Lice Simulation - Step Viewer</h1>
            <div class="step-info">Current Step: $current_step</div>
            <div class="info">
                <strong>Instructions:</strong> This page automatically refreshes every second. 
                Keep this browser tab open to see the latest plot as you step through the simulation.
                The plot shows the Gaussian belief distribution, true state, measurement, and action taken.
            </div>
            <div class="plot-container">
                <img src="step.png" alt="Step $current_step Plot" onerror="this.style.display='none'">
            </div>
        </div>
    </body>
    </html>
    """
    
    html_filename = joinpath(plot_dir, "live_viewer.html")
    open(html_filename, "w") do io
        write(io, html_content)
    end
    
    if current_step == 1
        println("Live viewer created at: $html_filename")
        println("Opening live viewer in browser...")
        # Open the HTML file in the default browser
        if Sys.isapple()
            run(`open $html_filename`)
        elseif Sys.islinux()
            run(`xdg-open $html_filename`)
        elseif Sys.iswindows()
            run(`start $html_filename`)
        end
    end
end