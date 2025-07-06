using GaussianFilters
using POMDPs
using POMDPModels
using POMDPTools
using DataFrames
using JLD2
using Plots
using Distributions
using Parameters

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
            discount_factor=pomdp.discount_factor
        )
    else
        SeaLiceSimMDP(
            lambda=pomdp.lambda,
            costOfTreatment=pomdp.costOfTreatment,
            growthRate=pomdp.growthRate,
            rho=pomdp.rho,
            discount_factor=pomdp.discount_factor
        )
    end

    # Create simulator
    sim = RolloutSimulator(max_steps=config.steps_per_episode)
    updaterStruct = KFUpdater(sim_pomdp, process_noise=config.process_noise, observation_noise=config.observation_noise)
    updater = config.ekf_filter ? updaterStruct.ekf : updaterStruct.ukf

    # Run simulation for each episode
    for _ in 1:config.num_episodes
        
        # Get initial state
        s = rand(initialstate(sim_pomdp))

        # Get initial belief from initial mean and sampling sd
        initial_belief = if typeof(sim_pomdp) <: SeaLiceLogSimMDP
            Normal(sim_pomdp.log_lice_initial_mean, sim_pomdp.sampling_sd)
        else
            Normal(sim_pomdp.sea_lice_initial_mean, sim_pomdp.sampling_sd)
        end

        r_total, action_hist, state_hist, measurement_hist, reward_hist, belief_hist = simulate_helper(sim, sim_pomdp, policy, updater, initial_belief, s)

        # If we are using log space, convert the state history to raw space
        # if typeof(sim_pomdp) <: SeaLiceLogSimMDP
        #     state_hist = [pomdp.SeaLiceLogState(exp(s.SeaLiceLevel)) for s in state_hist]
        #     measurement_hist = [pomdp.SeaLiceLogObservation(exp(o.SeaLiceLevel)) for o in measurement_hist]
        #     # belief_hist = [Normal(exp(b.μ[1]), exp(b.Σ[1,1])) for b in belief_hist]
        # end

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
function simulate_helper(sim::RolloutSimulator, sim_pomdp::POMDP, policy::Policy, updater::Any, initial_belief, s)
    
    # Store histories
    action_hist = []
    state_hist = []
    measurement_hist = []
    reward_hist = []
    belief_hist = []
    disc = 1.0
    r_total = 0.0

    b = initialize_belief(updater, initial_belief)

    step = 1

    while disc > sim.eps && !isterminal(sim_pomdp, s) && step <= sim.max_steps

        # Calculate b as beliefvec from normal distribution
        norm_distr = Normal(b.μ[1], b.Σ[1,1])

        # Generate a belief vector from the normal distribution for POMDP policies
        if typeof(policy) <: ValueIterationPolicy
            a = action(policy, s)
        else
            # For POMDP policies, use the belief state
            state_space = states(policy.pomdp)
            bvec = [pdf(norm_distr, s.SeaLiceLevel) for s in state_space]
            bvec = normalize(bvec, 1)
            a = action(policy, bvec)
        end

        sp, o, r = @gen(:sp,:o,:r)(sim_pomdp, s, a, sim.rng)

        r_total += disc * r

        s = sp

        b = runKalmanFilter(updater, b, a, o)

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