
using POMDPs
using POMDPModels
using POMDPTools
using Distributions
using Parameters
using GaussianFilters
using DataFrames
using JLD2
using Plots

# ----------------------------
# Create POMDP and MDP
# ----------------------------
function create_pomdp_mdp(config)

    sim_cfg = config.simulation_config
    adult_mean = max(sim_cfg.adult_mean, 1e-6)
    motile_ratio = sim_cfg.motile_mean / adult_mean
    sessile_ratio = sim_cfg.sessile_mean / adult_mean
    base_temperature = get_location_params(config.solver_config.location).T_mean

    sc = config.solver_config
    if sc.log_space
        pomdp = SeaLiceLogPOMDP(
            reward_lambdas=sc.reward_lambdas,
            discount_factor=sc.discount_factor,
            discretization_step=sc.discretization_step,
            adult_sd=sc.adult_sd,
            regulation_limit=sc.regulation_limit,
            season_regulation_limits=sc.season_regulation_limits,
            full_observability_solver=sc.full_observability_solver,
            location=sc.location,
            reproduction_rate=sc.reproduction_rate,
            motile_ratio=motile_ratio,
            sessile_ratio=sessile_ratio,
            base_temperature=base_temperature,
            salmon_price_MNOK_per_tonne=sc.salmon_price_MNOK_per_tonne,
            regulatory_violation_cost_MNOK=sc.regulatory_violation_cost_MNOK,
            welfare_cost_MNOK=sc.welfare_cost_MNOK,
            chronic_lice_cost_MNOK=sc.chronic_lice_cost_MNOK,
        )
    else
        pomdp = SeaLicePOMDP(
            reward_lambdas=sc.reward_lambdas,
            discount_factor=sc.discount_factor,
            discretization_step=sc.discretization_step,
            adult_sd=sc.adult_sd,
            regulation_limit=sc.regulation_limit,
            season_regulation_limits=sc.season_regulation_limits,
            full_observability_solver=sc.full_observability_solver,
            location=sc.location,
            reproduction_rate=sc.reproduction_rate,
            motile_ratio=motile_ratio,
            sessile_ratio=sessile_ratio,
            base_temperature=base_temperature,
            salmon_price_MNOK_per_tonne=sc.salmon_price_MNOK_per_tonne,
            regulatory_violation_cost_MNOK=sc.regulatory_violation_cost_MNOK,
            welfare_cost_MNOK=sc.welfare_cost_MNOK,
            chronic_lice_cost_MNOK=sc.chronic_lice_cost_MNOK,
        )
    end

    mdp = UnderlyingMDP(pomdp)
    return pomdp, mdp
end

# ----------------------------
# Generate MDP and POMDP policies
# ----------------------------
function solve_policies(algorithms, config)

    pomdp, mdp = create_pomdp_mdp(config)

    all_policies = Dict{String, NamedTuple}()

    for algo in algorithms
        @info "Solving policy $(algo.solver_name)"
        policy = generate_policy(algo, pomdp, mdp)
        all_policies[algo.solver_name] = (policy=policy, pomdp=pomdp, mdp=mdp)
    end

    # Save all_policies, pomdp, and mdp to file
    policies_dir = joinpath(config.policies_dir)
    mkpath(policies_dir)
    @save joinpath(policies_dir, "policies_pomdp_mdp.jld2") all_policies pomdp mdp

    return all_policies
end

# ----------------------------
# Policy Generation
# ----------------------------
function generate_policy(algorithm, pomdp, mdp)

    # Heuristic Policy
    if algorithm.solver_name == "Heuristic_Policy"
        return HeuristicPolicy(pomdp, algorithm.solver_config)

    # Random policy
    elseif algorithm.solver_name == "Random_Policy"
        return RandomPolicy(pomdp)

    # No Treatment policy
    elseif algorithm.solver_name == "NeverTreat_Policy"
        return NeverTreatPolicy(pomdp)

    # Always Treat policy
    elseif algorithm.solver_name == "AlwaysTreat_Policy"
        return AlwaysTreatPolicy(pomdp)

    # Value Iteration policy
    elseif algorithm.solver isa ValueIterationSolver
        return solve(algorithm.solver, mdp)

    # SARSOP and QMDP policies
    else
        if algorithm.solver isa SARSOP.SARSOPSolver
            mkpath(dirname(algorithm.solver.policy_filename))
            mkpath(dirname(algorithm.solver.pomdp_filename))
        end
        return solve(algorithm.solver, pomdp)
    end
end

# ----------------------------
# Never Treat Policy
# ----------------------------
struct NeverTreatPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Never Treat action
function POMDPs.action(policy::NeverTreatPolicy, b)
    return NoTreatment
end

function POMDPs.updater(policy::NeverTreatPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Always Treat Policy
# ----------------------------
struct AlwaysTreatPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Always Treat action
function POMDPs.action(policy::AlwaysTreatPolicy, b)
    return MechanicalTreatment
end

function POMDPs.updater(policy::AlwaysTreatPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Random Policy
# ----------------------------
struct RandomPolicy{P<:POMDP} <: Policy
    pomdp::P
end

# Random action
function POMDPs.action(policy::RandomPolicy, b)
    return rand((NoTreatment, MechanicalTreatment, ChemicalTreatment, ThermalTreatment))
end

function POMDPs.updater(policy::RandomPolicy)
    return DiscreteUpdater(policy.pomdp)
end

# ----------------------------
# Heuristic Policy
# ----------------------------
struct HeuristicPolicy{P<:POMDP} <: Policy
    pomdp::P
    solver_config::SolverConfig
end

# Heuristic action
# TODO: add some stochasticity
function POMDPs.action(policy::HeuristicPolicy, b)

    @assert policy.solver_config.heuristic_belief_threshold_thermal >
            policy.solver_config.heuristic_belief_threshold_mechanical >
            policy.solver_config.heuristic_belief_threshold_chemical "Heuristic thresholds must be strictly descending: thermal > mechanical > chemical"

    # Get the probability of the current sea lice level being above the threshold
    state_space = states(policy.pomdp)
    threshold = policy.solver_config.heuristic_threshold
    if policy.pomdp isa SeaLiceLogPOMDP
        threshold = log(threshold)
    end
    prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)

    # If the probability of the current sea lice level being above the threshold is greater than the thermal threshold, choose ThermalTreatment
    if prob_above_threshold > policy.solver_config.heuristic_belief_threshold_thermal
        return ThermalTreatment
    # If the probability of the current sea lice level being above the threshold is greater than the mechanical threshold, choose MechanicalTreatment
    elseif prob_above_threshold > policy.solver_config.heuristic_belief_threshold_mechanical
        return MechanicalTreatment
    # Chemical treatment for lower-level infestations
    elseif prob_above_threshold > policy.solver_config.heuristic_belief_threshold_chemical
        return ChemicalTreatment
    # Otherwise, choose NoTreatment
    else
        return NoTreatment
    end
end

# Function to decide whether we choose the action or randomize
function heuristicChooseAction(policy::HeuristicPolicy, b, use_cdf=true)

    # Convert belief vector to a probability distribution
    state_space = states(policy.pomdp)

    # Convert the threshold in log space if needed
    if policy.pomdp isa SeaLiceLogPOMDP
        threshold = log(policy.solver_config.heuristic_threshold)
    else
        threshold = policy.solver_config.heuristic_threshold
    end

    if use_cdf
        # Method 1: Calculate probability of being above threshold
        prob_above_threshold = sum(b[i] for (i, s) in enumerate(state_space) if s.SeaLiceLevel > threshold)
        return prob_above_threshold > policy.solver_config.heuristic_belief_threshold_mechanical
    else
        # Method 2: Use mode of belief vector
        mode_sealice_level_index = argmax(b)
        mode_sealice_level = state_space[mode_sealice_level_index]
        return mode_sealice_level.SeaLiceLevel > threshold
    end
end

function POMDPs.updater(policy::HeuristicPolicy)
    return DiscreteUpdater(policy.pomdp)
end


# ----------------------------
# Enriched Belief / Updater
# Wraps the Kalman filter updater to carry the last observation and action
# alongside the Gaussian belief, giving the AdaptorPolicy direct access
# to simulator state (season, biomass, cooldown) without estimation.
# ----------------------------
struct EnrichedBelief
    gaussian_belief    # GaussianBelief from KF
    last_observation   # EvaluationObservation or nothing (nothing on first step)
    last_action        # Action or nothing (nothing on first step)
end

# Forward .μ and .Σ to the underlying Gaussian belief so that code
# accessing b.μ[BELIEF_IDX_ADULT] (e.g. Evaluation.jl) works transparently.
function Base.getproperty(b::EnrichedBelief, s::Symbol)
    if s === :μ || s === :Σ
        return getproperty(getfield(b, :gaussian_belief), s)
    else
        return getfield(b, s)
    end
end

struct EnrichedUpdater{U} <: POMDPs.Updater
    kf_updater::U
end

function POMDPs.initialize_belief(up::EnrichedUpdater, dist)
    gb = POMDPs.initialize_belief(up.kf_updater, dist)
    return EnrichedBelief(gb, nothing, nothing)
end

function POMDPs.update(up::EnrichedUpdater, b::EnrichedBelief, a::Action, o)
    gb_new = POMDPs.update(up.kf_updater, b.gaussian_belief, a, o)
    return EnrichedBelief(gb_new, o, a)
end

# ----------------------------
# Adaptor Policy
# Bridges the high-fidelity SimPOMDP (EKF belief) to the 4D solver POMDP.
# Reads season, biomass, and cooldown directly from the EnrichedBelief's
# observation rather than estimating them.
# ----------------------------
struct AdaptorPolicy <: Policy
    lofi_policy::Policy
    pomdp::POMDP
    location::String
    reproduction_rate::Float64
end

# Adaptor action
function POMDPs.action(policy::AdaptorPolicy, b::EnrichedBelief)
    gb = b.gaussian_belief

    # --- 1. Derive season, biomass, cooldown from simulator observation ---
    if b.last_observation !== nothing
        season = week_to_season(b.last_observation.AnnualWeek)
        biomass = biomass_tons(b.last_observation)
        cooldown = (b.last_action !== nothing && b.last_action != NoTreatment) ? 1 : 0
    else
        # First step — no observation yet, use production-start defaults
        season = week_to_season(policy.pomdp.production_start_week)
        biomass = policy.pomdp.biomass_range[1]
        cooldown = 0
    end
    biomass_bin_idx = argmin(abs.(policy.pomdp.biomass_range .- biomass))

    # --- 2. Get lice prediction from EKF belief ---
    pred_adult, pred_motile, pred_sessile = predict_next_abundances(
        gb.μ[BELIEF_IDX_ADULT][1], gb.μ[BELIEF_IDX_MOTILE][1], gb.μ[BELIEF_IDX_SESSILE][1], gb.μ[BELIEF_IDX_TEMPERATURE][1],
        policy.location, policy.reproduction_rate)
    adult_variance = gb.Σ[1,1]

    pred_adult_raw = max(pred_adult, 1e-3)

    # Convert to log space if needed for the policy
    if policy.pomdp isa SeaLiceLogPOMDP
        pred_adult_squared_floored = max(pred_adult_raw^2, 0.0025)
        cv_squared = max(adult_variance, 0.0) / pred_adult_squared_floored
        adult_sd = sqrt(log(1.0 + cv_squared))
        pred_adult_final = log(pred_adult_raw)
    else
        adult_sd = sqrt(max(adult_variance, 0.0))
        pred_adult_final = pred_adult_raw
    end

    # --- 3. Build 4D belief vector ---
    # Only lice dimension has uncertainty; season, biomass, cooldown are point estimates.
    n_lice = policy.pomdp.n_lice
    n_season = policy.pomdp.n_season
    n_biomass = policy.pomdp.n_biomass
    n_total = n_lice * n_season * n_biomass * policy.pomdp.n_cooldown

    lice_probs = discretize_values(Normal(pred_adult_final, adult_sd), policy.pomdp.sea_lice_range)

    bvec = zeros(n_total)
    si = season
    bi = biomass_bin_idx
    for li in 1:n_lice
        if lice_probs[li] > 1e-12
            idx = li + n_lice * ((si - 1) + n_season * ((bi - 1) + n_biomass * cooldown))
            bvec[idx] = lice_probs[li]
        end
    end

    # --- 4. Select action ---
    if policy.lofi_policy isa ValueIterationPolicy
        all_states = states(policy.pomdp)
        best_action = NoTreatment
        best_value = -Inf
        for a in actions(policy.pomdp)
            q_val = sum(bvec[i] * value(policy.lofi_policy, s, a)
                        for (i, s) in enumerate(all_states) if bvec[i] > 1e-12)
            if q_val > best_value
                best_value = q_val
                best_action = a
            end
        end
        return best_action
    end

    # For QMDP/SARSOP: use alpha vector dot product with belief
    return action(policy.lofi_policy, bvec)
end

# ----------------------------
# LOFI Adaptor Policy
# ----------------------------
struct LOFIAdaptorPolicy{LP <: Policy, P <: POMDP} <: Policy
    lofi_policy::LP
    pomdp::P
end

function POMDPs.action(policy::LOFIAdaptorPolicy{<:ValueIterationPolicy}, b)
    all_states = states(policy.pomdp)
    @assert length(b.b) == length(all_states)

    # Belief-weighted Q-values (same principle as QMDP)
    best_action = NoTreatment
    best_value = -Inf
    for a in actions(policy.pomdp)
        q_val = sum(b.b[i] * value(policy.lofi_policy, s, a)
                    for (i, s) in enumerate(all_states))
        if q_val > best_value
            best_value = q_val
            best_action = a
        end
    end
    return best_action
end

# Adaptor action
function POMDPs.action(policy::LOFIAdaptorPolicy, b)
    # Discretize alpha vectors (representation of utility over belief states per action)
    return action(policy.lofi_policy, b)
end

# ----------------------------
# Full Observability Adaptor Policy
# Uses the true EvaluationState to derive all 4 solver dimensions directly.
# ----------------------------
mutable struct FullObservabilityAdaptorPolicy <: Policy
    lofi_policy::Policy
    pomdp::POMDP
    mdp::MDP
    location::String
    reproduction_rate::Float64
    last_action::Action
end

# Convenience constructor matching existing call sites
function FullObservabilityAdaptorPolicy(lofi_policy, pomdp, mdp, location, reproduction_rate)
    return FullObservabilityAdaptorPolicy(lofi_policy, pomdp, mdp, location, reproduction_rate, NoTreatment)
end

# Adaptor action
function POMDPs.action(policy::FullObservabilityAdaptorPolicy, s)

    # Predict the next lice level
    pred_adult, pred_motile, pred_sessile = predict_next_abundances(
        s.Adult, s.Motile, s.Sessile, s.Temperature, policy.location, policy.reproduction_rate)
    pred_adult = max(pred_adult, 1e-3)

    if policy.pomdp isa SeaLiceLogPOMDP
        pred_adult = log(pred_adult)
    end

    # Derive 4D state components from the EvaluationState
    season = week_to_season(s.AnnualWeek)
    biomass = biomass_tons(s)
    biomass_bin_idx = argmin(abs.(policy.pomdp.biomass_range .- biomass))
    biomass_level = policy.pomdp.biomass_range[biomass_bin_idx]
    cooldown = policy.last_action != NoTreatment ? 1 : 0

    # Get next action from policy
    if policy.lofi_policy isa ValueIterationPolicy
        # Point estimate: construct full 4D state
        closest_lice_idx = argmin(abs.(policy.pomdp.sea_lice_range .- pred_adult))
        lice_level = policy.pomdp.sea_lice_range[closest_lice_idx]

        if policy.pomdp isa SeaLiceLogPOMDP
            solver_state = SeaLiceLogState(lice_level, season, biomass_level, cooldown)
        else
            solver_state = SeaLiceState(lice_level, season, biomass_level, cooldown)
        end
        chosen_action = action(policy.lofi_policy, solver_state)
    else
        # For QMDP/SARSOP: build 4D belief vector with lice uncertainty
        n_lice = policy.pomdp.n_lice
        n_season = policy.pomdp.n_season
        n_biomass = policy.pomdp.n_biomass
        n_total = n_lice * n_season * n_biomass * policy.pomdp.n_cooldown

        lice_probs = discretize_values(Normal(pred_adult, policy.pomdp.adult_sd), policy.pomdp.sea_lice_range)
        bvec = zeros(n_total)
        si = season
        bi = biomass_bin_idx
        for li in 1:n_lice
            if lice_probs[li] > 1e-12
                idx = li + n_lice * ((si - 1) + n_season * ((bi - 1) + n_biomass * cooldown))
                bvec[idx] = lice_probs[li]
            end
        end
        chosen_action = action(policy.lofi_policy, bvec)
    end

    policy.last_action = chosen_action
    return chosen_action
end
