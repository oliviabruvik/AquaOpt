include("../Utils/Utils.jl")

using DataFrames
using JLD2
using POMDPs
using QuickPOMDPs
using POMDPTools
using POMDPModels
using QMDP
using DiscreteValueIteration
using POMDPLinter
using Distributions
using Parameters
using Discretizers
using Random

# -------------------------
# State, Observation, Action
# -------------------------
"State representing the sea lice level in log space."
struct SeaLiceLogState
	SeaLiceLevel::Float64
end

"Observation representing an observed sea lice level in log space."
struct SeaLiceLogObservation
	SeaLiceLevel::Float64
end

"Available actions: NoTreatment, Treatment, or ThermalTreatment."
@enum Action NoTreatment Treatment ThermalTreatment

# -------------------------
# SeaLiceLogMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects in log space."
@with_kw struct SeaLiceLogMDP <: POMDP{SeaLiceLogState, Action, SeaLiceLogObservation}

    # Parameters
	lambda::Float64 = 0.5
    reward_lambdas::Vector{Float64} = [0.5, 0.5, 0.0, 0.0, 0.0] # [treatment, regulatory, biomass, health, sea lice]
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 1.2
	rho::Float64 = 0.7
    discount_factor::Float64 = 0.95

    # Regulation parameters
    regulation_limit::Float64 = 0.5

    # Parameters from Aldrin et al. 2023
    n_sample::Int = 20                 # number of fish counted (ntc)
    ρ_adult::Float64 = 0.175            # aggregation/over-dispersion "ρ" (adult default)
    use_underreport::Bool = false      # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535   # farm intercept for under-count (if used)
    beta1_Scount::Float64 = 0.039      # weight slope for under-count (if used)
    mean_fish_weight_kg::Float64 = 1.5 # mean fish weight
    W0::Float64 = 0.1                  # weight centering (kg)

    # Count bounds
    sea_lice_bounds::Tuple{Float64, Float64} = (log(1e-3), log(30.0))
    initial_bounds::Tuple{Float64, Float64} = (log(1e-3), log(0.25))
    initial_mean::Float64 = log(0.13)

    # Sampling parameters
    adult_sd::Float64 = abs(log(0.1))
    rng::AbstractRNG = Random.GLOBAL_RNG
    
    # Log space discretization
    log_discretization_step::Float64 = 0.1
    initial_range::Vector{Float64} = collect(range(initial_bounds[1], stop=initial_bounds[2], step=log_discretization_step))
    log_sea_lice_range::Vector{Float64} = collect(range(sea_lice_bounds[1], stop=sea_lice_bounds[2], step=log_discretization_step))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, Treatment, ThermalTreatment])
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(mdp::SeaLiceLogMDP) = [SeaLiceLogState(i) for i in mdp.log_sea_lice_range]
POMDPs.actions(mdp::SeaLiceLogMDP) = [NoTreatment, Treatment, ThermalTreatment]
POMDPs.observations(mdp::SeaLiceLogMDP) = [SeaLiceLogObservation(i) for i in mdp.log_sea_lice_range]
POMDPs.discount(mdp::SeaLiceLogMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceLogMDP, s::SeaLiceLogState) = false
POMDPs.actionindex(mdp::SeaLiceLogMDP, a::Action) = encode(mdp.catdisc, a)

# -------------------------
# State and Observation Index
# -------------------------
function POMDPs.stateindex(mdp::SeaLiceLogMDP, s::SeaLiceLogState)
    closest_idx = argmin(abs.(mdp.log_sea_lice_range .- s.SeaLiceLevel))
    return closest_idx
end

function POMDPs.obsindex(mdp::SeaLiceLogMDP, o::SeaLiceLogObservation)
    closest_idx = argmin(abs.(mdp.log_sea_lice_range .- o.SeaLiceLevel))
    return closest_idx
end

# -------------------------
# Transition, Observation, Reward, Initial State
# -------------------------
function POMDPs.transition(mdp::SeaLiceLogMDP, s::SeaLiceLogState, a::Action)

    # Apply treatment
    rf_a, rf_m, rf_s = get_treatment_effectiveness(a)
    adult = s.SeaLiceLevel + log(1 - rf_a)

    # Calculate the mean of the transition distribution
    μ = mdp.growthRate + adult

    # Clamp the mean to the range of the sea lice range
    μ = clamp(μ, mdp.sea_lice_bounds...)

    # Get the distribution
    dist = truncated(Normal(μ, mdp.adult_sd), mdp.sea_lice_bounds...)

    # Get the states
    states = POMDPs.states(mdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states)

    return SparseCat(states, probs)
end

# -------------------------
# Observation function: the current count of adults is measured with a negative binomial
# distribution.
# -------------------------
function POMDPs.observation(mdp::SeaLiceLogMDP, a::Action, s::SeaLiceLogState)

    # Observation grid
    observations = POMDPs.observations(mdp)

    # (Optional) under-counting correction like p^Scount_ftc
    # paper uses (W - 0.1) with W in kg
    p_scount = if mdp.use_underreport
        η = mdp.beta0_Scount_f + mdp.beta1_Scount*(mdp.mean_fish_weight_kg - mdp.W0)
        logistic(η)
    else
        1.0
    end

    # NB parameters on TOTAL counts over n_sample fish
    # mean of total counts = n_sample * p_scount * (true lice per fish)
    raw_adult = exp(s.SeaLiceLevel)
    μ_total_adult = max(1e-12, mdp.n_sample * p_scount * raw_adult)
    k = max(1e-9, mdp.n_sample * mdp.ρ_adult)  # NB "size" scales with n_sample
    r, p = nb_params_from_mean_k(μ_total_adult, k)
    nb = NegativeBinomial(r, p)

    # Build CDF at each grid value (threshold on total counts)
    cdfs = similar(observations, Float64)
    @inbounds for (i, o) in enumerate(observations)
        adult_threshold = clamp(o.SeaLiceLevel, mdp.sea_lice_bounds...)
        raw_adult_threshold = exp(adult_threshold)
        total_adult_threshold = max(0, floor(Int, mdp.n_sample * raw_adult_threshold))
        ci = cdf(nb, total_adult_threshold)
        if !isfinite(ci)
            @warn "NaN in observation distribution (SeaLiceLogPOMDP)"
            ci = 0.0
        end
        cdfs[i] = ci
    end

    # Ensure the last bin captures any tail mass beyond max grid value
    cdfs[end] = 1.0

    # Convert CDFs to bin probabilities
    probs = similar(cdfs)
    past = 0.0
    @inbounds for i in eachindex(cdfs)
        ci = cdfs[i]
        # guard against NaNs or non-monotonic numeric issues
        if !isfinite(ci) || ci < past
            @warn "NaN or non-monotonic in observation distribution (SeaLiceLogPOMDP)"
            ci = past
        end
        probs[i] = max(ci - past, 0.0)
        past = ci
    end

    # Normalize (and fallback if degenerate)
    if sum(probs) <= 0 || !isfinite(sum(probs))
        # Put all mass on the nearest bin to the mean of the NB distribution
        @warn "Degenerate observation distribution (SeaLiceLogPOMDP)"
        μ_adult = log(max(μ_total_adult / mdp.n_sample, 1e-12))
        idx = findmin(abs.([o.SeaLiceLevel - μ_adult for o in observations]))[2]
        probs .= 0.0
        probs[idx] = 1.0
    else
        probs = normalize(probs, 1)
    end

    return SparseCat(observations, probs)

end

# -------------------------
# Reward
# -------------------------
function POMDPs.reward(mdp::SeaLiceLogMDP, s::SeaLiceLogState, a::Action, sp::SeaLiceLogState)

    # λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = mdp.reward_lambdas
    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = [0.8, 0.1, 0.0, 0.3, 0.1]
    # λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = [0.7, 0.2, 0.0, 0.5, 0.5]

    # Treatment cost
    treatment_cost = get_treatment_cost(a)

    # Regulatory penalty
    regulatory_penalty = get_regulatory_penalty(a) * (exp(s.SeaLiceLevel) > mdp.regulation_limit ? 1.0 : 0.0)

    # Lost biomass
    lost_biomass = 0

    # Fish disease
    fish_disease = get_fish_disease(a)

    # Sea lice level
    sea_lice_level = exp(s.SeaLiceLevel)

    return - (λ_trt * treatment_cost + λ_reg * regulatory_penalty + λ_bio * lost_biomass + λ_health * fish_disease + λ_sea_lice * sea_lice_level)
end

# -------------------------
# Initial state
# -------------------------
function POMDPs.initialstate(mdp::SeaLiceLogMDP)

    # Get the distribution
    dist = truncated(Normal(mdp.initial_mean, mdp.adult_sd), mdp.sea_lice_bounds...)

    # Get the states
    states = POMDPs.states(mdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states)

    return SparseCat(states, probs)
end