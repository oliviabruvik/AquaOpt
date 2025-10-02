using DataFrames
import Distributions: Normal, Uniform
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

# Include shared types first
include("../Utils/SharedTypes.jl")
include("../Utils/Utils.jl")

# -------------------------
# State, Observation, Action
# -------------------------
"State representing the predicted sea lice level the following week."
struct SeaLiceState
	SeaLiceLevel::Float64
end

"Observation representing an observed sea lice level the following week."
struct SeaLiceObservation
	SeaLiceLevel::Float64
end

# Action enum is imported from SharedTypes.jl

# -------------------------
# SeaLiceMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects."
@with_kw struct SeaLiceMDP <: POMDP{SeaLiceState, Action, SeaLiceObservation}

    # Parameters
	lambda::Float64 = 0.5
    reward_lambdas::Vector{Float64} = [0.5, 0.5, 0.0, 0.0] # [treatment, regulatory, biomass, health]
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 0.3
	rho::Float64 = 0.95
    discount_factor::Float64 = 0.95
    full_observability_solver::Bool = false

    # Regulation parameters
    regulation_limit::Float64 = 0.5

    # Parameters from Aldrin et al. 2023
    n_sample::Int = 20                 # number of fish counted (ntc)
    rho_nb::Float64 = 0.175            # aggregation/over-dispersion "ρ" (adult default)
    use_underreport::Bool = false      # toggle logistic under-count correction
    beta0_Scount_f::Float64 = -1.535   # farm intercept for under-count (if used)
    beta1_Scount::Float64 = 0.039      # weight slope for under-count (if used)
    mean_fish_weight_kg::Float64 = 1.5 # mean fish weight
    W0::Float64 = 0.1                  # weight centering (kg)

    # Count bounds
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 5.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 0.25)
    initial_mean::Float64 = 0.13
    
    # Sampling parameters
    adult_sd::Float64 = 0.1
    rng::AbstractRNG = Random.GLOBAL_RNG

    # Discretization
    discretization_step::Float64 = 0.05 # 0.01
    sea_lice_range::Vector{Float64} = collect(sea_lice_bounds[1]:discretization_step:sea_lice_bounds[2])
    initial_range::Vector{Float64} = collect(initial_bounds[1]:discretization_step:initial_bounds[2])
    lindisc::LinearDiscretizer = LinearDiscretizer(collect(sea_lice_bounds[1]:discretization_step:(sea_lice_bounds[2]+discretization_step)))
    catdisc::CategoricalDiscretizer = CategoricalDiscretizer([NoTreatment, Treatment, ThermalTreatment])
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.states(mdp::SeaLiceMDP) = [SeaLiceState(i) for i in mdp.sea_lice_range]
POMDPs.actions(mdp::SeaLiceMDP) = [NoTreatment, Treatment, ThermalTreatment]
POMDPs.observations(mdp::SeaLiceMDP) = [SeaLiceObservation(i) for i in mdp.sea_lice_range]
POMDPs.discount(mdp::SeaLiceMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceMDP, s::SeaLiceState) = false
POMDPs.stateindex(mdp::SeaLiceMDP, s::SeaLiceState) = encode(mdp.lindisc, s.SeaLiceLevel)
POMDPs.actionindex(mdp::SeaLiceMDP, a::Action) = encode(mdp.catdisc, a)
POMDPs.obsindex(mdp::SeaLiceMDP, o::SeaLiceObservation) = encode(mdp.lindisc, o.SeaLiceLevel)

# -------------------------
# Transition
# -------------------------
function POMDPs.transition(mdp::SeaLiceMDP, s::SeaLiceState, a::Action)

    # Apply treatment
    rf_a, rf_m, rf_s = get_treatment_effectiveness(a)
    adult = s.SeaLiceLevel * (1 - rf_a)

    # Calculate the mean of the transition distribution
    μ = exp(mdp.growthRate) * adult

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
function POMDPs.observation(mdp::SeaLiceMDP, a::Action, s::SeaLiceState)

    # Observation grid
    observations = POMDPs.observations(mdp)

    # If full observability solver, we return the exact state as the observation
    # to mimic full observability.
    if mdp.full_observability_solver
        # Set the probs to 1 for the state we are at and 0 for the rest
        observations = POMDPs.observations(mdp)
        probs = zeros(length(observations))
        probs[obsindex(mdp, s)] = 1.0
        return SparseCat(observations, probs)
    end

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
    μ_total_adult = max(1e-12, mdp.n_sample * p_scount * s.SeaLiceLevel)
    k = max(1e-9, mdp.n_sample * mdp.rho_nb)  # NB "size" scales with n_sample
    r, p = nb_params_from_mean_k(μ_total_adult, k)
    nb = NegativeBinomial(r, p)

    # Build CDF at each grid value (threshold on total counts)
    cdfs = similar(observations, Float64)
    @inbounds for (i, o) in enumerate(observations)
        adult_threshold = clamp(o.SeaLiceLevel, mdp.sea_lice_bounds...)
        total_adult_threshold = max(0, floor(Int, mdp.n_sample * adult_threshold))
        ci = cdf(nb, total_adult_threshold)
        if !isfinite(ci)
            @warn "NaN in observation distribution (SeaLicePOMDP)"
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
            @warn "NaN or non-monotonic in observation distribution (SeaLicePOMDP)"
            ci = past
        end
        probs[i] = max(ci - past, 0.0)
        past = ci
    end

    # Normalize (and fallback if degenerate)
    if sum(probs) <= 0 || !isfinite(sum(probs))
        # Put all mass on the nearest bin to the mean of the NB distribution
        @warn "Degenerate observation distribution (SeaLicePOMDP)"
        idx = findmin(abs.([o.SeaLiceLevel - μ_total_adult / mdp.n_sample for o in observations]))[2]
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
function POMDPs.reward(mdp::SeaLiceMDP, s::SeaLiceState, a::Action, sp::SeaLiceState)

    λ_trt, λ_reg, λ_bio, λ_health, λ_sea_lice = mdp.reward_lambdas

    # Treatment cost
    treatment_cost = get_treatment_cost(a)

    # Regulatory penalty
    over_limit = s.SeaLiceLevel > mdp.regulation_limit
    if (over_limit && a == NoTreatment)
        regulatory_penalty = 1000.0
    else
        regulatory_penalty = 15.0
    end
    
    # Lost biomass
    lost_biomass = 0

    # Fish disease
    fish_disease = get_fish_disease(a) + 100.0 * s.SeaLiceLevel

    return - (λ_health * fish_disease + λ_trt * treatment_cost + λ_bio * lost_biomass + λ_reg * regulatory_penalty + λ_sea_lice * s.SeaLiceLevel)
end

# -------------------------
# Initial state
# -------------------------
function POMDPs.initialstate(mdp::SeaLiceMDP)

    # Get the distribution
    dist = truncated(Normal(mdp.initial_mean, mdp.adult_sd), mdp.sea_lice_bounds...)

    # Get the states
    states = POMDPs.states(mdp)

    # Calculate the probs using the cdf
    probs = discretize_distribution(dist, states)

    return SparseCat(states, probs)
end