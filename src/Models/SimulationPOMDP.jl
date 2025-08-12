include("../Utils/Utils.jl")

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
using Random

# -------------------------
# State, Observation, Action
# -------------------------
"State representing the predicted sea lice level the following week and the current sea lice level."
struct EvaluationState
	SeaLiceLevel::Float64 # The predicted adult sea lice level the following week (without treatment)
    Adult::Float64 # The adult sea lice level this week
    Motile::Float64 # The motile sea lice level this week
    Sessile::Float64 # The sessile sea lice level this week
    Temperature::Float64 # The mean temperature (°C) over the last 7 days at the farm (based on approximately daily measurements)
    ProductionWeek::Int64 # The number of weeks since production start
    AnnualWeek::Int64 # The week of the year
    NumberOfFish::Int64 # The number of fish in the pen
    Salinity::Float64 # The salinity of the water (psu)
end

"Observation representing the predicted observed sea lice level the following week and the current observed sea lice level."
struct EvaluationObservation
	SeaLiceLevel::Float64 # The observed predicted adult sea lice level the following week (without treatment)
    Adult::Float64 # The observed adult sea lice level this week
    Motile::Float64 # The observed motile sea lice level this week
    Sessile::Float64 # The observed sessile sea lice level this week
    Temperature::Float64 # The observed mean temperature (°C) over the last 7 days at the farm (based on approximately daily measurements)
    ProductionWeek::Int64 # The number of weeks since production start
    AnnualWeek::Int64 # The week of the year
    NumberOfFish::Int64 # The number of fish in the pen
    Salinity::Float64 # The salinity of the water (psu)
end

"Available actions: NoTreatment or Treatment."
@enum Action NoTreatment Treatment

# -------------------------
# SimulationPOMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects."
@with_kw struct SeaLiceSimMDP <: POMDP{EvaluationState, Action, EvaluationObservation}
	lambda::Float64 = 0.5
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 0.3
	rho::Float64 = 0.95
    discount_factor::Float64 = 0.95
    skew::Bool = false

    # Sea lice bounds
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 30.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 0.25)

    # Means
    adult_mean::Float64 = 0.125
    sessile_mean::Float64 = 0.1
    motile_mean::Float64 = 0.1

    # Transition and Observation Noise
    adult_sd::Float64 = 0.1
    motile_sd::Float64 = 0.1
    sessile_sd::Float64 = 0.1
    temp_sd::Float64 = 0.1

    # Distributions
    adult_dist::Distribution = Normal(0, adult_sd)
    motile_dist::Distribution = Normal(0, motile_sd)
    sessile_dist::Distribution = Normal(0, sessile_sd)
    temp_dist::Distribution = Normal(0, temp_sd)
    skew_adult_dist::Distribution = SkewNormal(0, adult_sd, 2.0)

    # Sampling parameters
    rng::AbstractRNG = Random.GLOBAL_RNG
    production_start_week::Int64 = 34 # Week 34 is approximately July 1st
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.actions(mdp::SeaLiceSimMDP) = [NoTreatment, Treatment]
POMDPs.discount(mdp::SeaLiceSimMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceSimMDP, s::EvaluationState) = false

# -------------------------
# Transition function: the current sea lice level and the predicted sea lice level the following week
# are affected by the treatment and growth rate. The predicted sea lice level the following week will 
# have an additional e^r term because it is a week later.
# -------------------------
function POMDPs.transition(pomdp::SeaLiceSimMDP, s::EvaluationState, a::Action)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        
        # Calculate temperature based on the current week
        # TODO: consider whether to use the next week or the current week since measurements are taken approximately daily
        next_temp = get_temperature(s.AnnualWeek)

        # Predict the next adult sea lice level based on the current state and temperature
        next_adult, next_motile, next_sessile = predict_next_abundances(s.Adult, s.Motile, s.Sessile, next_temp)

        # Apply treatment
        if a == Treatment
            # Relative reduction in adult sea lice abundances
            # https://ars.els-cdn.com/content/image/1-s2.0-S0044848623005239-mmc1.pdf Table 5
            reduction_factor_sessile = 0.74
            reduction_factor_motile = 0.84
            reduction_factor_adult = 0.75

            # Sample reduction factors from a normal distribution
            sampled_reduction_factor_adult = rand(rng, Normal(reduction_factor_adult, 0.1))
            sampled_reduction_factor_motile = rand(rng, Normal(reduction_factor_motile, 0.1))
            sampled_reduction_factor_sessile = rand(rng, Normal(reduction_factor_sessile, 0.1))

            # Apply treatment to the predicted sea lice levels
            next_adult *= (1 - sampled_reduction_factor_adult)
            next_motile *= (1 - sampled_reduction_factor_motile)
            next_sessile *= (1 - sampled_reduction_factor_sessile)
        end

        # Biomass loss
        # TODO: add a function to calculate the biomass loss
        # https://ars.els-cdn.com/content/image/1-s2.0-S0044848623005239-mmc1.pdf
        lambda_mech = 1.210


        # Add noise
        next_adult = next_adult + rand(rng, pomdp.adult_dist)
        next_motile = next_motile + rand(rng, pomdp.motile_dist)
        next_sessile = next_sessile + rand(rng, pomdp.sessile_dist)
        next_temp = next_temp + rand(rng, pomdp.temp_dist)

        # Clamp the sea lice levels to be positive and within the bounds of the SeaLicePOMDP
        next_adult = max(next_adult, 0.0)
        next_motile = max(next_motile, 0.0)
        next_sessile = max(next_sessile, 0.0)
        next_pred = clamp(next_adult, pomdp.sea_lice_bounds...)

        return EvaluationState(
            next_pred, # SeaLiceLevel
            next_adult, # Adult
            next_motile, # Motile
            next_sessile, # Sessile
            next_temp, # Temperature
            s.ProductionWeek + 1, # ProductionWeek
            (s.AnnualWeek + 1) % 52, # AnnualWeek
            s.NumberOfFish, # NumberOfFish is constant
            s.Salinity, # Salinity is constant
        )
    end
end

# -------------------------
# Observation function: the current sea lice level is measured with noise
# and the predicted sea lice level the following week is calculated based on
# the current sea lice level this week
# -------------------------
function POMDPs.observation(pomdp::SeaLiceSimMDP, a::Action, s::EvaluationState)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng

        # Get observations of temperature and abundances
        observed_adult = s.Adult + rand(rng, pomdp.adult_dist)
        observed_motile = s.Motile + rand(rng, pomdp.motile_dist)
        observed_sessile = s.Sessile + rand(rng, pomdp.sessile_dist)
        observed_temperature = s.Temperature + rand(rng, pomdp.temp_dist)

        # Predict the next adult sea lice level based on the current state and temperature
        pred_adult, pred_motile, pred_sessile = predict_next_abundances(observed_adult, observed_motile, observed_sessile, observed_temperature)
        pred_adult = clamp(pred_adult, pomdp.sea_lice_bounds...)

        return EvaluationObservation(
            pred_adult, # SeaLiceLevel
            observed_adult, # Adult
            observed_motile, # Motile
            observed_sessile, # Sessile
            observed_temperature, # Temperature
            s.ProductionWeek, # ProductionWeek is fully observable
            s.AnnualWeek, # AnnualWeek is fully observable
            s.NumberOfFish, # NumberOfFish is fully observable
            s.Salinity, # Salinity is fully observable
        )
    end
end

# -------------------------
# Reward function: for now, we only penalize the current sea lice level
# Penalize the following:
# - Regulatory non-compliance: exceeding the bounds of the sea lice level
# - Treatment cost
# - Fish mortality
# - Fish health
# -------------------------
function POMDPs.reward(pomdp::SeaLiceSimMDP, s::EvaluationState, a::Action, sp::EvaluationState)

    alpha_1 = pomdp.lambda
    alpha_2 = 1
    alpha_3 = 1
    alpha_4 = 1

    lice_penalty = alpha_1 * s.SeaLiceLevel
    regulatory_penalty = alpha_2 * (s.Adult > 0.5)
    lost_biomass_penalty = alpha_3 * (s.NumberOfFish - sp.NumberOfFish)
    treatment_penalty = alpha_4 * get_cost(pomdp, a)

    return - (lice_penalty + treatment_penalty + lost_biomass_penalty + regulatory_penalty)
end

# -------------------------
# Initial state distribution
# -------------------------
function POMDPs.initialstate(pomdp::SeaLiceSimMDP)
    ImplicitDistribution(pomdp) do pomdp, rng

        # Initial temperature upon production start
        temperature = get_temperature(pomdp.production_start_week) + rand(rng, pomdp.temp_dist)

        # Initial sea lice level upon production start
        adult = pomdp.adult_mean + rand(rng, pomdp.adult_dist)
        motile  = pomdp.motile_mean + rand(rng, pomdp.motile_dist)
        sessile = pomdp.sessile_mean + rand(rng, pomdp.sessile_dist)

        # Next week's predicted adult sea lice level
        pred_adult, pred_motile, pred_sessile = predict_next_abundances(adult, motile, sessile, temperature)

        # Clamp the sea lice levels to be positive
        adult = max(adult, 0.0)
        motile = max(motile, 0.0)
        sessile = max(sessile, 0.0)
        pred_adult = clamp(pred_adult, pomdp.sea_lice_bounds...)

        return EvaluationState(
            pred_adult, # Predicted adult sea lice level the following week
            adult, # Adult
            motile, # Motile
            sessile, # Sessile
            temperature, # Temperature
            1, # ProductionWeek
            pomdp.production_start_week, # AnnualWeek
            200000, # NumberOfFish at the start of production
            30.0, # Salinity at the start of production
        )
    end
end

# -------------------------
# Cost function
# -------------------------
function get_cost(pomdp::SeaLiceSimMDP, a::Action)
    if a == Treatment
        return pomdp.costOfTreatment
    else
        return 0.0
    end
end

# -------------------------
# Fish mortality function
# -------------------------
function get_fish_mortality(pomdp::SeaLiceSimMDP, s::EvaluationState)
    return pomdp.fish_mortality_rate * s.Adult
end