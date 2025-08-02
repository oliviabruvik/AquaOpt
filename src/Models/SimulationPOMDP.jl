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
    Sessile::Float64 # The sessile sea lice level this week
    Motile::Float64 # The motile sea lice level this week
    Temperature::Float64 # The mean temperature (°C) over the last 7 days at the farm (based on approximately daily measurements)
    ProductionWeek::Int64 # The number of weeks since production start
    AnnualWeek::Int64 # The week of the year
end

"Observation representing the predicted observed sea lice level the following week and the current observed sea lice level."
struct EvaluationObservation
	SeaLiceLevel::Float64 # The observed predicted adult sea lice level the following week (without treatment)
    Adult::Float64 # The observed adult sea lice level this week
    Sessile::Float64 # The observed sessile sea lice level this week
    Motile::Float64 # The observed motile sea lice level this week
    Temperature::Float64 # The observed mean temperature (°C) over the last 7 days at the farm (based on approximately daily measurements)
    ProductionWeek::Int64 # The number of weeks since production start
    AnnualWeek::Int64 # The week of the year
end

"Available actions: NoTreatment or Treatment."
@enum Action NoTreatment Treatment

# -------------------------
# SeaLiceMDP Definition
# -------------------------
"Sea lice MDP with growth dynamics and treatment effects."
@with_kw struct SeaLiceSimMDP <: POMDP{EvaluationState, Action, EvaluationObservation}
	lambda::Float64 = 0.5
	costOfTreatment::Float64 = 10.0
	growthRate::Float64 = 0.3
	rho::Float64 = 0.95
    discount_factor::Float64 = 0.95
    skew::Bool = false
    sea_lice_bounds::Tuple{Float64, Float64} = (0.0, 30.0)
    initial_bounds::Tuple{Float64, Float64} = (0.0, 0.25)
    sea_lice_initial_mean::Float64 = 0.125
    sampling_sd::Float64 = 0.5
    rng::AbstractRNG = Random.GLOBAL_RNG
    normal_dist::Distribution = Normal(0, sampling_sd)
    skew_normal_dist::Distribution = SkewNormal(0, sampling_sd, 2.0)
    production_start_week::Int64 = 34 # Week 34 is approximately July 1st
end

# -------------------------
# POMDPs.jl Interface
# -------------------------
POMDPs.actions(mdp::SeaLiceSimMDP) = [NoTreatment, Treatment]
POMDPs.discount(mdp::SeaLiceSimMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::SeaLiceSimMDP, s::EvaluationState) = false

# -------------------------
# Temperature model: Return estimated weekly sea surface temperature (°C) for Norwegian salmon farms.
# -------------------------
function temperature_model(annual_week::Int64)
    T_mean = 9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27  # aligns peak with July (week ~27)
    return T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)
end

# -------------------------
# Development rate model: Return the expected development rate based on the temperature.
# Based on A salmon lice prediction model, Stige et al. 2025.
# https://www.sciencedirect.com/science/article/pii/S0167587724002915
# -------------------------
function d1(temperature::Float64)
    return 1 / (1 + exp(-(-2.4 + 0.37 * (temperature - 9))))
end

function d2(temperature::Float64)
    return 1 / (1 + exp(-(-2.1 + 0.037 * (temperature - 9))))
end

# -------------------------
# Predict the next adult sea lice level based on the current state and temperature
# -------------------------
function predict_next_lice(adult::Float64, motile::Float64, sessile::Float64, temp::Float64)
    
    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    s1 = 0.49  # sessile
    s2 = 2.3   # sessile → motile scaling
    s3 = 0.88  # motile
    s4 = 0.61  # adult

    d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temp - 9))))
    d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temp - 9))))

    pred_motile = s3 * (1 - d2_val) * motile + s2 * d1_val * sessile
    pred_adult = s4 * adult + d2_val * 0.5 * (s3 + s4) * motile

    return pred_adult  # or full tuple if you want
end

# -------------------------
# Transition function: the current sea lice level and the predicted sea lice level the following week
# are affected by the treatment and growth rate. The predicted sea lice level the following week will 
# have an additional e^r term because it is a week later.
# -------------------------
function POMDPs.transition(pomdp::SeaLiceSimMDP, s::EvaluationState, a::Action)
    ImplicitDistribution(pomdp, s, a) do pomdp, s, a, rng
        
        # Calculate temperature based on the current week
        # TODO: consider whether to use the next week or the current week since measurements are taken approximately daily
        next_temperature = temperature_model(s.AnnualWeek)

        # Weekly survival probabilities from Table 1 of Stige et al. 2025.
        s1 = 0.49  # sessile
        s2 = 2.3   # sessile → motile scaling
        s3 = 0.88  # motile
        s4 = 0.61  # adult

        # Development fractions
        d1_val = d1(next_temperature)
        d2_val = d2(next_temperature)

        # Stage transitions
        next_sessile = s1 * s.Sessile
        next_motile = s3 * (1 - d2_val) * s.Motile + s2 * d1_val * s.Sessile
        next_adult = s4 * s.SeaLiceLevel + d2_val * 0.5 * (s3 + s4) * s.Motile


        # Apply treatment
        if a == Treatment
            next_adult *= (1 - pomdp.rho)
        end

        # Add noise
        next_sessile = clamp(next_sessile + rand(rng, pomdp.normal_dist), pomdp.sea_lice_bounds...)
        next_motile = clamp(next_motile + rand(rng, pomdp.normal_dist), pomdp.sea_lice_bounds...)
        next_adult = clamp(next_adult + rand(rng, pomdp.normal_dist), pomdp.sea_lice_bounds...)

        return EvaluationState(
            next_adult, # SeaLiceLevel
            next_adult, # Adult
            next_sessile, # Sessile
            next_motile, # Motile
            next_temperature, # Temperature
            s.ProductionWeek + 1, # ProductionWeek
            (s.AnnualWeek + 1) % 52, # AnnualWeek
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

        # Get observation of temperature
        observed_temperature = s.Temperature + rand(rng, pomdp.normal_dist)
        observed_adult = clamp(s.SeaLiceLevel + rand(rng, pomdp.normal_dist), pomdp.sea_lice_bounds...)
        observed_sessile = clamp(s.Sessile + rand(rng, pomdp.normal_dist), pomdp.sea_lice_bounds...)
        observed_motile = clamp(s.Motile + rand(rng, pomdp.normal_dist), pomdp.sea_lice_bounds...)

        # Calculate the predicted abundance for the following week
        # Weekly survival probabilities from Table 1 of Stige et al. 2025.
        s1 = 0.49  # sessile
        s2 = 2.3   # sessile → motile scaling
        s3 = 0.88  # motile
        s4 = 0.61  # adult

        # Development fractions
        d1_val = d1(observed_temperature)
        d2_val = d2(observed_temperature)

        # Stage transitions
        # Note that the predicted adult sea lice level is fully observable
        predicted_adult = clamp(s4 * observed_adult + d2_val * 0.5 * (s3 + s4) * observed_motile, pomdp.sea_lice_bounds...)

        return EvaluationObservation(
            predicted_adult, # SeaLiceLevel
            observed_adult, # Adult
            observed_sessile, # Sessile
            observed_motile, # Motile
            observed_temperature, # Temperature
            s.ProductionWeek, # ProductionWeek is fully observable
            s.AnnualWeek # AnnualWeek is fully observable
        )
    end
end

# -------------------------
# Reward function: for now, we only penalize the current sea lice level
# -------------------------
function POMDPs.reward(pomdp::SeaLiceSimMDP, s::EvaluationState, a::Action)
    lice_penalty = pomdp.lambda * s.SeaLiceLevel
    treatment_penalty = a == Treatment ? (1 - pomdp.lambda) * pomdp.costOfTreatment : 0.0
    return - (lice_penalty + treatment_penalty)
end

# -------------------------
# Initial state distribution: returns a two-dimensional state with the predicted abundance the
# and the current abundance
# -------------------------
function POMDPs.initialstate(pomdp::SeaLiceSimMDP)
    ImplicitDistribution(pomdp) do pomdp, rng

        # Initial sea lice level upon production start
        adult = pomdp.sea_lice_initial_mean + rand(rng, pomdp.normal_dist)
        adult = clamp(adult, pomdp.initial_bounds...)
        sessile = clamp(rand(rng, Normal(0.1, 0.05)), pomdp.initial_bounds...)
        motile  = clamp(rand(rng, Normal(0.1, 0.05)), pomdp.initial_bounds...)

        # Next week's predicted adult sea lice level
        predicted_adult = adult + rand(rng, pomdp.normal_dist)
        predicted_adult = clamp(predicted_adult, pomdp.sea_lice_bounds...)

        # Initial temperature upon production start
        temperature = temperature_model(pomdp.production_start_week)

        return EvaluationState(
            predicted_adult, # SeaLiceLevel
            adult, # Adult
            sessile, # Sessile
            motile, # Motile
            temperature, # Temperature
            1, # ProductionWeek
            pomdp.production_start_week, # AnnualWeek
        )
    end
end