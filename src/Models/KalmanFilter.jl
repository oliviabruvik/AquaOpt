include("SimulationPOMDP.jl")
include("../Utils/Utils.jl")

using POMDPs
using GaussianFilters
using Distributions
using LinearAlgebra
using Random

# --------------------------------------------
# Updater wrapper struct
# --------------------------------------------
mutable struct KalmanUpdater <: POMDPs.Updater
    filter::Union{ExtendedKalmanFilter, UnscentedKalmanFilter}
end

# x is state, u is action
function step(x, u)

    # Copy the state
    xp = copy(x)

    # Unpack the action
    treatment = u[1]
    annual_week = u[2]

    # Unpack the state
    adult = x[1]
    motile = x[2]
    sessile = x[3]

    # Get the temperature
    temp = get_temperature(Int64(annual_week))

    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    s1 = 0.49  # sessile
    s2 = 2.3   # sessile → motile scaling
    s3 = 0.88  # motile
    s4 = 0.61  # adult

    d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temp - 9))))
    d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temp - 9))))

    next_sessile = s1 * sessile
    next_motile = s3 * (1 - d2_val) * motile + s2 * d1_val * sessile
    next_adult = s4 * adult + d2_val * 0.5 * (s3 + s4) * motile

    # Clamp the sea lice levels to be positive
    next_adult = max(next_adult, 0.0)
    next_motile = max(next_motile, 0.0)
    next_sessile = max(next_sessile, 0.0)

    # Apply treatment
    if treatment == 1.0
        next_adult *= (1 - 0.95)
    end

    # Update the state
    xp[1] = next_adult
    xp[2] = next_motile
    xp[3] = next_sessile
    xp[4] = temp

    return xp
end

# --------------------------------------------
# Observation model (fully observed)
# --------------------------------------------
observe(x, u) = x  # noise handled by V

# --------------------------------------------
# Build EKF or UKF filter
# --------------------------------------------
function build_kf(pomdp::SeaLiceSimMDP; ekf_filter=ekf_filter)
    
    # Create noise matrices with the diagonal elements being adult, sessile, motile, temperature
    # The W matrix stores the process noise for each state variable
    # The V matrix stores the observation noise for each state variable
    W = Diagonal([pomdp.adult_sd^2, pomdp.motile_sd^2, pomdp.sessile_sd^2, pomdp.temp_sd^2])
    V = Diagonal([pomdp.adult_sd^2, pomdp.motile_sd^2, pomdp.sessile_sd^2, pomdp.temp_sd^2])

    # Create dynamics and observation models
    dmodel = NonlinearDynamicsModel(step, W)
    omodel = NonlinearObservationModel(observe, V)

    # Create and return the KF
    if ekf_filter
        return ExtendedKalmanFilter(dmodel, omodel)
    else
        return UnscentedKalmanFilter(dmodel, omodel)
    end
end

# --------------------------------------------
# Belief initialization
# --------------------------------------------
function POMDPs.initialize_belief(updater::KalmanUpdater, dists::NTuple{4, Distribution})
    μ0 = mean.([dists[1], dists[2], dists[3], dists[4]])
    σ2s = std.([dists[1], dists[2], dists[3], dists[4]]).^2
    Σ0 = Diagonal(σ2s)
    
    return GaussianBelief(μ0, Σ0)
end

# --------------------------------------------
# Kalman update
# --------------------------------------------
function POMDPs.update(updater::KalmanUpdater, b0::GaussianBelief, a::Action, o::Union{SeaLiceObservation, EvaluationObservation})
    
    @assert (a == NoTreatment || a == Treatment)
    # We're passing in the annual week to the action because the dynamics model depends on it for the temperature model
    action = [a == Treatment ? 1.0 : 0.0, o.AnnualWeek]
    observation = [o.Adult, o.Motile, o.Sessile, o.Temperature]
    
    b = GaussianFilters.update(updater.filter, b0, action, observation)

    return b
end