include("../Utils/SharedTypes.jl")
include("../Utils/Utils.jl")

using POMDPs
using GaussianFilters
using Distributions
using LinearAlgebra
using Random

# TODO: uncertainty in kalman filter a bit bigger than in the simulation
# TODO: simLog

# --------------------------------------------
# Updater wrapper struct
# --------------------------------------------
mutable struct KalmanUpdater <: POMDPs.Updater
    filter::Union{ExtendedKalmanFilter, UnscentedKalmanFilter}
end

# x is state, u is action
function step(x, u)

    # Unpack the state and action
    adult, motile, sessile, temp = x
    treatment, annual_week = u

    # Apply treatment
    config = get_action_config(Action(Int(treatment)))
    adult *= (1 - config.adult_reduction)
    motile *= (1 - config.motile_reduction)
    sessile *= (1 - config.sessile_reduction)

    # Predict the next abundances
    next_adult, next_motile, next_sessile = predict_next_abundances(adult, motile, sessile, temp)

    # Get the temperature for the next week
    next_annual_week = (annual_week + 1) % 52
    next_temp = get_temperature(next_annual_week)

    return [next_adult, next_motile, next_sessile, next_temp]
end

# --------------------------------------------
# Observation model (fully observed)
# --------------------------------------------
observe(x, u) = x  # noise handled by V

# --------------------------------------------
# Build EKF or UKF filter
# --------------------------------------------
function build_kf(sim_pomdp::Any; ekf_filter=false)
    
    # Create noise matrices with the diagonal elements being adult, sessile, motile, temperature
    # The W matrix stores the process noise for each state variable
    # The V matrix stores the observation noise for each state variable
    W = Diagonal([sim_pomdp.adult_sd^2, sim_pomdp.motile_sd^2, sim_pomdp.sessile_sd^2, sim_pomdp.temp_sd^2])
    V = Diagonal([sim_pomdp.adult_sd^2, sim_pomdp.motile_sd^2, sim_pomdp.sessile_sd^2, sim_pomdp.temp_sd^2])

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
    
    # TODO: add full state distribution with 0 noise
    μ0 = mean.([dists[1], dists[2], dists[3], dists[4]])
    σ2s = std.([dists[1], dists[2], dists[3], dists[4]]).^2
    Σ0 = Diagonal(σ2s)
    
    return GaussianBelief(μ0, Σ0)
end

# --------------------------------------------
# Kalman update
# --------------------------------------------
function POMDPs.update(updater::KalmanUpdater, b0::GaussianBelief, a::Action, o::Any)

    # We're passing in the annual week to the action because the dynamics model depends on it for the temperature model
    action = [Int(a), o.AnnualWeek]
    observation = [o.Adult, o.Motile, o.Sessile, o.Temperature]
    b = GaussianFilters.update(updater.filter, b0, action, observation)

    return b
end