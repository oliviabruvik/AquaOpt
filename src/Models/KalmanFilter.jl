include("SimulationPOMDP.jl")
include("SimulationLogPOMDP.jl")

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

# --------------------------------------------
# Temperature model used by both step and step_log
# --------------------------------------------
function get_temperature(annual_week)
    T_mean = 9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27  # aligns peak with July (week ~27)
    return T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)
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
    temperature = get_temperature(annual_week)

    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    s1 = 0.49  # sessile
    s2 = 2.3   # sessile → motile scaling
    s3 = 0.88  # motile
    s4 = 0.61  # adult

    # Development fractions
    d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temperature - 9))))
    d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temperature - 9))))

    # Stage transitions
    next_sessile = s1 * sessile
    next_motile = s3 * (1 - d2_val) * motile + s2 * d1_val * sessile
    next_adult = s4 * adult + d2_val * 0.5 * (s3 + s4) * motile

    # Apply treatment
    if treatment == 1.0
        next_adult *= (1 - 0.95)
    end

    # Update the state
    xp[1] = next_adult
    xp[2] = next_motile
    xp[3] = next_sessile
    xp[4] = temperature

    return xp
end

# x is state, u is action
function step_log(x, u)

    # Copy the state
    xp = copy(x)

    # Convert the state to raw space
    x_raw = exp.(x)

    # Unpack the action
    treatment = u[1]
    annual_week = u[2]

    # Get the temperature
    temperature = get_temperature(annual_week)

    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    s1 = 0.49  # sessile
    s2 = 2.3   # sessile → motile scaling
    s3 = 0.88  # motile
    s4 = 0.61  # adult

    # Development fractions
    d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temperature - 9))))
    d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temperature - 9))))

    # Stage transitions
    next_sessile = s1 * x_raw[2]
    next_motile = s3 * (1 - d2_val) * x_raw[3] + s2 * d1_val * x_raw[2]
    next_adult = s4 * x_raw[1] + d2_val * 0.5 * (s3 + s4) * x_raw[3]

    # Apply treatment
    if treatment == 1.0
        next_adult *= (1 - 0.95)
    end

    # Update the state
    xp[1] = log(next_adult)
    xp[2] = log(next_motile)
    xp[3] = log(next_sessile)
    xp[4] = temperature

    return xp
end

# --------------------------------------------
# Observation model (fully observed)
# --------------------------------------------
observe(x, u) = x  # noise handled by V

# --------------------------------------------
# Build EKF or UKF filter
# --------------------------------------------
function build_kf(pomdp::Union{SeaLiceSimMDP, SeaLiceLogSimMDP}; ekf_filter=ekf_filter)
    
    # Create noise matrices with the diagonal elements being adult, sessile, motile, temperature
    # The W matrix stores the process noise for each state variable
    # The V matrix stores the observation noise for each state variable
    W = Diagonal([pomdp.adult_sd^2, pomdp.motile_sd^2, pomdp.sessile_sd^2, pomdp.temp_sd^2])
    V = Diagonal([pomdp.adult_sd^2, pomdp.motile_sd^2, pomdp.sessile_sd^2, pomdp.temp_sd^2])

    # Create dynamics and observation models
    step_func = typeof(pomdp) <: SeaLiceLogSimMDP ? step_log : step
    dmodel = NonlinearDynamicsModel(step_func, W)
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
function POMDPs.update(updater::KalmanUpdater, b0::GaussianBelief, a::Action, o::Union{SeaLiceObservation, SeaLiceLogObservation, EvaluationObservation})
    
    @assert (a == NoTreatment || a == Treatment)
    # We're passing in the annual week to the action because the dynamics model depends on it for the temperature model
    action = [a == Treatment ? 1.0 : 0.0, o.AnnualWeek]
    observation = [o.Adult, o.Motile, o.Sessile, o.Temperature]
    
    b = GaussianFilters.update(updater.filter, b0, action, observation)

    return b
end