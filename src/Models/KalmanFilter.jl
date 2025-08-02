include("SimulationPOMDP.jl")
include("SimulationLogPOMDP.jl")

using POMDPs
using GaussianFilters
using Distributions
using LinearAlgebra
using Random

# struct KFUpdaterStruct
#     # ekf_abundance::ExtendedKalmanFilter
#     # ukf_abundance::UnscentedKalmanFilter
#     # ekf_temp::ExtendedKalmanFilter
#     # ukf_temp::UnscentedKalmanFilter
#     ekf::ExtendedKalmanFilter
#     ukf::UnscentedKalmanFilter
#     pomdp::Union{SeaLiceSimMDP, SeaLiceLogSimMDP}
# end

struct KalmanUpdater <: POMDPs.Updater
    filter::Union{ExtendedKalmanFilter, UnscentedKalmanFilter}
end


# x is state, u is action
function step(x, u)

    # Copy the state
    xp = copy(x)

    # Unpack the action
    treatment = u[1]
    annual_week = u[2]

    # Get the temperature
    T_mean = 9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27  # aligns peak with July (week ~27)
    temperature = T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)

    # Weekly survival probabilities from Table 1 of Stige et al. 2025.
    s1 = 0.49  # sessile
    s2 = 2.3   # sessile → motile scaling
    s3 = 0.88  # motile
    s4 = 0.61  # adult

    # Development fractions
    d1_val = 1 / (1 + exp(-(-2.4 + 0.37 * (temperature - 9))))
    d2_val = 1 / (1 + exp(-(-2.1 + 0.037 * (temperature - 9))))

    # Stage transitions
    next_sessile = s1 * x[2]
    next_motile = s3 * (1 - d2_val) * x[3] + s2 * d1_val * x[2]
    next_adult = s4 * x[1] + d2_val * 0.5 * (s3 + s4) * x[3]

    # Apply treatment
    if treatment == 1.0
        next_adult *= (1 - 0.95)
    end

    # Update the state
    xp[1] = next_adult
    xp[2] = next_sessile
    xp[3] = next_motile
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
    T_mean = 9.0      # average annual temperature (°C)
    T_amp = 4.5       # amplitude (°C)
    peak_week = 27  # aligns peak with July (week ~27)
    temperature = T_mean + T_amp * cos(2π * (annual_week - peak_week) / 52)

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
    xp[2] = log(next_sessile)
    xp[3] = log(next_motile)
    xp[4] = temperature

    return xp
end

# build dynamics model
function observe(x, u)
    return x # Now returns a 4D vector: [lice_abundance, sessile, motile, temperature]
end

# Extended Kalman Filter Updater
function build_kf(pomdp::Union{SeaLiceSimMDP, SeaLiceLogSimMDP}; process_noise=process_noise, observation_noise=observation_noise, ekf_filter=ekf_filter)
    
    # Create noise matrices
    W = process_noise^2 * Matrix{Float64}(I, 4, 4)
    V = observation_noise^2 * Matrix{Float64}(I, 4, 4)

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

# POMDPs.updater(policy::Policy, pomdp::SeaLiceSimMDP) = EKFUpdater(pomdp, process_noise=ST_DEV, observation_noise=ST_DEV)

function POMDPs.initialize_belief(updater::KalmanUpdater, dist::Tuple{Distribution, Distribution, Distribution, Distribution})
    mean_vec = [mean(dist[1]), mean(dist[2]), mean(dist[3]), mean(dist[4])]
    cov_mat =Matrix{Float64}(I, 4, 4)
    cov_mat[1, 1] = std(dist[1])^2
    cov_mat[2, 2] = std(dist[2])^2
    cov_mat[3, 3] = std(dist[3])^2
    cov_mat[4, 4] = std(dist[4])^2
    return GaussianBelief(mean_vec, cov_mat)
end

function POMDPs.update(updater::KalmanUpdater, b0::GaussianBelief, a::Action, o::Union{SeaLiceObservation, SeaLiceLogObservation, EvaluationObservation})
    
    @assert (a == NoTreatment || a == Treatment)
    # We're passing in the annual week to the action because the dynamics model depends on it for the temperature model
    action = [a == Treatment ? 1.0 : 0.0, o.AnnualWeek]
    observation = [o.SeaLiceLevel, o.Sessile, o.Motile, o.Temperature]
    return GaussianFilters.update(updater.filter, b0, action, observation)
end