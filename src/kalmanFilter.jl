using POMDPs
using GaussianFilters
using Distributions
using LinearAlgebra
using Random

include("SimulationPOMDP.jl")
include("SimulationLogPOMDP.jl")

struct KFUpdaterStruct
    ekf::ExtendedKalmanFilter
    ukf::UnscentedKalmanFilter
    pomdp::Union{SeaLiceSimMDP, SeaLiceLogSimMDP}
end

# x is state, u is action
function step(x, u)
    growth_rate = 1.2
    rho = 0.7
    xp = copy(x)
    xp[1] = (1 - (u[1] == 1.0 ? rho : 0.0)) * exp(growth_rate) * x[1]
    return xp
end

# x is state, u is action
function step_log(x, u)
    growth_rate = 1.2
    rho = 0.7
    xp = copy(x)
    xp[1] = log(1 - (u[1] == 1.0 ? rho : 0.0)) + growth_rate + x[1]
    return xp
end

# build dynamics model
function observe(x, u)
    return x
end

# Extended Kalman Filter Updater
function KFUpdater(pomdp::Union{SeaLiceSimMDP, SeaLiceLogSimMDP}; process_noise=process_noise, observation_noise=observation_noise)
    
    # Create noise matrices
    W = process_noise^2 * Matrix{Float64}(I, 1, 1)
    V = observation_noise^2 * Matrix{Float64}(I, 1, 1)

    # Create dynamics and observation models
    step_func = typeof(pomdp) <: SeaLiceLogSimMDP ? step_log : step
    dmodel = NonlinearDynamicsModel(step_func, W)
    omodel = NonlinearObservationModel(observe, V)

    # Create and return the KF
    ekf = ExtendedKalmanFilter(dmodel, omodel)
    ukf = UnscentedKalmanFilter(dmodel, omodel)
    return KFUpdaterStruct(ekf, ukf, pomdp)
end

# POMDPs.updater(policy::Policy, pomdp::SeaLiceSimMDP) = EKFUpdater(pomdp, process_noise=ST_DEV, observation_noise=ST_DEV)

function POMDPs.initialize_belief(updater::Union{ExtendedKalmanFilter, UnscentedKalmanFilter}, dist::Distribution)
    return GaussianBelief([mean(dist)], Matrix{Float64}(I, 1, 1) * std(dist))
end

function runKalmanFilter(kf::Union{ExtendedKalmanFilter, UnscentedKalmanFilter}, b0::GaussianBelief, a::Action, o::Union{SeaLiceObservation, SeaLiceLogObservation})
    
    # Convert action to vector
    action = [a == Treatment ? 1.0 : 0.0]

    # Update belief
    return GaussianFilters.update(kf, b0, action, [o.SeaLiceLevel])
end