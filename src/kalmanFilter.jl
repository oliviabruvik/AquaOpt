using POMDPs
using GaussianFilters
using Distributions
using LinearAlgebra
using Random

include("SimulationPOMDP.jl")

struct EKFUpdater
    ekf::ExtendedKalmanFilter
    pomdp::SeaLiceSimMDP
end

function step(x, u)
    growth_rate = 1.2
    rho = 0.7
    xp = copy(x)
    xp[1] = (1 - (u[1] == 1.0 ? rho : 0.0)) * exp(growth_rate) * x[1]
    return xp
end

# build dynamics model
function observe(x, u)
    return x
end

# Extended Kalman Filter Updater
function EKFUpdater(pomdp::SeaLiceSimMDP; process_noise=STD_DEV, observation_noise=STD_DEV)
    
    # Create noise matrices
    W = process_noise^2 * Matrix{Float64}(I, 1, 1)
    V = observation_noise^2 * Matrix{Float64}(I, 1, 1)

    # Create dynamics and observation models
    dmodel = NonlinearDynamicsModel(step, W)
    omodel = NonlinearObservationModel(observe, V)

    # Create and return the EKF
    ekf = ExtendedKalmanFilter(dmodel, omodel)
    return EKFUpdater(ekf, pomdp)
end

POMDPs.updater(policy::Policy, pomdp::SeaLiceSimMDP) = EKFUpdater(pomdp, process_noise=STD_DEV, observation_noise=STD_DEV)

function POMDPs.initialize_belief(updater::EKFUpdater, dist::Any)
    # Mimic DiscreteUpdater: initialize belief from distribution or sampled value
    if dist isa ImplicitDistribution
        # Sample to estimate mean and variance
        samples = [rand(dist).SeaLiceLevel for _ in 1:1000]
        mean_val = mean(samples)
        var_val = var(samples)
    elseif dist isa Number
        mean_val = dist
        var_val = STD_DEV^2
    else
        mean_val = mean(dist)
        var_val = var(dist)
    end

    # updater.ekf.x = [mean_val]
    # updater.ekf.P = [var_val]
    return Normal(mean_val, sqrt(var_val))
end

function POMDPs.update(updater::EKFUpdater, belief::Normal, a::Action, o::SeaLiceObservation)
    
    # Convert action to control input
    u = [a == Treatment ? 1.0 : 0.0]
    
    # Predict step
    updater.ekf = predict(updater.ekf, u)
    
    # Update step with observation
    z = [o.SeaLiceLevel]
    updater.ekf = update(updater.ekf, z)
    
    # Return new belief as Normal distribution
    mean_val = updater.ekf.x[1]
    var_val = updater.ekf.P[1,1]
    return Normal(mean_val, sqrt(var_val))
end