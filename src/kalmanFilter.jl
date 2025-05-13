using POMDPs
using GaussianFilters
using Distributions
using LinearAlgebra
using Random

include("SimulationPOMDP.jl")

struct EKFUpdaterStruct
    ekf::ExtendedKalmanFilter
    pomdp::SeaLiceSimMDP
end

# x is state, u is action
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
    return EKFUpdaterStruct(ekf, pomdp)
end

POMDPs.updater(policy::Policy, pomdp::SeaLiceSimMDP) = EKFUpdater(pomdp, process_noise=STD_DEV, observation_noise=STD_DEV)

function POMDPs.initialize_belief(updater::ExtendedKalmanFilter, dist::Any)
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

    return GaussianBelief([mean_val], Matrix{Float64}(I, 1, 1) * sqrt(var_val))
end

function kalmanFilterUpdate(ekf::ExtendedKalmanFilter, belief::GaussianBelief, a::Action, o::SeaLiceObservation)
    
    # TODO:Update step with observation or simulated data?
    # z = [o.SeaLiceLevel]

    # Create action sequence
    action_sequence = [[a == Treatment ? 1.0 : 0.0]]

    # Simulate model
    sim_states, sim_measurements = GaussianFilters.simulation(ekf, belief, action_sequence)

    # Run filter on simulated data
    filtered_beliefs = run_filter(ekf, belief, action_sequence, sim_measurements)

    # Turn array of belief structs into simple tensors
    μ, Σ = unpack(filtered_beliefs);
    
    # Return new belief as Normal distribution
    return GaussianBelief([μ[1]], Matrix{Float64}(I, 1, 1) * sqrt(Σ[1,1]))
end