const ACTION_SHORT_LABELS = Dict(
    NoTreatment => "",
    MechanicalTreatment => "M",
    ChemicalTreatment => "C",
    ThermalTreatment => "Th",
)

action_short_label(a) = get(ACTION_SHORT_LABELS, a, "")

# Shared policy colors/markers for Plots.jl-based figures.
# Used across Timeseries.jl, Comparison.jl, ParallelPlots.jl, etc.
const POLICY_STYLES = Dict(
    "Heuristic_Policy"   => (color=:blue,   marker=:circle),
    "VI_Policy"          => (color=:red,     marker=:square),
    "NUS_SARSOP_Policy"     => (color=:green,   marker=:diamond),
    "Native_SARSOP_Policy"  => (color=:green,   marker=:diamond),
    "QMDP_Policy"        => (color=:purple,  marker=:dtriangle),
    "Random_Policy"      => (color=:orange,  marker=:rect),
    "AlwaysTreat_Policy" => (color=:brown,   marker=:dtriangle),
)
