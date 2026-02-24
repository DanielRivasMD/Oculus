module Oculus

# ────────────────────────────────────────────────────────────────
# Load Avicenna (the framework)
# ────────────────────────────────────────────────────────────────
using Avicenna

# ────────────────────────────────────────────────────────────────
# Load internal modules of the analysis repo
# ────────────────────────────────────────────────────────────────
# These paths assume your repo structure is:
#   src/
#     Oculus.jl
#     util/feature.jl
#     wflow/feature.jl
#     inter/cli.jl
#     inter/repl.jl

include("util/feature.jl")
include("wflow/feature.jl")
include("inter/cli.jl")
include("inter/repl.jl")

# Bring submodules into the namespace
using .Process
using .FeatureWorkflow
using .CLI
using .REPL

# ────────────────────────────────────────────────────────────────
# Re-export public API
# ────────────────────────────────────────────────────────────────
# This lets users call:
#   Oculus.REPL.run_features(...)
#   Oculus.CLI.run_feature_extraction(...)
#   Oculus.FeatureWorkflow.feature_workflow
#   Oculus.Process.load_fasta(...)
#
# without needing to know the internal folder structure.

export Process
export FeatureWorkflow
export CLI
export REPL

end # module
