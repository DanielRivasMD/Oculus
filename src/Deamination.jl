####################################################################################################

"""
Compare per‑position base composition of FASTA sequences:
- Two‑file mode: loads `ancient` and `modern` FASTA, computes A/T/G/C
  percentages at every position, writes a combined CSV, and plots both
  profiles on one graph
- Single‑file mode: loads one FASTA, computes its composition, writes
  a simple CSV, and plots its profile
Outputs are written to the directory specified by `--outdir` (default: current
directory).
"""
module DA

####################################################################################################

include("util/shared.jl")
include("util/dautil.jl")
include("flow/daflow.jl")
include("inter/repl/darepl.jl")
include("inter/cli/dacli.jl")

####################################################################################################

export DACore, DAFlow, DACLI, DAREPL

####################################################################################################

end

####################################################################################################
