####################################################################################################

module Paths

"project root"
const PROJECT = normpath(joinpath(@__DIR__, "..", ".."))

"data"
const DATA = joinpath(PROJECT, "data")
const BAM = joinpath(DATA, "bam")
const FASTA = joinpath(DATA, "fasta")
const INFERENCE = joinpath(DATA, "inference")

"graph"
const GRAPH = joinpath(PROJECT, "graph")
const PERFORMANCE = joinpath(GRAPH, "performance")
const ROC = joinpath(GRAPH, "roc")

"model"
const MODEL = joinpath(PROJECT, "model")

# sh: only called from shell

"src"
const SRC = joinpath(PROJECT, "src")
const CONFIG = joinpath(SRC, "config")
const UTIL = joinpath(SRC, "util")
const BIN = joinpath(SRC, "bin")

# toml: only called from shell

"Ensure directories exist (for outputs)"
function ensure_dirs()
    for d in (DATA, BAM, FASTA, INFERENCE, GRAPH, PERFORMANCE, ROC, MODEL)
        isdir(d) || mkpath(d)
    end
end

end

####################################################################################################
