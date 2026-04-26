module NN

include("util/nnutil.jl")
include("flow/nnflow.jl")
include("inter/cli/nncli.jl")
include("inter/repl/nnrepl.jl")

export NNCore, NNFlow, NNCLI, NNREPL

end
