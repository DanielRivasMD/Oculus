module NNREPL

using Avicenna.Flow: Cache, launch
using ..NNFlow: flow

export run

"""
    run(;
        cnn::String,
        sample::String,
        out::String = "model.bson",
        no_cache::Bool = false,
    ) -> Result

Launch the CNN training workflow from the REPL.
"""
function run(;
  cnn::String,
  sample::String,
  out::String = "model.bson",
  no_cache::Bool = false,
)
  config = Dict{String,Any}("cnn" => cnn, "sample" => sample, "out" => out)
  cache = Cache("cache/nn", !no_cache)
  return launch(flow, config, cache = cache)
end

end
