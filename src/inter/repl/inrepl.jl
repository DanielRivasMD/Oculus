####################################################################################################

module INREPL

####################################################################################################

using Avicenna.Flow: Cache, launch
using ..INFlow: flow

####################################################################################################

export run

####################################################################################################

"""
    run(;
        model::String,
        data::String,
        out::String = "predictions.csv",
        no_cache::Bool = false,
    ) -> Result

Run the inference workflow from the REPL.
"""
function run(;
  model::String,
  data::String,
  out::String = "predictions.csv",
  no_cache::Bool = false,
)
  config = Dict{String,Any}("model" => model, "data" => data, "out" => out)
  cache = Cache("cache/inference", !no_cache)
  return launch(flow, config, cache = cache)
end

####################################################################################################

end

####################################################################################################
