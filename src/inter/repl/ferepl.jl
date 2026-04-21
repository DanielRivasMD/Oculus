####################################################################################################

module FEREPL

####################################################################################################

using Avicenna.Flow: Cache, launch
using ..FEFlow: flow
using ..FECore: file_hash

####################################################################################################

export run

####################################################################################################

"""
    run(;
      ancient::String,
      modern::String,
      out::String="features.csv",
      onehot::Bool=false,
      no_cache::Bool=false) -> Avicenna.Result

Run the feature extraction workflow from the REPL.
"""
function run(;
  ancient::String,
  modern::String,
  out::String = "features.csv",
  onehot::Bool = false,
  no_cache::Bool = false,
)
  config = Dict{String,Any}(
    "ancient" => ancient,
    "modern" => modern,
    "out" => out,
    "onehot" => onehot,
    "_ancient_hash" => file_hash(ancient),
    "_modern_hash" => file_hash(modern),
  )
  cache = Cache("cache/feature", !no_cache)
  return launch(flow, config, cache = cache)
end

####################################################################################################

end

####################################################################################################
