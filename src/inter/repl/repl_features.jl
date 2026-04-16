####################################################################################################

module FEREPL

####################################################################################################

using Avicenna.Flow: Cache, run
using ..FEFlow: features_flow
using ..FECore: file_hash

####################################################################################################

export run_features

####################################################################################################

"""
    run_features(;
      modern::String,
      ancient::String,
      out::String="features.csv",
      onehot::Bool=false,
      no_cache::Bool=false) -> Avicenna.Result

Run the feature extraction workflow from the REPL.
"""
function run_features(;
  modern::String,
  ancient::String,
  out::String = "features.csv",
  onehot::Bool = false,
  no_cache::Bool = false,
)
  config = Dict{String,Any}(
    "modern" => modern,
    "ancient" => ancient,
    "out" => out,
    "onehot" => onehot,
    "_modern_hash" => file_hash(modern),
    "_ancient_hash" => file_hash(ancient),
  )
  cache = Cache("cache/feature", !no_cache)
  return run(features_flow, config, cache = cache)
end

####################################################################################################

end

####################################################################################################
