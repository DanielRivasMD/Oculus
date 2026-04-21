####################################################################################################

module PEREPL

####################################################################################################

using Avicenna.Flow: Cache, launch
using ..PEFlow: flow

####################################################################################################

export run

####################################################################################################

"""
    run(csv_path::String; no_cache=false) -> Result

Run performance evaluation on a predictions CSV (must contain columns 'truth' and 'prediction').
"""
function run(csv_path::String; no_cache::Bool = false)
  config = Dict{String,Any}("infile" => csv_path)
  cache = Cache("cache/performance", !no_cache)
  return launch(flow, config, cache = cache)
end

####################################################################################################

end

####################################################################################################
