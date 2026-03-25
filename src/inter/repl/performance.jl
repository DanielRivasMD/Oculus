module PEREPL

using Avicenna.Flow: Cache, run
using ..PEFlow: performance_flow

export run_performance

"""
    run_performance(csv_path::String; no_cache=false) -> Result

Run performance evaluation on a predictions CSV (must contain columns 'truth' and 'prediction').
"""
function run_performance(csv_path::String; no_cache::Bool = false)
  config = Dict{String,Any}("infile" => csv_path)
  cache = Cache("cache/performance", !no_cache)
  return run(performance_flow, config, cache = cache)
end

end
