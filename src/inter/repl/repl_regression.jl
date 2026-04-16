module RGREPL

using Avicenna.Flow: Cache, run
using ..RGFlow: regression_flow

export run_regression

"""
    run_regression(infile::String; out=nothing, reg="none", alpha=0.5, nfolds=10, split=0.6, seed=42, no_cache=false)

Run the regression workflow from the REPL.
"""
function run_regression(
  infile::String;
  out::Union{String,Nothing} = nothing,
  reg::String = "none",
  alpha::Float64 = 0.5,
  nfolds::Int = 10,
  split::Float64 = 0.6,
  seed::Int = 42,
  no_cache::Bool = false,
)
  config = Dict{String,Any}(
    "infile" => infile,
    "out" => out,
    "reg" => reg,
    "alpha" => alpha,
    "nfolds" => nfolds,
    "split" => split,
    "seed" => seed,
  )
  cache = Cache("cache/regression", !no_cache)
  return run(regression_flow, config, cache = cache)
end

end
