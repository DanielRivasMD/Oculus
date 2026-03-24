module DTREPL

using Avicenna.Flow: Cache, run
using ..DTFlow: decision_tree_flow

export run_decision_tree

"""
    run_decision_tree(infile::String; out=nothing, model="tree", split=0.2, seed=42,
                      max_depth=6, min_samples_leaf=5, n_trees=100, rf_partial_sampling=0.7,
                      xgb_rounds=200, xgb_eta=0.1, xgb_max_depth=6, xgb_subsample=0.8,
                      xgb_colsample_bytree=0.8, no_cache=false) -> WorkflowResult

Run the decision tree/random forest/XGBoost workflow from the REPL.
"""
function run_decision_tree(
  infile::String;
  out::Union{String,Nothing} = nothing,
  model::String = "tree",
  split::Float64 = 0.2,
  seed::Int = 42,
  max_depth::Int = 6,
  min_samples_leaf::Int = 5,
  n_trees::Int = 100,
  rf_partial_sampling::Float64 = 0.7,
  xgb_rounds::Int = 200,
  xgb_eta::Float64 = 0.1,
  xgb_max_depth::Int = 6,
  xgb_subsample::Float64 = 0.8,
  xgb_colsample_bytree::Float64 = 0.8,
  no_cache::Bool = false,
)
  config = Dict{String,Any}(
    "infile" => infile,
    "out" => out,
    "model" => model,
    "split" => split,
    "seed" => seed,
    "max_depth" => max_depth,
    "min_samples_leaf" => min_samples_leaf,
    "n_trees" => n_trees,
    "rf_partial_sampling" => rf_partial_sampling,
    "xgb_rounds" => xgb_rounds,
    "xgb_eta" => xgb_eta,
    "xgb_max_depth" => xgb_max_depth,
    "xgb_subsample" => xgb_subsample,
    "xgb_colsample_bytree" => xgb_colsample_bytree,
  )
  cache = Cache("cache/decision_tree", !no_cache)
  return run(decision_tree_flow, config, cache = cache)
end

end
