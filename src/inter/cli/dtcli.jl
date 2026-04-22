####################################################################################################

module DTCLI

####################################################################################################

using ArgParse
using Avicenna.Flow: Cache, launch
using ..DTFlow: flow

####################################################################################################

export run

####################################################################################################

function run(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--in"
    help = "Input CSV file"
    arg_type = String
    required = true
    "--out"
    help = "Output CSV file for predictions"
    arg_type = String
    default = nothing
    "--model"
    help = "Model type: tree, forest, xgboost"
    arg_type = String
    default = "tree"
    "--split"
    help = "Fraction of data to use as test set (0.0 = no split)"
    arg_type = Float64
    default = 0.2
    "--seed"
    help = "Random seed"
    arg_type = Int
    default = 42
    "--max_depth"
    help = "Maximum tree depth (for tree/forest)"
    arg_type = Int
    default = 6
    "--min_samples_leaf"
    help = "Minimum samples per leaf (for tree/forest)"
    arg_type = Int
    default = 5
    "--n_trees"
    help = "Number of trees (for forest)"
    arg_type = Int
    default = 100
    "--rf_partial_sampling"
    help = "Fraction of samples used per tree (for forest)"
    arg_type = Float64
    default = 0.7
    "--xgb_rounds"
    help = "Number of boosting rounds (for xgboost)"
    arg_type = Int
    default = 200
    "--xgb_eta"
    help = "Learning rate (for xgboost)"
    arg_type = Float64
    default = 0.1
    "--xgb_max_depth"
    help = "Maximum depth of XGBoost trees"
    arg_type = Int
    default = 6
    "--xgb_subsample"
    help = "Subsample ratio for XGBoost"
    arg_type = Float64
    default = 0.8
    "--xgb_colsample_bytree"
    help = "Column subsample ratio for XGBoost"
    arg_type = Float64
    default = 0.8
    "--no-cache"
    help = "Disable caching"
    action = :store_true
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end

  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "infile" => parsed["in"],
    "out" => parsed["out"],
    "model" => parsed["model"],
    "split" => parsed["split"],
    "seed" => parsed["seed"],
    "max_depth" => parsed["max_depth"],
    "min_samples_leaf" => parsed["min_samples_leaf"],
    "n_trees" => parsed["n_trees"],
    "rf_partial_sampling" => parsed["rf_partial_sampling"],
    "xgb_rounds" => parsed["xgb_rounds"],
    "xgb_eta" => parsed["xgb_eta"],
    "xgb_max_depth" => parsed["xgb_max_depth"],
    "xgb_subsample" => parsed["xgb_subsample"],
    "xgb_colsample_bytree" => parsed["xgb_colsample_bytree"],
  )

  cache = Cache("cache/decision_tree", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["verbose"]
    if parsed["out"] !== nothing
      println("Predictions: ", parsed["out"])
    end
    if !isempty(result.stage_outputs["05_evaluate"])
      println("Evaluation metrics:")
      for (k, v) in result.stage_outputs["05_evaluate"]
        println("  $k: $v")
      end
    end
  end
  return result
end

####################################################################################################

end

####################################################################################################
