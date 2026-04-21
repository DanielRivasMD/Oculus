####################################################################################################

module RGCLI

####################################################################################################

using ArgParse
using Avicenna.Flow: Cache, launch
using ..RGFlow: flow

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
    "--reg"
    help = "Regularization method: none, ridge, lasso, elasticnet"
    arg_type = String
    default = "none"
    "--alpha"
    help = "Elastic net mixing parameter (0<alpha<1)"
    arg_type = Float64
    default = 0.5
    "--nfolds"
    help = "Number of cross‑validation folds for GLMNet"
    arg_type = Int
    default = 10
    "--split"
    help = "Fraction of data to use as test set (0.0 = no split)"
    arg_type = Float64
    default = 0.6
    "--seed"
    help = "Random seed"
    arg_type = Int
    default = 42
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
    "reg" => parsed["reg"],
    "alpha" => parsed["alpha"],
    "nfolds" => parsed["nfolds"],
    "split" => parsed["split"],
    "seed" => parsed["seed"],
  )

  cache = Cache("cache/regression", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["out"] !== nothing
    println("Predictions written to ", parsed["out"])
  end
  if !parsed["verbose"]
    if !isempty(result.stage_outputs["evaluate"])
      println("Evaluation metrics:")
      for (k, v) in result.stage_outputs["evaluate"]
        println("  $k: $v")
      end
    end
  end
  return result
end

####################################################################################################

end

####################################################################################################
