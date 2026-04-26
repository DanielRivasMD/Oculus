module NNCLI

using ArgParse
using Avicenna.Flow: Cache, launch
using ..NNFlow: flow

export run

function run(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--cnn"
    help = "Path to CNN hyperparameters TOML file"
    arg_type = String
    required = true
    "--sample"
    help = "Path to sample TOML file"
    arg_type = String
    required = true
    "--out"
    help = "Base path for output BSON model (fold suffix and timestamp will be added)"
    arg_type = String
    default = "model.bson"
    "--no-cache"
    help = "Disable caching"
    action = :store_true
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end

  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "cnn" => parsed["cnn"],
    "sample" => parsed["sample"],
    "out" => parsed["out"],
  )

  cache = Cache("cache/nn", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["verbose"]
    println("Neural network training complete")
    metrics = result.stage_outputs["02_train"].metrics
    paths = result.stage_outputs["02_train"].model_paths
    println("Models saved to: ", join(paths, ", "))
    println("Metrics of last fold:")
    for (k, v) in metrics
      println("  $k: $v")
    end
  end
  return result
end

end
