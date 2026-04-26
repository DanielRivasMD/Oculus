module INCLI

using ArgParse
using Avicenna.Flow: Cache, launch
using ..INFlow: flow

export run

function run(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--model"
    help = "Path to trained model (BSON file)"
    arg_type = String
    required = true
    "--data"
    help = "FASTA/FASTQ file (or .gz) to classify"
    arg_type = String
    required = true
    "--out"
    help = "Output CSV file"
    arg_type = String
    default = "predictions.csv"
    "--no-cache"
    help = "Disable caching"
    action = :store_true
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end

  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "model" => parsed["model"],
    "data" => parsed["data"],
    "out" => parsed["out"],
  )

  cache = Cache("cache/inference", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["verbose"]
    println("Inference complete")
    println("Predictions written to $(config["out"])")
  end
  return result
end

end
