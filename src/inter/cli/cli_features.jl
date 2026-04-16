####################################################################################################

module FECLI

####################################################################################################

using ArgParse
using TOML
using Avicenna.Flow: Cache, run
using ..FEFlow: features_flow
using ..FECore: file_hash

####################################################################################################

export run_features

####################################################################################################

function run_features(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--modern"
    help = "Path to modern FASTA file"
    arg_type = String
    required = true
    "--ancient"
    help = "Path to ancient FASTA file"
    arg_type = String
    required = true
    "--out"
    help = "Output CSV file"
    arg_type = String
    default = "features.csv"
    "--onehot"
    help = "Use one-hot encoding (default: false)"
    action = :store_true
    "--no-cache"
    help = "Disable caching"
    action = :store_true
  end
  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "modern" => parsed["modern"],
    "ancient" => parsed["ancient"],
    "out" => parsed["out"],
    "onehot" => parsed["onehot"],
  )

  # Add file hashes for cache invalidation
  config["_modern_hash"] = file_hash(config["modern"])
  config["_ancient_hash"] = file_hash(config["ancient"])

  cache = Cache("cache/feature", !parsed["no-cache"])
  result = run(features_flow, config, cache = cache)

  println("Feature extraction complete. Output written to ", config["out"])
  return result
end

####################################################################################################

end

####################################################################################################
