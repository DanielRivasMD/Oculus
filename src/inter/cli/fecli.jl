####################################################################################################

module FECLI

####################################################################################################

using ArgParse
using TOML
using Avicenna.Flow: Cache, launch
using ..FEFlow: flow
using ..FECore: file_hash

####################################################################################################

export run

####################################################################################################

function run(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--ancient"
    help = "Path to ancient FASTA file"
    arg_type = String
    required = true
    "--modern"
    help = "Path to modern FASTA file"
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
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end
  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "modern" => parsed["modern"],
    "ancient" => parsed["ancient"],
    "out" => parsed["out"],
    "onehot" => parsed["onehot"],
  )

  # TODO: Add file hashes for cache invalidation?
  config["_modern_hash"] = file_hash(config["modern"])
  config["_ancient_hash"] = file_hash(config["ancient"])

  cache = Cache("cache/feature", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["verbose"]
    println("Feature extraction complete")
    println("CSV: ", config["out"])
  end
  return result
end

####################################################################################################

end

####################################################################################################
