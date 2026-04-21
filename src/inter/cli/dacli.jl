####################################################################################################

module DACLI

####################################################################################################

using ArgParse
using Avicenna.Flow: Cache, launch
using ..DAFlow: flow
using ..DACore: fname

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
    "--csv"
    help = "Output CSV file"
    arg_type = String
    default = "out.csv"
    "--png"
    help = "Output PNG plot file"
    arg_type = String
    default = "out.png"
    "--no-cache"
    help = "Disable caching"
    action = :store_true
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end

  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "ancient" => parsed["ancient"],
    "modern" => parsed["modern"],
    "csv" => parsed["csv"],
    "png" => parsed["png"],
    "ancient_name" => fname(parsed["ancient"]),
    "modern_name" => fname(parsed["modern"]),
  )

  cache = Cache("cache/deamination", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["verbose"]
    println("Deamination analysis complete")
    println("CSV: ", config["csv"])
    println("Plot: ", config["png"])
  end
  return result
end

####################################################################################################

end

####################################################################################################
