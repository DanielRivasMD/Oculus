module DACLI

using ArgParse
using Avicenna.Flow: Cache, run
using ..DAFlow: deamination_workflow
using ..DACore: fname

export run_deamination

function run_deamination(args)
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
    "--csv"
    help = "Output CSV file"
    arg_type = String
    default = "out.csv"
    "--png"
    help = "Output PNG plot file"
    arg_type = String
    default = "out.png"
    "--verbose"
    help = "Print detailed information"
    action = :store_true
    "--no-cache"
    help = "Disable caching"
    action = :store_true
  end

  parsed = parse_args(args, s)

  config = Dict{String,Any}(
    "modern" => parsed["modern"],
    "ancient" => parsed["ancient"],
    "csv" => parsed["csv"],
    "png" => parsed["png"],
    "verbose" => parsed["verbose"],
    "modern_name" => fname(parsed["modern"]),
    "ancient_name" => fname(parsed["ancient"]),
  )

  cache = Cache("cache/deamination", !parsed["no-cache"])
  result = run(deamination_workflow, config, cache = cache)

  println("Deamination analysis complete.")
  println("CSV: ", config["csv"])
  println("Plot: ", config["png"])
  return result
end

end
