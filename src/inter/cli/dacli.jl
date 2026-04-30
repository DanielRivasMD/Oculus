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
    "--single"
    help = "Single FASTA file (single‑file mode)"
    arg_type = String
    "--ancient"
    help = "Ancient FASTA file (dual‑file mode)"
    arg_type = String
    "--modern"
    help = "Modern FASTA file (dual‑file mode)"
    arg_type = String
    "--csv"
    help = "Output CSV file name"
    arg_type = String
    default = "out.csv"
    "--png"
    help = "Output PNG plot file name"
    arg_type = String
    default = "out.png"
    "--outdir"
    help = "Directory to write output files (default: current directory)"
    arg_type = String
    default = "."
    "--no-cache"
    help = "Disable caching"
    action = :store_true
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end

  parsed = parse_args(args, s)

  # Determine mode
  single = parsed["single"]
  ancient = parsed["ancient"]
  modern = parsed["modern"]

  if single != ""
    if ancient != "" || modern != ""
      error("Cannot use --single together with --ancient/--modern")
    end
    mode = :single
    config = Dict{String,Any}(
      "single" => single,
      "csv" => joinpath(parsed["outdir"], parsed["csv"]),
      "png" => joinpath(parsed["outdir"], parsed["png"]),
    )
  elseif ancient != "" && modern != ""
    mode = :dual
    config = Dict{String,Any}(
      "ancient" => ancient,
      "modern" => modern,
      "ancient_name" => fname(ancient),
      "modern_name" => fname(modern),
      "csv" => joinpath(parsed["outdir"], parsed["csv"]),
      "png" => joinpath(parsed["outdir"], parsed["png"]),
    )
  else
    error("Provide either --single or both --ancient and --modern")
  end

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
