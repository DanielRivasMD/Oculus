module DAREPL

using Avicenna.Flow: Cache, run
using ..DAFlow: deamination_flow
using ..DACore: fname, compare_fasta_files, plot_composition

export run_deamination, compare_fasta_files, plot_composition

"""
    run_deamination(
      modern::String,
      ancient::String,
      csv::String="out.csv",
      png::String="out.png",
      verbose::Bool=false,
      no_cache::Bool=false) -> Result

Run the deamination analysis workflow from the REPL.
"""
function run_deamination(
  modern::String,
  ancient::String;
  csv::String = "out.csv",
  png::String = "out.png",
  verbose::Bool = false,
  no_cache::Bool = false,
)
  config = Dict{String,Any}(
    "modern" => modern,
    "ancient" => ancient,
    "csv" => csv,
    "png" => png,
    "verbose" => verbose,
    "modern_name" => fname(modern),
    "ancient_name" => fname(ancient),
  )
  cache = Cache("cache/deamination", !no_cache)
  return run(deamination_flow, config, cache = cache)
end

end
