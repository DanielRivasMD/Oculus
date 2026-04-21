####################################################################################################

module DAREPL

####################################################################################################

using Avicenna.Flow: Cache, launch
using ..DAFlow: flow
using ..DACore: fname, compare_fasta_files, plot_composition

####################################################################################################

export run, compare_fasta_files, plot_composition

####################################################################################################

"""
    run_deamination(
      ancient::String,
      modern::String,
      csv::String="out.csv",
      png::String="out.png",
      no_cache::Bool=false) -> Result

Run the deamination analysis workflow from the REPL.
"""
function run(
  ancient::String;
  modern::String,
  csv::String = "out.csv",
  png::String = "out.png",
  no_cache::Bool = false,
)
  config = Dict{String,Any}(
    "ancient" => ancient,
    "modern" => modern,
    "csv" => csv,
    "png" => png,
    "ancient_name" => fname(ancient),
    "modern_name" => fname(modern),
  )
  cache = Cache("cache/deamination", !no_cache)
  return launch(flow, config, cache = cache)
end

####################################################################################################

end

####################################################################################################
