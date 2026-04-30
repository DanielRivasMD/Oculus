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
    run(;
        single::Union{String,Nothing}=nothing,
        ancient::Union{String,Nothing}=nothing,
        modern::Union{String,Nothing}=nothing,
        csv::String="out.csv",
        png::String="out.png",
        outdir::String=".",
        no_cache::Bool=false,
    ) -> Result

Run deamination analysis from the REPL

- Provide `single` for single‑file mode
- Provide both `ancient` and `modern` for two‑file comparison
"""
function run(;
             # TODO: use only string without union?
  single::Union{String,Nothing} = nothing,
  ancient::Union{String,Nothing} = nothing,
  modern::Union{String,Nothing} = nothing,
  csv::String = "out.csv",
  png::String = "out.png",
  outdir::String = ".",
  no_cache::Bool = false,
)
  if single !== nothing
    if ancient !== nothing || modern !== nothing
      error("Cannot use single together with ancient/modern")
    end
    config = Dict{String,Any}(
      "single" => single,
      "csv" => joinpath(outdir, csv),
      "png" => joinpath(outdir, png),
    )
  elseif ancient !== nothing && modern !== nothing
    config = Dict{String,Any}(
      "ancient" => ancient,
      "modern" => modern,
      "ancient_name" => fname(ancient),
      "modern_name" => fname(modern),
      "csv" => joinpath(outdir, csv),
      "png" => joinpath(outdir, png),
    )
  else
    error("Provide either single or both ancient and modern")
  end

  cache = Cache("cache/deamination", !no_cache)
  return launch(flow, config, cache = cache)
end

####################################################################################################

end

####################################################################################################
