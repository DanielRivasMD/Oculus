####################################################################################################
# cli args
####################################################################################################

begin
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.UTIL, "args.jl"))
end

# Parse CLI arguments
args = deamination_args()

####################################################################################################
# Imports
####################################################################################################

using FilePathsBase: basename, splitext
using DelimitedFiles
using Plots

####################################################################################################
# Utilities
####################################################################################################

fname(path::String) = splitpath(path)[end]

# Load FASTA into vector of sequences
function load_fasta(path::String)
  seqs = String[]
  buf = IOBuffer()
  open(path) do f
    for line in eachline(f)
      if startswith(line, '>')
        if position(buf) > 0
          push!(seqs, String(take!(buf)))
        end
      else
        write(buf, strip(line))
      end
    end
    if position(buf) > 0
      push!(seqs, String(take!(buf)))
    end
  end
  return seqs
end

# Compute per-position ATGC percentages
function position_composition(seqs::Vector{String})
  isempty(seqs) && error("No sequences found.")
  L = length(seqs[1])
  for s in seqs
    length(s) == L || error("Sequences differ in length")
  end

  comps = [Dict('A' => 0.0, 'T' => 0.0, 'G' => 0.0, 'C' => 0.0, 'N' => 0.0) for _ = 1:L]

  for s in seqs
    for i = 1:L
      b = s[i]
      if haskey(comps[i], b)
        comps[i][b] += 1
      else
        comps[i]['N'] += 1
      end
    end
  end

  total = length(seqs)
  for i = 1:L
    for k in keys(comps[i])
      comps[i][k] = 100 * comps[i][k] / total
    end
  end

  return comps
end

# Write CSV with ATGC percentages
function write_csv(outpath::String, modern::String, ancient::String, comp1, comp2)
  open(outpath, "w") do io
    println(
      io,
      "position,$(fname(modern))_A,$(fname(modern))_T,$(fname(modern))_G,$(fname(modern))_C," *
      "$(fname(ancient))_A,$(fname(ancient))_T,$(fname(ancient))_G,$(fname(ancient))_C",
    )
    L = length(comp1)
    for i = 1:L
      println(
        io,
        "$i,$(comp1[i]['A']),$(comp1[i]['T']),$(comp1[i]['G']),$(comp1[i]['C'])," *
        "$(comp2[i]['A']),$(comp2[i]['T']),$(comp2[i]['G']),$(comp2[i]['C'])",
      )
    end
  end
end

####################################################################################################
# Plotting
####################################################################################################

function plot_composition(csvfile::String; outfile::Union{Nothing,String} = nothing)
  raw = readdlm(csvfile, ',', String)
  header = raw[1, :]
  data = raw[2:end, :]

  positions = parse.(Int, data[:, 1])

  # Modern and Ancient labels
  modern_label = "Modern"
  ancient_label = "Ancient"

  colors = Dict("A" => "green", "T" => "red", "G" => "orange", "C" => "blue")

  plt = plot(
    size = (1000, 600),
    xlabel = "Position",
    ylabel = "Percentage (%)",
    title = "Base Composition Along Sequence",
  )

  # Modern (solid)
  for (j, base) in enumerate(["A", "T", "G", "C"])
    col = parse.(Float64, data[:, 1+j])
    plot!(
      plt,
      positions,
      col,
      label = "$(modern_label) $base",
      color = colors[base],
      linestyle = :solid,
    )
  end

  # Ancient (dashed)
  for (j, base) in enumerate(["A", "T", "G", "C"])
    col = parse.(Float64, data[:, 5+j])
    plot!(
      plt,
      positions,
      col,
      label = "$(ancient_label) $base",
      color = colors[base],
      linestyle = :dash,
    )
  end

  if outfile !== nothing
    savefig(plt, outfile)
    println("Plot saved to $outfile")
  else
    display(plt)
  end
end

####################################################################################################
# Core comparison logic
####################################################################################################

function compare_fasta_files(modern::String, ancient::String; csv_out, verbose)
  seqs1 = load_fasta(modern)
  seqs2 = load_fasta(ancient)

  verbose && println("Modern ($(fname(modern))) sequences: $(length(seqs1))")
  verbose && println("Ancient ($(fname(ancient))) sequences: $(length(seqs2))")

  isempty(seqs1) && error("Modern FASTA contains no sequences")
  isempty(seqs2) && error("Ancient FASTA contains no sequences")

  length(seqs1[1]) == length(seqs2[1]) || error("Sequence lengths differ between files")

  comp1 = position_composition(seqs1)
  comp2 = position_composition(seqs2)

  L = length(comp1)
  diffs = 0
  for i = 1:L
    if comp1[i]['A'] != comp2[i]['A'] ||
       comp1[i]['T'] != comp2[i]['T'] ||
       comp1[i]['G'] != comp2[i]['G'] ||
       comp1[i]['C'] != comp2[i]['C']
      diffs += 1
    end
  end

  frac = round(diffs / L, digits = 4)
  println("Positions with differing ATGC distributions: $diffs / $L (fraction=$frac)")

  write_csv(csv_out, modern, ancient, comp1, comp2)
  println("CSV written to $csv_out")
end

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing
  modern = args["modern"]
  ancient = args["ancient"]
  csv_out = args["csv"]
  png_out = args["png"]
  verbose = args["verbose"]

  compare_fasta_files(modern, ancient; csv_out = csv_out, verbose = verbose)
  plot_composition(csv_out; outfile = png_out)
end

####################################################################################################
