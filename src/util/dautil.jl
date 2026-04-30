####################################################################################################

module DACore

####################################################################################################

using DelimitedFiles
using Plots
using FilePathsBase: splitpath, basename
using ..USCore: load_fasta

####################################################################################################

export position_composition,
  write_csv_single, write_csv_dual, plot_composition, compare_fasta_files, fname

####################################################################################################

fname(path::String) = splitpath(path)[end]

####################################################################################################

"""
    position_composition(seqs::Vector{String}) -> Vector{Dict{Char,Float64}}

Compute per‑position percentages of A, T, G, C, N for a list of sequences of equal length
"""
function position_composition(seqs::Vector{String})::Vector{Dict{Char,Float64}}
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

####################################################################################################

"""
    write_csv_single(outpath::String, comp::Vector{Dict{Char,Float64}})

Write a CSV with columns: position, A, T, G, C
"""
function write_csv_single(outpath::String, comp::Vector{Dict{Char,Float64}})
  L = length(comp)
  data = Matrix{Float64}(undef, L, 5)
  for i = 1:L
    data[i, 1] = i
    data[i, 2] = comp[i]['A']
    data[i, 3] = comp[i]['T']
    data[i, 4] = comp[i]['G']
    data[i, 5] = comp[i]['C']
  end
  header = hcat("position", "A", "T", "G", "C")
  writedlm(outpath, vcat(header, data), ',')
end

####################################################################################################

"""
    write_csv_dual(outpath::String,
                   modern_name::String,
                   ancient_name::String,
                   comp1::Vector{Dict{Char,Float64}},
                   comp2::Vector{Dict{Char,Float64}})

Write the combined CSV for two‑file comparison
"""
function write_csv_dual(
  outpath::String,
  modern_name::String,
  ancient_name::String,
  comp1::Vector{Dict{Char,Float64}},
  comp2::Vector{Dict{Char,Float64}},
)
  L = length(comp1)
  data = Matrix{Float64}(undef, L, 9)
  for i = 1:L
    data[i, 1] = i
    data[i, 2] = comp1[i]['A']
    data[i, 3] = comp1[i]['T']
    data[i, 4] = comp1[i]['G']
    data[i, 5] = comp1[i]['C']
    data[i, 6] = comp2[i]['A']
    data[i, 7] = comp2[i]['T']
    data[i, 8] = comp2[i]['G']
    data[i, 9] = comp2[i]['C']
  end

  header = hcat(
    "position",
    modern_name * "_A",
    modern_name * "_T",
    modern_name * "_G",
    modern_name * "_C",
    ancient_name * "_A",
    ancient_name * "_T",
    ancient_name * "_G",
    ancient_name * "_C",
  )

  writedlm(outpath, vcat(header, data), ',')
end

####################################################################################################

"""
    plot_composition(csvfile::String; outfile::Union{Nothing,String}=nothing)

Generate a line plot of per‑position base composition
Detects single‑file (5 columns) vs two‑file (9 columns) format automatically
"""
function plot_composition(csvfile::String; outfile::Union{Nothing,String} = nothing)
  raw = readdlm(csvfile, ',', String)
  header = raw[1, :]
  data = raw[2:end, :]
  positions = parse.(Float64, data[:, 1])

  ncols = size(data, 2)
  if ncols == 5   # single file
    groups = [("Sample", :solid)]
    bases = ["A", "T", "G", "C"]
    cols_offset = 1
  elseif ncols == 9   # two files
    modern_name = replace(header[2], r"_A$" => "")
    ancient_name = replace(header[6], r"_A$" => "")
    groups = [(modern_name, :solid), (ancient_name, :dash)]
    bases = ["A", "T", "G", "C"]
    cols_offset = 1
  else
    error("Unsupported CSV format: expected 5 or 9 columns")
  end

  colors = Dict("A" => "green", "T" => "red", "G" => "orange", "C" => "blue")
  plt = plot(
    ylim = (15, 35),
    size = (1000, 600),
    xlabel = "Position",
    ylabel = "Percentage (%)",
    title = "Base Composition Along Sequence",
  )

  for (group_name, linestyle) in groups
    for (j, base) in enumerate(bases)
      col_idx = cols_offset + (groups[1] == (group_name, linestyle) ? j : 4 + j - 1)
      col = parse.(Float64, data[:, col_idx])
      plot!(
        plt,
        positions,
        col,
        label = "$group_name $base",
        color = colors[base],
        linestyle = linestyle,
      )
    end
  end

  if outfile !== nothing
    savefig(plt, outfile)
    println("Plot saved to $outfile")
  else
    display(plt)
  end
end

####################################################################################################

"""
    compare_fasta_files(modern::String, ancient::String; csv_out::String, verbose::Bool=false)

Load two FASTA files, compute per‑position composition, write CSV, and report differences
"""
function compare_fasta_files(
  modern::String,
  ancient::String;
  csv_out::String,
  verbose::Bool = false,
)
  seqs_modern = load_fasta(modern)
  seqs_ancient = load_fasta(ancient)

  verbose && println("Modern ($(fname(modern))) sequences: $(length(seqs_modern))")
  verbose && println("Ancient ($(fname(ancient))) sequences: $(length(seqs_ancient))")

  isempty(seqs_modern) && error("Modern FASTA contains no sequences")
  isempty(seqs_ancient) && error("Ancient FASTA contains no sequences")
  length(seqs_modern[1]) == length(seqs_ancient[1]) || error("Sequence lengths differ")

  comp_modern = position_composition(seqs_modern)
  comp_ancient = position_composition(seqs_ancient)

  L = length(comp_modern)
  diffs = sum(1:L) do i
    (
      comp_modern[i]['A'] != comp_ancient[i]['A'] ||
      comp_modern[i]['T'] != comp_ancient[i]['T'] ||
      comp_modern[i]['G'] != comp_ancient[i]['G'] ||
      comp_modern[i]['C'] != comp_ancient[i]['C']
    )
  end
  frac = round(diffs / L, digits = 4)
  println("Positions with differing ATGC distributions: $diffs / $L (fraction=$frac)")

  write_csv_dual(csv_out, fname(modern), fname(ancient), comp_modern, comp_ancient)
  println("CSV written to $csv_out")
end

####################################################################################################

end

####################################################################################################
