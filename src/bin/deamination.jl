#!/usr/bin/env julia

# --- utilities ---

# File basename without directories
fname(path::String) = splitpath(path)[end]

# Load a FASTA file into a vector of sequences, concatenating multi-line records.
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

# Compute per-position nucleotide percentages across all sequences (A,T,G,C,N).
# Assumes all sequences have the same length.
function position_composition(seqs::Vector{String})
  isempty(seqs) && error("No sequences found.")
  L = length(seqs[1])
  for s in seqs
    length(s) == L || error("Sequences differ in length")
  end

  comps = [
    Dict{Char,Float64}('A' => 0.0, 'T' => 0.0, 'G' => 0.0, 'C' => 0.0, 'N' => 0.0) for
    _ = 1:L
  ]

  for s in seqs
    for i = 1:L
      b = s[i]
      if haskey(comps[i], b)
        comps[i][b] += 1.0
      else
        comps[i]['N'] += 1.0
      end
    end
  end

  total = length(seqs)
  inv_total = 100.0 / total
  for i = 1:L
    for k in keys(comps[i])
      comps[i][k] *= inv_total
    end
  end

  return comps
end

# Write CSV: position, file1_A, file1_T, file1_G, file1_C, file2_A, file2_T, file2_G, file2_C
function write_csv(outpath::String, file1::String, file2::String, comp1, comp2)
  open(outpath, "w") do io
    println(
      io,
      "position,$(fname(file1))_A,$(fname(file1))_T,$(fname(file1))_G,$(fname(file1))_C,$(fname(file2))_A,$(fname(file2))_T,$(fname(file2))_G,$(fname(file2))_C",
    )
    L = length(comp1)
    for i = 1:L
      f1A = comp1[i]['A']
      f1T = comp1[i]['T']
      f1G = comp1[i]['G']
      f1C = comp1[i]['C']
      f2A = comp2[i]['A']
      f2T = comp2[i]['T']
      f2G = comp2[i]['G']
      f2C = comp2[i]['C']
      println(io, "$i,$f1A,$f1T,$f1G,$f1C,$f2A,$f2T,$f2G,$f2C")
    end
  end
end

# Compare and optionally write CSV
function compare_fasta_files(
  file1::String,
  file2::String;
  csv_out::Union{Nothing,String} = nothing,
  verbose::Bool = false,
)
  seqs1 = load_fasta(file1)
  seqs2 = load_fasta(file2)

  if verbose
    println("File1 ($(fname(file1))) sequences: $(length(seqs1))")
    println("File2 ($(fname(file2))) sequences: $(length(seqs2))")
  end

  isempty(seqs1) && error("File1 contains no sequences")
  isempty(seqs2) && error("File2 contains no sequences")

  length(seqs1[1]) == length(seqs2[1]) || error("Sequence lengths differ between files")

  comp1 = position_composition(seqs1)
  comp2 = position_composition(seqs2)

  # Count positions where ATGC distributions differ (ignore N)
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

  println(
    "Positions with differing ATGC distributions: $diffs / $L (fraction=$(round(diffs/L, digits=4)))",
  )

  if csv_out !== nothing
    write_csv(csv_out, file1, file2, comp1, comp2)
    println("CSV written to $(csv_out)")
  end
end

# --- main / CLI parsing ---

function main()
  if length(ARGS) < 2
    println(
      "Usage: julia deamination.jl file1.fasta file2.fasta [--csv out.csv] [--verbose]",
    )
    exit(1)
  end
  file1 = ARGS[1]
  file2 = ARGS[2]

  csv_out = nothing
  verbose = false

  idx = 3
  while idx <= length(ARGS)
    arg = ARGS[idx]
    if arg == "--csv"
      if idx + 1 > length(ARGS)
        error("--csv requires an output filename")
      end
      csv_out = ARGS[idx+1]
      idx += 2
    elseif arg == "--verbose"
      verbose = true
      idx += 1
    else
      error("Unexpected argument: $arg")
    end
  end

  compare_fasta_files(file1, file2; csv_out = csv_out, verbose = verbose)
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
