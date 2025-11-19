"""
Load a FASTA file into a single sequence string.
Ignores header lines starting with '>'.
"""
function load_fasta(path::String)
  buf = IOBuffer()
  open(path) do f
    for line in eachline(f)
      if !startswith(line, '>')
        write(buf, strip(line))
      end
    end
  end
  return String(take!(buf))
end

"""
Compare two sequences base by base and report differences.
Also compute overall nucleotide composition for each file.
"""
function compare_sequences(seq1::String, seq2::String)
  if length(seq1) != length(seq2)
    println("Error: sequences differ in length ($(length(seq1)) vs $(length(seq2))).")
    return
  end

  L = length(seq1)
  println("Loaded sequences of length $L")

  # Count differences
  diffs = count(i -> seq1[i] != seq2[i], 1:L)

  println("Total positions compared: $L")
  println("Differences detected: $diffs")
  println("Fraction different: $(round(diffs/L, digits=4))")

  if diffs == 0
    println("Sequences are identical at all positions.")
  else
    println("Sequences differ at $diffs positions.")
  end

  # Composition summary
  function composition(seq::String)
    counts = Dict('A' => 0, 'C' => 0, 'G' => 0, 'T' => 0, 'N' => 0)
    for b in seq
      if haskey(counts, b)
        counts[b] += 1
      end
    end
    return counts
  end

  comp1 = composition(seq1)
  comp2 = composition(seq2)

  println("\nBase composition (sequence 1):")
  for (base, count) in comp1
    println("  $base: $count ($(round(count/L*100, digits=2))%)")
  end

  println("\nBase composition (sequence 2):")
  for (base, count) in comp2
    println("  $base: $count ($(round(count/L*100, digits=2))%)")
  end
end

# --- CLI ---
if abspath(PROGRAM_FILE) == @__FILE__
  if length(ARGS) < 2
    println("Usage: julia compare_fasta.jl seq1.fasta seq2.fasta")
    exit(1)
  end
  seq1 = load_fasta(ARGS[1])
  seq2 = load_fasta(ARGS[2])
  compare_sequences(seq1, seq2)
end
