# Simulate Illumina-like reads from a genome FASTA

using Random

"""
simulate_reads(genome_fasta::String, out_fasta::String; read_len::Int=75, num_reads::Int=1000)

Reads a genome FASTA file, samples random substrings of length `read_len`,
and writes them out as FASTA records to `out_fasta`.
"""
function simulate_reads(
  genome_fasta::String,
  out_fasta::String;
  num_reads::Int = 1000,
  read_len::Int = 75,
)
  # Read genome into a single string
  genome = IOBuffer()
  open(genome_fasta) do f
    for line in eachline(f)
      if !startswith(line, '>')
        write(genome, strip(line))
      end
    end
  end
  genome_seq = String(take!(genome))
  genome_size = length(genome_seq)
  println("Genome size: $genome_size bases")

  open(out_fasta, "w") do out
    for i = 1:num_reads
      start = rand(1:(genome_size-read_len+1))
      read = genome_seq[start:start+read_len-1]
      println(out, ">read$i")
      println(out, read)
    end
  end

  println("Generated $num_reads reads of length $read_len → $out_fasta")
end

# Command-line interface
if abspath(PROGRAM_FILE) == @__FILE__
  if length(ARGS) < 2
    println(
      "Usage: julia simulate_reads.jl genome.fasta output.fasta [read_len] [num_reads]",
    )
    exit(1)
  end
  genome_fasta = ARGS[1]
  out_fasta = ARGS[2]
  num_reads = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
  read_len = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 75

  simulate_reads(genome_fasta, out_fasta; num_reads = num_reads, read_len = read_len)
end
