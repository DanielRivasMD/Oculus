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
args = extract_args()

####################################################################################################
# Imports
####################################################################################################

using Random
using FilePathsBase: basename, splitext

####################################################################################################
# Helper functions
####################################################################################################

"""
    extract_reads(genome_fasta, out_fasta; num_reads=1000, read_len=76)

Extract Illumina‑like reads by sampling random substrings from a genome FASTA.
"""
function extract_reads(
  genome_fasta::String,
  out_fasta::String;
  num_reads::Int = 1000,
  read_len::Int = 76,
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

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing
  genome_fasta = args["genome"]
  out_fasta = args["out"]
  num_reads = args["num_reads"]
  read_len = args["read_len"]

  extract_reads(genome_fasta, out_fasta; num_reads = num_reads, read_len = read_len)
end

####################################################################################################
