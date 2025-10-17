####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    SampleParams(; kwargs...)

Configuration container for dataset sampling and preprocessing.

This struct defines sequence‑level parameters and file paths for reproducible
data loading. It is constructed with `@with_kw`, so any field can be overridden
at creation time via keyword arguments.

# Fields
- `seqlen::Int = 50`  
  Target sequence length (in nucleotides). Sequences are filtered to this
  exact length before training.

- `seed::Int = 42`  
  Random seed used for reproducible shuffling, sampling, and train/validation
  splits.

- `datadir::String = Paths.FASTA`  
  Base directory where FASTA files are stored.

- `modern::String = joinpath(datadir, "French_sample.fasta")`  
  Path to the FASTA file containing modern DNA sequences.

- `ancient::String = joinpath(datadir, "Neandertal_sample.fasta")`  
  Path to the FASTA file containing ancient DNA sequences.

# Usage
```julia
sparams = SampleParams(seqlen=75,
                       seed=123,
                       datadir="data/fasta",
                       modern=joinpath("data/fasta", "modern.fasta"),
                       ancient=joinpath("data/fasta", "ancient.fasta"))
"""
@with_kw mutable struct SampleParams
  seqlen::Int = 37
  seed::Int = 42
  datadir::String = Paths.FASTA
  modern::String = joinpath(datadir, "French_37nt_head.fasta")
  ancient::String = joinpath(datadir, "Neandertal_37nt_head.fasta")
end

####################################################################################################
