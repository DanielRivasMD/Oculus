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

- `modern::String = joinpath(Paths.FASTA, "French_sample.fa")`  
  Path to the FASTA file containing modern DNA sequences.

- `ancient::String = joinpath(Paths.FASTA, "Neandertal_sample.fa")`  
  Path to the FASTA file containing ancient DNA sequences.

# Usage
```julia
sparams = SampleParams(seqlen=75,
                       seed=123,
                       modern="data/modern.fa",
                       ancient="data/ancient.fa")

"""
@with_kw mutable struct SampleParams
    seqlen::Int              = 50
    seed::Int                = 42
    modern::String           = joinpath(Paths.FASTA, "French_sample.fa")
    ancient::String          = joinpath(Paths.FASTA, "Neandertal_sample.fa")
end

####################################################################################################
