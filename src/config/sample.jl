####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    SampleParams(; kwargs...)

Configuration container for dataset sampling and preprocessing.

# Fields
- `seqlen::Int = 37`
- `seed::Int = 42`
- `datadir::String = Paths.FASTA`
- `modern::String = joinpath(datadir, "French_37nt_head.fasta")`
- `ancient::String = joinpath(datadir, "Neandertal_37nt_head.fasta")`
"""
@with_kw mutable struct SampleParams
    seqlen::Int     = 37
    seed::Int       = 42
    datadir::String = Paths.FASTA
    modern::String  = joinpath(datadir, "French_37nt_head.fasta")
    ancient::String = joinpath(datadir, "Neandertal_37nt_head.fasta")
end

####################################################################################################
