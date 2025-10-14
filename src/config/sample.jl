####################################################################################################

using Parameters: @with_kw

####################################################################################################

@with_kw mutable struct SampleParams
    seqlen::Int              = 50
    seed::Int                = 42
    modern::String           = joinpath(Paths.FASTA, "French_sample.fa")
    ancient::String          = joinpath(Paths.FASTA, "Neandertal_sample.fa")
end

####################################################################################################
