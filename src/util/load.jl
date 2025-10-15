####################################################################################################

using FASTX
using BioSequences
using Random
using StatsBase
using Flux: onehotbatch

####################################################################################################

# One-hot mapping: A,C,G,T -> columns 1..4; ambiguity (N) becomes zeros
nt2ix = Dict(DNA_A=>1, DNA_C=>2, DNA_G=>3, DNA_T=>4)

function onehot_encode(seq::LongDNA{4})
    L = length(seq)
    X = zeros(Float32, L, 4)
    @inbounds for (i, nt) in enumerate(seq)
        ix = get(nt2ix, nt, 0)
        if ix != 0
            X[i, ix] = 1.0f0
        end
    end
    return X
end

# Batch: right-pad to max length across seqs to form (maxL, 4, B)
function onehot_batch(seqs::Vector{LongDNA{4}})
    maxL = maximum(length, seqs)
    B = length(seqs)
    X = zeros(Float32, maxL, 4, B)
    @inbounds for (b, s) in enumerate(seqs)
        Xi = onehot_encode(s)
        L = size(Xi, 1)
        X[1:L, :, b] = Xi
    end
    return X  # (maxL, 4, B)
end

# Utility: load sequences from a FASTA file
function load_sequences_fasta(path::AbstractString)
    FASTA.Reader(open(path)) do reader
        [sequence(LongDNA{4}, record) for record in reader]
    end
end

####################################################################################################

# Data preparation
function load_balanced_data(params::SampleParams)
    modern  = load_sequences_fasta(params.modern)
    ancient = load_sequences_fasta(params.ancient)

    println("French:     $(length(modern)) reads")
    println("Neandertal: $(length(ancient)) reads")

    # Filter to fixed length
    modern_filt     = filter(seq -> length(seq) == params.seqlen, modern)
    ancient_filt = filter(seq -> length(seq) == params.seqlen, ancient)

    println("French $(params.seqlen)nt:     $(length(modern_filt))")
    println("Neandertal $(params.seqlen)nt: $(length(ancient_filt))")

    # Balance
    minN = min(length(modern_filt), length(ancient_filt))
    Random.seed!(params.seed)
    modern_bal = sample(modern_filt, minN; replace=false)
    ancient_bal  = sample(ancient_filt, minN; replace=false)

    println("Balanced French:     $(length(modern_bal))")
    println("Balanced Neandertal: $(length(ancient_bal))")

    all_seqs = vcat(modern_bal, ancient_bal)
    labels   = vcat(zeros(Int, length(modern_bal)),
                    ones(Int,  length(ancient_bal)))

    return all_seqs, labels
end

####################################################################################################

function make_dataset(sparams::SampleParams, hparams::CNNParams)
    all_seqs, labels = load_balanced_data(sparams)
    X = onehot_batch(all_seqs)
    Y = onehotbatch(labels, 0:1)

    B = size(X, 3)
    @assert size(Y, 2) == B "X/Y batch size mismatch"

    folds = split_indices(B, hparams, sparams)

    datasets = []
    for fold in folds
        train_idx, val_idx = fold.train, fold.val
        Xtrain, Ytrain = X[:, :, train_idx], Y[:, train_idx]
        Xval,   Yval   = X[:, :, val_idx],   Y[:, val_idx]
        push!(datasets, ((Xtrain, Ytrain), (Xval, Yval)))
    end

    return datasets, (B=B, L=size(X,1))
end

####################################################################################################

function split_indices(B::Int, hparams::CNNParams, sparams::SampleParams)
    Random.seed!(sparams.seed)
    idx = shuffle(1:B)

    if hparams.k == 0
        # Vanilla validation
        ntrain = round(Int, hparams.train_frac * B)
        return [(train=idx[1:ntrain], val=idx[ntrain+1:end])]
    else
        # k-fold CV
        foldsize = ceil(Int, B / hparams.k)
        return [(train=setdiff(idx, idx[((i-1)*foldsize+1):min(i*foldsize, B)]),
                 val=idx[((i-1)*foldsize+1):min(i*foldsize, B)])
                for i in 1:hparams.k]
    end
end

####################################################################################################
