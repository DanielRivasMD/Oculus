####################################################################################################

using FASTX
using BioSequences
using Random
using StatsBase
using Flux: onehotbatch

####################################################################################################

# One-hot mapping: A,C,G,T -> columns 1..4; ambiguity (N) becomes zeros
nt2ix = Dict(DNA_A=>1, DNA_C=>2, DNA_G=>3, DNA_T=>4)

"""
    onehot_encode(seq::LongDNA{4}) -> Matrix{Float32}

Convert a DNA sequence into a one‑hot encoded matrix.

Each nucleotide is mapped to one of four columns:
- A → column 1
- C → column 2
- G → column 3
- T → column 4
Ambiguous bases (e.g. N) are encoded as all zeros.

# Arguments
- `seq::LongDNA{4}` : DNA sequence.

# Returns
- `Matrix{Float32}` of shape `(L, 4)`, where `L` is the sequence length.

# Example
```julia
seq = dna"ACGTN"
X = onehot_encode(seq)
size(X)  # (5, 4)
"""
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

"""
    onehot_batch(seqs::Vector{LongDNA{4}}) -> Array{Float32,3}

Batch one‑hot encode a collection of DNA sequences.

Each sequence is converted with [`onehot_encode`](@ref) and then right‑padded
with zeros so that all sequences match the maximum length in the batch.
The result is a 3‑dimensional tensor suitable for input into a CNN.

# Arguments
- `seqs::Vector{LongDNA{4}}`  
  Collection of DNA sequences to encode.

# Returns
- `Array{Float32,3}` of shape `(maxL, 4, B)` where:
  - `maxL` = maximum sequence length across all sequences in the batch
  - `4`    = one‑hot channels (A, C, G, T)
  - `B`    = number of sequences in the batch

# Notes
- Ambiguous bases (e.g. `N`) are encoded as all‑zero rows.
- Padding ensures consistent tensor shapes for batching.

# Example
```julia
seqs = [dna"ACGT", dna"AC"]
X = onehot_batch(seqs)
size(X)  # (4, 4, 2)
"""
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

"""
    load_sequences_fasta(path::AbstractString) -> Vector{LongDNA{4}}

Load DNA sequences from a FASTA file into `LongDNA{4}` objects.

This function uses `FASTX.FASTA.Reader` to parse the file and converts
each record into a `LongDNA{4}` sequence from BioSequences.jl. It is
intended for preparing raw sequence data for downstream one‑hot encoding
and model training.

# Arguments
- `path::AbstractString`  
  Path to a FASTA file containing DNA sequences.

# Returns
- `Vector{LongDNA{4}}` : All sequences contained in the FASTA file.

# Notes
- Only the sequence content is returned; FASTA headers are ignored.
- Sequences are loaded fully into memory.

# Example
```julia
seqs = load_sequences_fasta("data/sample.fa")
println(length(seqs))   # number of sequences
println(typeof(seqs[1]))  # LongDNA{4}
"""
function load_sequences_fasta(path::AbstractString)
    FASTA.Reader(open(path)) do reader
        [sequence(LongDNA{4}, record) for record in reader]
    end
end

####################################################################################################

"""
    load_balanced_data(params::SampleParams) -> (Vector{LongDNA{4}}, Vector{Int})

Load, filter, and balance DNA sequences from modern and ancient FASTA files.

This function prepares a dataset for binary classification by:
1. Loading sequences from the FASTA files specified in `SampleParams`.
2. Filtering sequences to a fixed length (`params.seqlen`).
3. Balancing the dataset by downsampling each class to the same size.
4. Returning the concatenated sequences and corresponding labels.

# Arguments
- `params::SampleParams`  
  Configuration struct specifying:
  - `modern` : path to modern DNA FASTA file  
  - `ancient`: path to ancient DNA FASTA file  
  - `seqlen` : required sequence length  
  - `seed`   : random seed for reproducible sampling  

# Returns
- `all_seqs::Vector{LongDNA{4}}`  
  Balanced set of DNA sequences (modern + ancient).
- `labels::Vector{Int}`  
  Integer labels aligned with `all_seqs`:  
  - `0` = modern (French)  
  - `1` = ancient (Neandertal)

# Notes
- Balancing is performed by sampling without replacement from the smaller class size.
- Printed messages report counts before and after filtering/balancing.

# Example
```julia
sparams = SampleParams(seqlen=50, seed=123,
                       modern="data/French.fa",
                       ancient="data/Neandertal.fa")

seqs, labels = load_balanced_data(sparams)
println(size(seqs)), println(size(labels))
"""
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

"""
    make_dataset(sparams::SampleParams, hparams::CNNParams) 
        -> (Vector{Tuple{Tuple{Array,Array},Tuple{Array,Array}}}, NamedTuple)

Prepare one‑hot encoded datasets and train/validation splits for CNN training.

This function ties together sequence loading, encoding, and splitting:
1. Loads and balances modern/ancient DNA sequences via [`load_balanced_data`](@ref).
2. Converts sequences into one‑hot encoded tensors with [`onehot_batch`](@ref).
3. Converts labels into one‑hot vectors with `Flux.onehotbatch`.
4. Splits the dataset into training and validation sets according to
   `hparams.k` (k‑fold cross‑validation) or `hparams.train_frac` (vanilla split).
5. Packages each split into `(Xtrain, Ytrain), (Xval, Yval)` tuples.

# Arguments
- `sparams::SampleParams`  
  Provides sequence length, random seed, and FASTA file paths.

- `hparams::CNNParams`  
  Provides hyperparameters controlling split strategy (`k` or `train_frac`).

# Returns
- `datasets::Vector`  
  A vector of folds. Each element is a tuple:
  - `(Xtrain, Ytrain)` : training data and labels
  - `(Xval,   Yval)`   : validation data and labels  
  Shapes:  
  - `X` = `(L, 4, Bsplit)` where `L` is sequence length, `Bsplit` is batch size  
  - `Y` = `(2, Bsplit)` one‑hot labels

- `meta::NamedTuple`  
  Metadata with fields:  
  - `B` : total number of sequences  
  - `L` : sequence length after encoding

# Notes
- If `hparams.k == 0`, a single dataset split is returned using `train_frac`.  
- If `hparams.k > 0`, `k` folds are returned for cross‑validation.  
- An assertion ensures that the number of encoded sequences matches the number of labels.

# Example
```julia
sparams = SampleParams(seqlen=50, seed=123)
hparams = CNNParams(train_frac=0.8, k=0)

datasets, meta = make_dataset(sparams, hparams)
(Xtrain, Ytrain), (Xval, Yval) = datasets[1]

@info "Dataset metadata" meta
"""
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

"""
    split_indices(B::Int, hparams::CNNParams, sparams::SampleParams)
        -> Vector{NamedTuple{(:train,:val),Tuple{Vector{Int},Vector{Int}}}}

Generate train/validation index splits for dataset partitioning.

This function supports both vanilla validation (single split) and
k‑fold cross‑validation, controlled by `hparams.k`:

- If `hparams.k == 0`:  
  Perform a single train/validation split using `hparams.train_frac`.

- If `hparams.k > 0`:  
  Perform k‑fold cross‑validation. The dataset is partitioned into
  `k` folds of size `ceil(B/k)`. Each fold is used once as validation,
  with the remaining data as training.

# Arguments
- `B::Int`  
  Total number of samples in the dataset.

- `hparams::CNNParams`  
  Provides split strategy:
  - `k` : number of folds (`0` = vanilla validation).  
  - `train_frac` : fraction of data used for training (only when `k=0`).

- `sparams::SampleParams`  
  Provides `seed` for reproducible shuffling.

# Returns
- `Vector{NamedTuple}`  
  Each element is a named tuple `(train=Vector{Int}, val=Vector{Int})`
  containing indices for training and validation sets.

# Notes
- Shuffling is seeded with `sparams.seed` for reproducibility.
- In k‑fold mode, the last fold may be slightly smaller if `B` is not
  divisible by `k`.

# Example
```julia
sparams = SampleParams(seed=123)
hparams = CNNParams(train_frac=0.8, k=0)

folds = split_indices(100, hparams, sparams)
(train_idx, val_idx) = folds[1]
"""
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
