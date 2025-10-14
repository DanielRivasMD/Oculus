####################################################################################################

# Load utilities
using FASTX, BioSequences
using StatsBase
using Flux
using Flux: onehotbatch, DataLoader
using Flux: crossentropy

loss(ŷ, y) = crossentropy(ŷ, y)

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

# Utility: load sequences from a FASTA/FASTQ-like file, ignoring qualities
function load_sequences_fasta(path::AbstractString)
    FASTA.Reader(open(path)) do reader
        [sequence(LongDNA{4}, record) for record in reader]
    end
end

####################################################################################################

# Load datasets
french_seqs     = load_sequences_fasta("data/French_sample.fa")
neandertal_seqs = load_sequences_fasta("data/Neandertal_sample.fa")

println("French:     $(length(french_seqs)) reads")
println("Neandertal: $(length(neandertal_seqs)) reads")

####################################################################################################

# Filter to 50 nt and balance
french_50     = filter(seq -> length(seq) == 50, french_seqs)
neandertal_50 = filter(seq -> length(seq) == 50, neandertal_seqs)

println("French 50nt:     $(length(french_50))")
println("Neandertal 50nt: $(length(neandertal_50))")

# Balance by downsampling the larger set
minN = min(length(french_50), length(neandertal_50))
using Random
Random.seed!(42)  # for reproducibility

french_balanced     = sample(french_50, minN; replace=false)
neandertal_balanced = sample(neandertal_50, minN; replace=false)

println("Balanced French:     $(length(french_balanced))")
println("Balanced Neandertal: $(length(neandertal_balanced))")

# Concatenate sequences and labels
all_seqs = vcat(french_balanced, neandertal_balanced)
labels   = vcat(zeros(Int, length(french_balanced)),
                ones(Int,  length(neandertal_balanced)))

####################################################################################################

# One-hot encode and build DataLoader
X = onehot_batch(all_seqs)          # (50, 4, 2*minN)
Y = onehotbatch(labels, 0:1)        # (2, 2*minN)

@show size(X), size(Y)              # sanity check: (50,4,N), (2,N)
println("Sequences: ", length(all_seqs))
println("Labels:    ", length(labels))

# DataLoader: collate=true (default) returns (50,4,batch) and (2,batch)
loader = DataLoader((X, Y); batchsize=64, shuffle=true)

####################################################################################################

# Input shape is (50, 4, batch)
model = Flux.@autosize (50, 4, 1) Chain(
    # Block 1
    Conv((5,), 4 => 32, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(32), relu,
    Conv((5,), 32 => 32, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(32), relu,
    MaxPool((2,)), Dropout(0.2),

    # Block 2
    Conv((5,), 32 => 64, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(64), relu,
    Conv((5,), 64 => 64, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(64), relu,
    MaxPool((2,)), Dropout(0.3),

    # Block 3
    Conv((5,), 64 => 128, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(128), relu,
    Conv((5,), 128 => 128, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(128), relu,
    MaxPool((2,)), Dropout(0.4),

    # Dense head
    Flux.flatten,
    Dense(_ => 128, init=Flux.kaiming_uniform),
    BatchNorm(128), relu, Dropout(0.5),
    Dense(_ => 2),
    softmax
)

####################################################################################################

# Optimise & training (modern Flux API)
epochs = 50
lrate  = 0.01
opt    = OptimiserChain(Descent(lrate), Momentum(0.9))

# Setup optimiser state once
st = Flux.setup(opt, model)

for epoch in 1:epochs
    for (xb, yb) in loader
        # Compute gradient w.r.t. model
        gs, = gradient(model) do m
            loss(m(xb), yb)
        end
        # Update model with optimiser state
        Flux.update!(st, model, gs)
    end
    @info "Finished epoch $epoch"
end

####################################################################################################
