####################################################################################################

# load sequences

using FASTX, BioSequences

# Load sequences from a FASTA file
dna_seqs = FASTAReader(open("chrI.fa")) do reader
    [sequence(LongDNA{4}, record) for record in reader]
end

println("Loaded $(length(dna_seqs)) sequences")
println(typeof(dna_seqs[1]))  # LongDNA{4}

####################################################################################################

# one-hot encode sequences

# A,C,G,T -> columns 1..4. Ambiguity (e.g., N) leaves a zero row.
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

# Batch: pad to a common length for batching (simple right-padding with zeros)
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

####################################################################################################

using Flux: DataLoader

X = onehot_batch(dna_seqs)   # (maxL, 4, batch)
loader = DataLoader((X, Y); batchsize=64, shuffle=true)

using Flux: onehotbatch

Y = onehotbatch(labels, 1:2)  # shape (2, batch)

####################################################################################################

# vanilla cnn

using Flux

# Input shape (sequence_length, 4, batch)
# Use @autosize to infer Dense input sizes automatically
model = Flux.@autosize (100, 4, 1) Chain(
    # Block 1
    Conv((5,), 4 => 32, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(32),
    relu,
    Conv((5,), 32 => 32, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(32),
    relu,
    MaxPool((2,)),
    Dropout(0.2),

    # Block 2
    Conv((5,), 32 => 64, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(64),
    relu,
    Conv((5,), 64 => 64, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(64),
    relu,
    MaxPool((2,)),
    Dropout(0.3),

    # Block 3
    Conv((5,), 64 => 128, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(128),
    relu,
    Conv((5,), 128 => 128, pad=SamePad(), init=Flux.kaiming_uniform),
    BatchNorm(128),
    relu,
    MaxPool((2,)),
    Dropout(0.4),

    # Dense head
    Flux.flatten,
    Dense(_ => 128, init=Flux.kaiming_uniform),
    BatchNorm(128),
    relu,
    Dropout(0.5),
    Dense(_ => 2),
    softmax
)

####################################################################################################

# optimize & training

# Hyperparameters
epochs = 50
lrate  = 0.01

# Optimizer: SGD with momentum 0.9 (no decay, no Nesterov)
opt = OptimiserChain(Descent(lrate), Momentum(0.9))

# Binary cross-entropy on softmax logits (2 classes)
loss(x, y) = Flux.logitcrossentropy(model(x), y)

# Example training loop (expects a DataLoader providing (x, y) with y as 2×batch one-hot)
using Flux: DataLoader
for epoch in 1:epochs
    for (xb, yb) in loader
        gs = gradient(Flux.params(model)) do
            loss(xb, yb)
        end
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
end

####################################################################################################
