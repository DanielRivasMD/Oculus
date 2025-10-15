####################################################################################################

"""
    pool_out_len(seqlen::Int, p::Int, n::Int) -> Int

Compute the output sequence length after applying `n` successive
`MaxPool((p,))` layers (stride = p, no padding).

Flux’s 1D `MaxPool` reduces the length as `floor(L / p)` per layer.

# Arguments
- `seqlen::Int` : Initial sequence length.
- `p::Int`      : Pooling window size (and stride).
- `n::Int`      : Number of pooling layers.

# Returns
- Final sequence length after `n` pooling operations.

# Example
```julia
pool_out_len(50, 2, 3)  # -> 6
"""
pool_out_len(seqlen::Int, p::Int, n::Int) = begin
    L = seqlen
    for _ in 1:n
        L = fld(L, p)  # floor division
    end
    L
end

"""
    buildCNN(args::CNNParams, sample::SampleParams) -> Chain

Construct a 1D convolutional neural network (CNN) for DNA sequence
classification, parameterized by `CNNParams` and `SampleParams`.

# Architecture
The network consists of three convolutional blocks followed by a dense
classification head:

- **Block 1**
  - Conv1D: 4 input channels (A,C,G,T one‑hot) → `args.layerout1` filters
  - BatchNorm, activation (`args.σ`)
  - Conv1D: `args.layerout1` → `args.layerout1`
  - BatchNorm, activation
  - MaxPool with window size `args.maxpool`
  - Dropout with probability `args.dropout1`

- **Block 2**
  - Conv1D: `args.layerout1` → `args.layerout2`
  - BatchNorm, activation
  - Conv1D: `args.layerout2` → `args.layerout2`
  - BatchNorm, activation
  - MaxPool, Dropout with `args.dropout2`

- **Block 3**
  - Conv1D: `args.layerout2` → `args.layerout3`
  - BatchNorm, activation
  - Conv1D: `args.layerout3` → `args.layerout3`
  - BatchNorm, activation
  - MaxPool, Dropout with `args.dropout3`

- **Dense head**
  - Flatten pooled features
  - Dense: `fin` → `args.layerout3`
  - BatchNorm, activation, Dropout with `args.dropout_dense`
  - Dense: `args.layerout3` → 2 output units
  - Softmax for class probabilities

# Arguments
- `args::CNNParams`  
  Hyperparameters controlling kernel size, dropout rates, layer widths,
  activation, pooling, device, etc.

- `sample::SampleParams`  
  Provides sequence length (`seqlen`), used to compute the flattened
  feature dimension after pooling.

# Returns
- `Chain` : A Flux model ready for training. Outputs a `(2, batchsize)`
  probability matrix for binary classification.

# Notes
- The final layer applies `softmax`, so outputs are probabilities.
- For discrete labels, apply `argmax(model(x), dims=1) .- 1`.
- If you prefer logits instead of probabilities, remove the `softmax`
  layer and use `logitcrossentropy` during training.

# Example
```julia
hparams = CNNParams()
sparams = SampleParams()
model = buildCNN(hparams, sparams)
ŷ = model(X)                  # probabilities
labels = argmax(ŷ, dims=1) .- 1  # hard labels
"""
function buildCNN(args::CNNParams, sample::SampleParams)
    # After 3 pooling layers of size args.maxpool
    Lout = pool_out_len(sample.seqlen, args.maxpool, 3)
    fin  = Lout * args.layerout3  # flattened features into Dense

    Chain(
        # Block 1
        Conv((args.kernelsize,), 4 => args.layerout1, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(args.layerout1), args.σ,
        Conv((args.kernelsize,), args.layerout1 => args.layerout1, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(args.layerout1), args.σ,
        MaxPool((args.maxpool,)), Dropout(args.dropout1),

        # Block 2
        Conv((args.kernelsize,), args.layerout1 => args.layerout2, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(args.layerout2), args.σ,
        Conv((args.kernelsize,), args.layerout2 => args.layerout2, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(args.layerout2), args.σ,
        MaxPool((args.maxpool,)), Dropout(args.dropout2),

        # Block 3
        Conv((args.kernelsize,), args.layerout2 => args.layerout3, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(args.layerout3), args.σ,
        Conv((args.kernelsize,), args.layerout3 => args.layerout3, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(args.layerout3), args.σ,
        MaxPool((args.maxpool,)), Dropout(args.dropout3),

        # Dense head
        Flux.flatten,
        Dense(fin, args.layerout3, init=Flux.kaiming_uniform),  # hidden = layerout3 (matches your earlier head)
        BatchNorm(args.layerout3), args.σ, Dropout(args.dropout_dense),
        Dense(args.layerout3, 2),
        softmax
    ) |> hparams.device
end

####################################################################################################
