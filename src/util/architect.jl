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

####################################################################################################

function maybe_bn(channels, args)
    args.batchnorm ? BatchNorm(channels) : identity
end

function maybe_do(p, args)
    args.dropout ? Dropout(p) : identity
end

####################################################################################################

"""
    buildCNN(args::CNNParams, sample::SampleParams) -> Chain

Construct a 1D convolutional neural network whose depth is determined
by the length of `args.layerouts`.

This function generalizes variants:
- Each entry in `args.layerouts` defines the number of output channels
  for one convolutional block.
- Each entry in `args.dropouts[1:end-1]` defines the dropout probability
  applied after the corresponding block.
- The final entry `args.dropouts[end]` defines the dropout probability
  applied after the dense hidden layer.

# Architecture
For `n = length(args.layerouts)`:
- Input: one-hot encoded sequence with 4 channels.
- For each block `i = 1..n`:
  - Conv1D(kernel = args.kernelsize, in_channels, out_channels = layerouts[i])
  - BatchNorm(out_channels), activation = args.σ
  - Conv1D(kernel = args.kernelsize, out_channels => out_channels)
  - BatchNorm(out_channels), activation = args.σ
  - MaxPool(window = args.maxpool)
  - Dropout(dropouts[i])
- Dense head:
  - Flatten
  - Dense(fin, layerouts[end]) → BatchNorm → activation → Dropout(dropouts[end])
  - Dense(layerouts[end], 2)
  - softmax

Here `fin = pool_out_len(sample.seqlen, args.maxpool, n) * layerouts[end]`.

# Assertions
- `length(args.dropouts) == length(args.layerouts) + 1`
  (one dropout per block + one for the dense head).

# Usage
```julia
# 1-block CNN
hparams = CNNParams(layerouts=[32], dropouts=[0.2, 0.5])
model = buildCNN(hparams, sample)

# 3-block CNN
hparams = CNNParams(layerouts=[32, 64, 128], dropouts=[0.2, 0.3, 0.4, 0.5])
model = buildCNN(hparams, sample)
```
"""
function buildCNN(args::CNNParams, sample::SampleParams)
    nblocks = length(args.layerouts)
    @assert length(args.dropouts) == nblocks + 1

    # Compute flattened size after all pooling layers
    Lout = pool_out_len(sample.seqlen, args.maxpool, nblocks)
    fin  = Lout * args.layerouts[end]

    layers = Any[]

    # First block: input channels = 4
    push!(layers,
        Conv((args.kernelsize,), 4 => args.layerouts[1],
             pad=SamePad(), init=Flux.kaiming_uniform),
        maybe_bn(args.layerouts[1], args), args.σ,
        Conv((args.kernelsize,), args.layerouts[1] => args.layerouts[1],
             pad=SamePad(), init=Flux.kaiming_uniform),
        maybe_bn(args.layerouts[1], args), args.σ,
        MaxPool((args.maxpool,)), maybe_do(args.dropouts[1], args)
    )

    # Remaining blocks
    for i in 2:nblocks
        push!(layers,
            Conv((args.kernelsize,), args.layerouts[i-1] => args.layerouts[i],
                 pad=SamePad(), init=Flux.kaiming_uniform),
            maybe_bn(args.layerouts[i], args), args.σ,
            Conv((args.kernelsize,), args.layerouts[i] => args.layerouts[i],
                 pad=SamePad(), init=Flux.kaiming_uniform),
            maybe_bn(args.layerouts[i], args), args.σ,
            MaxPool((args.maxpool,)), maybe_do(args.dropouts[i], args)
        )
    end

    # Dense head
    push!(layers,
        Flux.flatten,
        Dense(fin, args.layerouts[end], init=Flux.kaiming_uniform),
        maybe_bn(args.layerouts[end], args), args.σ,
        maybe_do(args.dropouts[end], args),
        Dense(args.layerouts[end], 2),
        softmax
    )

    return Chain(layers...) |> args.device
end

####################################################################################################
