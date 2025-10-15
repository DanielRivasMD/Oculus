####################################################################################################

# Compute the output length after n MaxPool((p,)) layers (stride=p, no padding)
# Flux MaxPool reduces length as floor(L / p) per layer.
pool_out_len(seqlen::Int, p::Int, n::Int) = begin
    L = seqlen
    for _ in 1:n
        L = fld(L, p)  # floor division
    end
    L
end

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
