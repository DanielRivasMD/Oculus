####################################################################################################

function buildCNN(hparams::CNNParams, sparams::SampleParams)
    Flux.@autosize (sparams.seqlen, length(nt2ix), 1) Chain(
        # Block 1
        Conv((hparams.kernelsize,), length(nt2ix) => hparams.layerout1, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout1), hparams.σ,
        Conv((hparams.kernelsize,), hparams.layerout1 => hparams.layerout1, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout1), hparams.σ,
        MaxPool((hparams.maxpool,)), Dropout(hparams.dropout1),

        # Block 2
        Conv((hparams.kernelsize,), hparams.layerout1 => hparams.layerout2, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout2), hparams.σ,
        Conv((hparams.kernelsize,), hparams.layerout2 => hparams.layerout2, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout2), hparams.σ,
        MaxPool((hparams.maxpool,)), Dropout(hparams.dropout2),

        # Block 3
        Conv((hparams.kernelsize,), hparams.layerout2 => hparams.layerout3, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout3), hparams.σ,
        Conv((hparams.kernelsize,), hparams.layerout3 => hparams.layerout3, pad=SamePad(), init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout3), hparams.σ,
        MaxPool((hparams.maxpool,)), Dropout(hparams.dropout3),

        # Dense head
        Flux.flatten,
        Dense(_ => hparams.layerout3, init=Flux.kaiming_uniform),
        BatchNorm(hparams.layerout3), hparams.σ, Dropout(hparams.dropout_dense),
        Dense(_ => 2),
        softmax
    )
    # ) |> hparams.device
end

####################################################################################################
