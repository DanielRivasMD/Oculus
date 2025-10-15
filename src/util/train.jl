####################################################################################################

using Flux
using Flux: crossentropy, DataLoader

####################################################################################################

"""
    trainCNN!(model, train_loader, val_loader; hparams::CNNParams)

Logs per-epoch train loss (mean over batches) and validation loss.
Returns a NamedTuple with losses and the final model.
"""
function trainCNN!(model, train_loader::DataLoader, val_loader::Union{DataLoader,Nothing};
                   hparams::CNNParams)

    loss(ŷ, y) = crossentropy(ŷ, y)
    opt = OptimiserChain(Descent(hparams.η), Momentum(hparams.momentum))
    st  = Flux.setup(opt, model)

    train_losses = Float64[]
    val_losses   = Float64[]

    for epoch in 1:hparams.epochs
        # Train epoch
        total = 0.0
        count = 0
        for (xb, yb) in train_loader
            xb, yb = hparams.device(xb), hparams.device(yb)
            gs, = gradient(model) do m
                ŷ = m(xb)
                l = loss(ŷ, yb)
                total += l
                count += 1
                l
            end
            Flux.update!(st, model, gs)
        end
        train_mean = total / max(count, 1)
        push!(train_losses, train_mean)

        # Validation epoch
        if val_loader !== nothing
            vtotal = 0.0
            vcount = 0
            for (xb, yb) in val_loader
                xb, yb = hparams.device(xb), hparams.device(yb)
                ŷ = model(xb)
                l = loss(ŷ, yb)
                vtotal += l
                vcount += 1
            end
            val_mean = vtotal / max(vcount, 1)
            push!(val_losses, val_mean)
            @info "epoch=$(epoch) train_loss=$(train_mean) val_loss=$(val_mean)"
        else
            @info "epoch=$(epoch) train_loss=$(train_mean)"
        end
    end

    return (model=model, train_losses=train_losses, val_losses=val_losses)
end

####################################################################################################
