####################################################################################################

using Flux
using Flux: crossentropy, DataLoader

####################################################################################################

"""
    trainCNN!(model, train_loader, val_loader; hparams::CNNParams)
        -> NamedTuple

Train a convolutional neural network (CNN) for a fixed number of epochs,
logging mean training and validation losses per epoch.

# Workflow
1. Defines a cross‑entropy loss function.
2. Initializes an optimizer chain (`Descent` + `Momentum`) with learning
   rate `hparams.η` and momentum `hparams.momentum`.
3. Iterates over `hparams.epochs`:
   - For each batch in `train_loader`:
     - Moves data to the specified device (`hparams.device`).
     - Computes gradients and updates model parameters.
     - Accumulates batch losses to compute mean training loss.
   - If `val_loader` is provided:
     - Evaluates the model on validation data.
     - Computes mean validation loss.
   - Logs per‑epoch training (and validation) losses.

# Arguments
- `model` :: `Chain`  
  Flux model to be trained.

- `train_loader::DataLoader`  
  DataLoader providing training batches `(xb, yb)`.

- `val_loader::Union{DataLoader,Nothing}`  
  DataLoader for validation batches, or `nothing` to skip validation.

- `hparams::CNNParams`  
  Hyperparameters controlling epochs, learning rate, momentum, and device.

# Returns
- `NamedTuple` with fields:
  - `model` : the trained model (mutated in place).
  - `train_losses::Vector{Float64}` : mean training loss per epoch.
  - `val_losses::Vector{Float64}`   : mean validation loss per epoch
    (empty if `val_loader === nothing`).

# Notes
- Training and validation losses are logged with `@info` each epoch.
- Losses are averaged across batches for stability.
- The model is updated in place; the returned `model` is the same object.

# Example
```julia
result = trainCNN!(model, train_loader, val_loader; hparams=hparams)

plot(result.train_losses, label="train")
plot!(result.val_losses, label="val")
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
