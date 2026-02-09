####################################################################################################

using Flux: crossentropy, DataLoader
using Flux: onecold

using Optimisers
using MLUtils: DataLoader
using Zygote

####################################################################################################

function accuracy(model, data)
  X, Y = data
  yhat = model(X)
  mean(onecold(yhat) .== onecold(Y))
end

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
"""
function trainCNN!(
  model,
  train_loader::DataLoader,
  val_loader::Union{DataLoader,Nothing};
  hparams::CNNParams,
)

  # Optimiser setup
  opt = Optimisers.OptimiserChain(
    Optimisers.Descent(hparams.η),
    Optimisers.Momentum(hparams.momentum),
  )
  opt_state = Optimisers.setup(opt, model)

  # Loss function with optional L2 regularization
  function lossfn(m, xb, yb)
    ce = Flux.crossentropy(m(xb), yb)
    if !isnan(hparams.λ)
      reg = hparams.λ * Zygote.ignore() do
        sum(p -> sum(abs2, p), Flux.params(m))
      end
      return ce + reg
    else
      return ce
    end
  end

  # Metrics storage
  train_losses = Float32[]
  val_losses = Float32[]
  train_accs = Float32[]
  val_accs = Float32[]

  # Epoch loop
  for epoch = 1:hparams.epochs
    eloss = 0.0f0
    ecorr = 0
    ecnt = 0

    # Training loop
    for (xb, yb) in train_loader
      xb = hparams.device(xb)
      yb = hparams.device(yb)

      # Compute loss and gradient tree
      loss_val, grads = Flux.withgradient(m -> lossfn(m, xb, yb), model)

      # Update optimiser state and model
      opt_state, model = Optimisers.update!(opt_state, model, grads[1])

      # Metrics
      eloss += loss_val
      ŷ = model(xb)
      ecorr += sum(Flux.onecold(ŷ) .== Flux.onecold(yb))
      ecnt += size(yb, 2)
    end

    push!(train_losses, eloss / max(ecnt, 1))
    push!(train_accs, ecorr / max(ecnt, 1))

    # Validation loop (only if provided)
    if val_loader !== nothing
      vloss = 0.0f0
      vcorr = 0
      vcnt = 0
      for (xb, yb) in val_loader
        xb = hparams.device(xb)
        yb = hparams.device(yb)

        ŷ = model(xb)
        vloss += Flux.crossentropy(ŷ, yb)
        vcorr += sum(Flux.onecold(ŷ) .== Flux.onecold(yb))
        vcnt += size(yb, 2)
      end
      push!(val_losses, vloss / max(vcnt, 1))
      push!(val_accs, vcorr / max(vcnt, 1))
    else
      push!(val_losses, NaN32)
      push!(val_accs, NaN32)
    end

    @info "epoch $epoch" train_loss = train_losses[end] train_acc = train_accs[end] val_loss =
      val_losses[end] val_acc = val_accs[end]
  end

  return (; train_losses, val_losses, train_accs, val_accs, model, opt_state)
end

####################################################################################################
