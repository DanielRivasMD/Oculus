####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    CNNParams(; kwargs...)

Hyperparameter container for configuring CNN training and evaluation.

This struct centralizes all tunable parameters for reproducible experiments.
It is constructed with `@with_kw`, so any field can be overridden at creation
time via keyword arguments.

# Fields
- `batchsize::Int = 64`
  Number of samples per batch in the `DataLoader`.

- `shuffle::Bool = true`
  Whether to shuffle batches during training.

- `device::Function = gpu`
  Device mapping function (`gpu` or `cpu`) applied to data and model.

- `kernelsize::Int = 5`
  Size of the 1D convolution kernel.

- `epochs::Int = 100`
  Number of training epochs.

- `train_frac::Float64 = 0.8`
  Fraction of the dataset used for training when `k == 0` (vanilla validation).

- `k::Int = 0`
  Cross‑validation mode.
  - `0` → single train/validation split using `train_frac`.
  - `>0` → k‑fold cross‑validation with `k` folds.

- `σ::Function = relu`  
  Activation function applied after convolutions and dense layers.

- `maxpool::Int = 2`
  Pooling window size for `MaxPool`.

- `η::Float64 = 1e-2`
  Learning rate for the optimizer.

- `momentum::Float64 = 0.9`
  Momentum term for the optimizer.

- `dropouts::Vector{Float64} = [0.2, 0.3, 0.4, 0.5]`
  Dropout probabilities for each stage of the network.
  - Index n → convolutional block n
  - Last Index → dense head

- `layerouts::Vector{Int} = [32, 64, 128]`
  Number of output channels for each convolutional block.
  - Index n → block n filters

# Usage
```julia
# Override defaults at construction
hparams = CNNParams(epochs=100, batchsize=128, k=5,
                    dropouts=[0.1, 0.2, 0.3, 0.4],
                    layerouts=[16, 32, 64, 128])
```
"""
@with_kw mutable struct CNNParams
    batchsize::Int         = 64                            # dataloader batch size
    shuffle::Bool          = true                          # dataloader shuffle
    device::Function       = gpu                           # gpu or cpu
    kernelsize::Int        = 5                             # convolution kernel size
    epochs::Int            = 100                           # number of epochs
    train_frac::Float64    = 0.8                           # data fraction for training
    k::Int                 = 0                             # 0 = vanilla validation, >0 = k-fold CV

    σ::Function            = relu                          # activation function
    maxpool::Int           = 2                             # max pooling
    η::Float64             = 1e-3                          # learning rate
    momentum::Float64      = 0.9                           # optimizer momentum

    dropouts::Vector{Float64} = [0.2, 0.3, 0.4, 0.5]       # per-block + dense head
    layerouts::Vector{Int}    = [32, 64, 128]              # filters per conv block
end

####################################################################################################
