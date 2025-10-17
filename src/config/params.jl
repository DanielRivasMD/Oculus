####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    CNNParams(; kwargs...)

Hyperparameter container for configuring CNN training and evaluation

This struct centralizes all tunable parameters for reproducible experiments
It is constructed with `@with_kw`, so any field can be overridden at creation
time via keyword arguments

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

- `dropout1::Float64 = 0.2`  
  Dropout probability after convolutional block 1.

- `dropout2::Float64 = 0.3`  
  Dropout probability after convolutional block 2.

- `dropout3::Float64 = 0.4`  
  Dropout probability after convolutional block 3.

- `dropout_dense::Float64 = 0.5`  
  Dropout probability after the dense head.

- `layerout1::Int = 32`  
  Number of output channels in convolutional block 1.

- `layerout2::Int = 64`  
  Number of output channels in convolutional block 2.

- `layerout3::Int = 128`  
  Number of output channels in convolutional block 3.

# Usage
```julia
hparams = CNNParams(epochs=100, batchsize=128, k=5)
"""
@with_kw mutable struct CNNParams
    batchsize::Int           = 64           # dataloader batch size
    shuffle::Bool            = true         # dataloader shuffle
    device::Function         = gpu          # gpu or cpu
    kernelsize::Int          = 5            # convulution kernel size
    epochs::Int              = 100          # number of epochs
    train_frac::Float64      = 0.8          # data fraction for training
    k::Int                   = 0            # 0 = vanilla validation, > 0 = k-fold CV

    σ::Function              = relu         # activation function
    maxpool::Int             = 2            # max pooling
    η::Float64               = 1e-2         # learning rate
    momentum::Float64        = 0.9          # optimizer momentum
    dropout1::Float64        = 0.2          # block 1 dropout
    dropout2::Float64        = 0.3          # block 2 dropout
    dropout3::Float64        = 0.4          # block 3 dropout
    dropout_dense::Float64   = 0.5          # head dropout
    layerout1::Int           = 32           # neuron number out block 1
    layerout2::Int           = 64           # neuron number out block 2
    layerout3::Int           = 128          # neuron number out block 3
end

####################################################################################################
