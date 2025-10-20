####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    CNNParams(; kwargs...)

Hyperparameter container for configuring CNN training and evaluation.

# Fields
- `batchsize::Int = 64`
- `shuffle::Bool = true`
- `device::Function = gpu`
- `kernelsize::Int = 5`
- `epochs::Int = 100`
- `train_frac::Float64 = 0.8`
- `k::Int = 0`
- `σ::Function = relu`
- `maxpool::Int = 2`
- `η::Float64 = 1e-3`
- `momentum::Float64 = 0.9`
- `dropouts::Vector{Float64} = [0.2, 0.3, 0.4, 0.5]`
- `layerouts::Vector{Int} = [32, 64, 128]`
- `λ::Float64 = NaN`   # L2 regularization strength. If NaN, no regularization is applied.
- `batchnorm::Bool = true`  # Enable/disable BatchNorm layers
- `dropout::Bool = true`    # Enable/disable Dropout layers
"""
@with_kw mutable struct CNNParams
    batchsize::Int            = 64
    shuffle::Bool             = true
    device::Function          = gpu
    kernelsize::Int           = 5
    epochs::Int               = 100
    train_frac::Float64       = 0.8
    k::Int                    = 0

    σ::Function               = relu
    maxpool::Int              = 2
    η::Float64                = 1e-3
    momentum::Float64         = 0.9

    dropouts::Vector{Float64} = [0.2, 0.3, 0.4, 0.5]
    layerouts::Vector{Int}    = [32, 64, 128]

    λ::Float64                = NaN
    batchnorm::Bool           = true
    dropout::Bool             = true
end

####################################################################################################
