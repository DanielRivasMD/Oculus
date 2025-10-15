####################################################################################################

using Parameters: @with_kw

####################################################################################################

@with_kw mutable struct CNNParams
    batchsize::Int           = 64           # dataloader batch size
    shuffle::Bool            = true         # dataloader shuffle

    device::Function         = gpu          # gpu or cpu
    kernelsize::Int          = 5            # convulution kernel size
    epochs::Int              = 50           # number of epochs
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
