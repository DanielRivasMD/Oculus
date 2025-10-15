####################################################################################################

using CUDA
using Flux
using Flux              # DOC: observe that flux must be loaded twice to circumvent the dependency data race
using Flux: DataLoader

####################################################################################################

# load config
begin
  include(joinpath("..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.CONFIG, "sample.jl"))
  include(joinpath(Paths.CONFIG, "params.jl"))
  include(joinpath(Paths.UTIL, "architect.jl"))
  include(joinpath(Paths.UTIL, "load.jl"))
  include(joinpath(Paths.UTIL, "train.jl"))
end;

####################################################################################################

hparams = CNNParams(k = 5)        # train_frac = 0.75, k = 0
sparams = SampleParams()     # seed = 1

datasets, meta = make_dataset(sparams, hparams)
@info "dataset meta" meta

for (i, ((Xtrain, Ytrain), (Xval, Yval))) in enumerate(datasets)
    train_loader = DataLoader((Xtrain, Ytrain);
        batchsize=min(hparams.batchsize, size(Xtrain,3)),
        shuffle=hparams.shuffle)

    val_loader = DataLoader((Xval, Yval);
        batchsize=min(hparams.batchsize, size(Xval,3)),
        shuffle=false)

    model = buildCNN(hparams, sparams)
    result = trainCNN!(model, train_loader, val_loader; hparams=hparams)
    @info "Fold $i finished" val_loss=last(result.val_losses)
end

####################################################################################################
