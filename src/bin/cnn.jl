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

hparams = CNNParams()
sparams = SampleParams()

(Xtrain, Ytrain), (Xval, Yval), meta = make_dataset(sparams, hparams)
@info "dataset sizes" meta

train_loader = DataLoader((Xtrain, Ytrain);
    batchsize=hparams.batchsize, shuffle=hparams.shuffle)

val_loader = DataLoader((Xval, Yval);
    batchsize=hparams.batchsize, shuffle=false)

model = buildCNN(hparams, sparams)
result = trainCNN!(model, train_loader, val_loader; hparams=hparams)

####################################################################################################
