####################################################################################################

using CUDA
using Flux
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

hparams   = CNNParams()
sparams = SampleParams()

all_seqs, labels = load_balanced_data(sparams)
X = onehot_batch(all_seqs)
Y = onehotbatch(labels, 0:1)

@show size(X), size(Y)
loader = DataLoader((X, Y); batchsize = hparams.batchsize, shuffle = hparams.shuffle)

model = buildCNN(hparams, sparams)
trainCNN!(model, loader; hparams = hparams)

####################################################################################################
