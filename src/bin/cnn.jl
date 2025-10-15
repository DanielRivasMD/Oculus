####################################################################################################
# Imports
####################################################################################################

using CUDA                     # GPU acceleration
using Flux                     # Core deep learning library
using Flux                     # DOC: loaded twice to avoid dependency data race issues
using Flux: DataLoader         # Mini-batch data iterator

####################################################################################################
# Load configuration and utilities
####################################################################################################

begin
  # Load path definitions and ensure required directories exist
  include(joinpath("..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  # Load configuration structs and helper modules
  include(joinpath(Paths.CONFIG, "sample.jl"))    # SampleParams (data config)
  include(joinpath(Paths.CONFIG, "params.jl"))    # CNNParams (hyperparameters)
  include(joinpath(Paths.UTIL, "architect.jl"))   # Model architecture (buildCNN, etc.)
  include(joinpath(Paths.UTIL, "load.jl"))        # Data loading and preprocessing
  include(joinpath(Paths.UTIL, "train.jl"))       # Training loop (trainCNN!)
end;

####################################################################################################
# Experiment setup
####################################################################################################

# Define hyperparameters:
# - k = 5 → perform 5-fold cross-validation
# - if k = 0, use vanilla validation with train_frac (default 0.75)
hparams = CNNParams(k = 5)

# Define sample parameters (sequence length, seed, FASTA paths)
sparams = SampleParams()

# Build dataset(s) according to split strategy (k-fold or vanilla)
datasets, meta = make_dataset(sparams, hparams)
@info "dataset meta" meta   # Log dataset metadata (total size, sequence length)

####################################################################################################
# Training loop
####################################################################################################

# Iterate over folds (1 if vanilla validation, k if cross-validation)
for (i, ((Xtrain, Ytrain), (Xval, Yval))) in enumerate(datasets)

    # Construct training DataLoader
    # - batchsize capped at dataset size to avoid warnings
    # - shuffle enabled for training
    train_loader = DataLoader((Xtrain, Ytrain);
        batchsize=min(hparams.batchsize, size(Xtrain,3)),
        shuffle=hparams.shuffle)

    # Construct validation DataLoader
    # - no shuffling for validation
    val_loader = DataLoader((Xval, Yval);
        batchsize=min(hparams.batchsize, size(Xval,3)),
        shuffle=false)

    # Build a fresh CNN model for this fold
    model = buildCNN(hparams, sparams)

    # Train model and collect losses
    result = trainCNN!(model, train_loader, val_loader; hparams=hparams)

    # Log final validation loss for this fold
    @info "Fold $i finished" val_loss=last(result.val_losses)
end

####################################################################################################
