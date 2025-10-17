####################################################################################################
# Imports
####################################################################################################

using CUDA                     # GPU acceleration

redirect_stderr(devnull) do
    @eval using Flux                     # Core deep learning library
    @eval using Flux                     # DOC: loaded twice to avoid dependency data race issues
end

using Flux: DataLoader         # Mini-batch data iterator

using BSON: @save
using Dates

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
  include(joinpath(Paths.CONFIG, "args.jl"))      # Args API
  include(joinpath(Paths.UTIL, "architect.jl"))   # Model architecture (buildCNN)
  include(joinpath(Paths.UTIL, "load.jl"))        # Data loading and preprocessing
  include(joinpath(Paths.UTIL, "train.jl"))       # Training loop (trainCNN!)
end;

####################################################################################################
# Experiment setup
####################################################################################################

# Parse CLI arguments
args = cnn_args()

hparams = load_cnnparams(args["cnn"])
sparams = load_sampleparams(args["sample"])

@info hparams
@info sparams

# Build dataset(s) according to split strategy (k-fold or vanilla)
datasets, meta = make_dataset(sparams, hparams)
@info "dataset meta" meta   # Log dataset metadata (total size, sequence length)

####################################################################################################
# Training loop
####################################################################################################

# Iterate over folds (1 if vanilla validation, k if cross-validation)
for (i, ((Xtrain, Ytrain), (Xval, Yval))) in enumerate(datasets)

    # Construct training DataLoader
    train_loader = DataLoader((Xtrain, Ytrain); batchsize=hparams.batchsize, shuffle=hparams.shuffle)

    # Construct validation DataLoader
    val_loader = DataLoader((Xval, Yval); batchsize=hparams.batchsize, shuffle=false)

    # Build a fresh CNN model for this fold
    model = buildCNN(hparams, sparams)

    # Train model and collect metrics
    result = trainCNN!(model, train_loader, val_loader; hparams=hparams)

    # Log final metrics for this fold
    final_val_loss = last(result.val_losses)
    final_val_acc  = last(result.val_accs)
    @info "Fold $i finished" val_loss=final_val_loss val_acc=final_val_acc

    # Save model checkpoints
    stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    model_cpu = Flux.fmap(cpu, model)

    train_losses = result.train_losses
    val_losses   = result.val_losses
    train_accs   = result.train_accs
    val_accs     = result.val_accs

    # BSON
    bson_path = joinpath(Paths.MODEL, "fold$(i)_$(stamp).bson")
    @save bson_path model_cpu hparams sparams final_val_loss final_val_acc train_losses val_losses train_accs val_accs
end

####################################################################################################
