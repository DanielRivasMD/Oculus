####################################################################################################
# cli args
####################################################################################################

begin
  # Load path definitions
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.UTIL, "args.jl"))
end

# Parse CLI arguments
args = cnn_args()

####################################################################################################
# Imports
####################################################################################################

using CUDA                   # GPU acceleration

redirect_stderr(devnull) do
  @eval using Flux           # Core deep learning library
  @eval using Flux           # DOC: loaded twice to avoid dependency data race issues
end

using Flux: DataLoader       # Mini-batch data iterator

using BSON: @save
using Dates
using FilePathsBase: basename, splitext

####################################################################################################
# Load configuration
####################################################################################################

begin
  include(joinpath(Paths.CONFIG, "hparams.jl"))  # CNNParams (hyperparameters)
  include(joinpath(Paths.CONFIG, "sparams.jl"))  # SampleParams (data config)
  include(joinpath(Paths.UTIL, "architect.jl"))  # Model architecture (buildCNN)
  include(joinpath(Paths.UTIL, "load.jl"))       # Data loading and preprocessing
  include(joinpath(Paths.UTIL, "train.jl"))      # Training loop (trainCNN!)
end;

####################################################################################################
# Experiment setup
####################################################################################################

# Paths from CLI (empty string means "use defaults from struct")
cnn_path = args["cnn"]
sample_path = args["sample"]

# Load hyperparameters and sample config
hparams = loadHparams(cnn_path)
sparams = loadSparams(sample_path)

@info hparams
@info sparams

# Extract clean names for logging and filenames
cnn_name = cnn_path != "" ? splitext(basename(cnn_path))[1] : "cnn_default"
sample_name = sample_path != "" ? splitext(basename(sample_path))[1] : "sample_default"

# Build dataset(s) according to split strategy (k-fold or vanilla)
datasets, meta = make_dataset(sparams, hparams)
@info "dataset meta" meta   # Log dataset metadata (total size, sequence length)

####################################################################################################
# Training loop
####################################################################################################

for (i, ((Xtrain, Ytrain), (Xval, Yval))) in enumerate(datasets)

  train_loader =
    DataLoader((Xtrain, Ytrain); batchsize = hparams.batchsize, shuffle = hparams.shuffle)
  val_loader = DataLoader((Xval, Yval); batchsize = hparams.batchsize, shuffle = false)

  model = buildCNN(hparams, sparams)
  result = trainCNN!(model, train_loader, val_loader; hparams = hparams)

  final_val_loss = last(result.val_losses)
  final_val_acc = last(result.val_accs)
  @info "Fold $i finished" cnn = cnn_name sample = sample_name val_loss = final_val_loss val_acc =
    final_val_acc

  stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
  model_cpu = Flux.fmap(cpu, model)

  train_losses = result.train_losses
  val_losses = result.val_losses
  train_accs = result.train_accs
  val_accs = result.val_accs

  # BSON filename: CNN_SAMPLE_fN_TIMESTAMP.bson
  fname = "$(cnn_name)_$(sample_name)_f$(i)_$(stamp).bson"
  bson_path = joinpath(Paths.MODEL, fname)

  @save bson_path model_cpu hparams sparams final_val_loss final_val_acc train_losses val_losses train_accs val_accs
end

####################################################################################################
