####################################################################################################
# Imports
####################################################################################################

using CUDA                     # GPU acceleration

redirect_stderr(devnull) do
    @eval using Flux           # Core deep learning library
    @eval using Flux           # DOC: loaded twice to avoid dependency data race issues
end
using Flux: onehotbatch, onecold

using BSON
using BioSequences
using FASTX
using CodecZlib
using ArgParse
using FilePathsBase: basename, splitext, joinpath

####################################################################################################
# Load configuration and utilities
####################################################################################################

using Parameters: @with_kw

# temp declaration for compatibility
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

begin
  # Load path definitions and ensure required directories exist
  include(joinpath("..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  # Load configuration structs and helper modules
  include(joinpath(Paths.CONFIG, "sample.jl"))    # SampleParams (data config)
  # include(joinpath(Paths.CONFIG, "params.jl"))    # CNNParams (hyperparameters)
  include(joinpath(Paths.CONFIG, "args.jl"))      # Args API
  include(joinpath(Paths.UTIL, "load.jl"))        # Data loading and preprocessing
end;

####################################################################################################
# Helper function
####################################################################################################

"""
    fix_length_37(seq::LongDNA{4}) -> LongDNA{4}

Truncate or pad a DNA sequence to exactly 37 nt.
- Longer sequences are truncated to the first 37 bases.
- Shorter sequences are right‑padded with N.
"""
function fix_length_37(seq::LongDNA{4})
    if length(seq) > 37
        return seq[1:37]  # slicing preserves LongDNA{4}
    elseif length(seq) < 37
        padded_str = String(seq) * repeat("N", 37 - length(seq))
        return LongDNA{4}(padded_str)
    else
        return seq
    end
end

"""
    predict_one(model, seq::LongDNA{4}) -> (pred, probs)

Run inference on a single sequence, forcing length 37.
- Returns the predicted class (0 = French, 1 = Neandertal)
  and the raw probability vector.
"""
function predict_one(model, seq::LongDNA{4})
    s = fix_length_37(seq)
    X = onehot_encode(s)            # (37, 4)
    X = reshape(X, 37, 4, 1)        # (length, channels, batch)
    probs = model(X)                # (2, 1)
    pred  = onecold(probs, 0:1)[1]  # decode to 0 or 1
    return pred, probs[:,1]
end

####################################################################################################
# Sequence loader
####################################################################################################

function load_sequences(path::String)
    open(path) do io
        stream = endswith(path, ".gz") ? GzipDecompressorStream(io) : io
        if endswith(path, ".fa") || endswith(path, ".fasta") || endswith(path, ".fa.gz") || endswith(path, ".fasta.gz")
            reader = FASTA.Reader(stream)
            return [LongDNA{4}(sequence(record)) for record in reader]
        elseif endswith(path, ".fq") || endswith(path, ".fastq") || endswith(path, ".fq.gz") || endswith(path, ".fastq.gz")
            reader = FASTQ.Reader(stream)
            return [LongDNA{4}(sequence(record)) for record in reader]
        else
            error("Unsupported file extension: $path")
        end
    end
end

####################################################################################################
# Inference CLI
####################################################################################################

using FilePathsBase: basename, splitext, dirname

args = infer_args()

# Load model
bs = BSON.load(args["model"])
model = bs[:model_cpu] |> cpu

# Parse model name
modelname = splitext(basename(args["model"]))[1]

# Parse sample name and rootdir from data path
datapath   = args["data"]
samplefile = basename(datapath)
samplebase = replace(samplefile, r"\.fastq.*" => "")
rootdir    = basename(dirname(datapath))

# Load sequences
seqs = load_sequences(datapath)

# Predict all
preds = Vector{Int}(undef, length(seqs))
probs = Matrix{Float32}(undef, 2, length(seqs))

for (i, seq) in enumerate(seqs)
    pred, prob = predict_one(model, seq)
    preds[i]   = pred
    probs[:,i] = prob
end

# Write CSV
open(args["out"], "w") do io
    println(io, "id,p0,p1")
    for i in 1:length(seqs)
        read_id = i
        id_str  = string(rootdir, "-", samplebase, "-", modelname, "-", read_id)
        println(io, "$id_str,$(probs[1,i]),$(probs[2,i])")
    end
end

println("Predictions written to $(args["out"])")

####################################################################################################
