####################################################################################################
# Imports
####################################################################################################

# TODO: fixed output naming
# TODO: pass seq len as arg

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
# Load configuration
####################################################################################################

begin
  # Load path definitions
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  # Load configuration structs
  include(joinpath(Paths.CONFIG, "sample.jl"))    # SampleParams (data config)
  include(joinpath(Paths.CONFIG, "params.jl"))    # CNNParams (hyperparameters)
  include(joinpath(Paths.CONFIG, "args.jl"))      # Args API
  include(joinpath(Paths.UTIL, "load.jl"))        # Data loading and preprocessing
end;

####################################################################################################
# Helper function
####################################################################################################

"""
    fix_length(seq::LongDNA{4}, L::Int) -> LongDNA{4}

Truncate or pad a DNA sequence to exactly `L` nucleotides.

- If the sequence is longer than `L`, it is truncated to the first `L` bases.
- If the sequence is shorter than `L`, it is right‑padded with `N`.
- If the sequence is already length `L`, it is returned unchanged.

# Example
```julia
julia> using BioSequences

julia> seq = dna"ACGTACGT"
8nt DNA Sequence:
ACGTACGT

julia> fix_length(seq, 12)
12nt DNA Sequence:
ACGTACGTNNNN

julia> fix_length(seq, 4)
4nt DNA Sequence:
ACGT
"""
function fix_length(seq::LongDNA{4}, L::Int)
    len = length(seq)
    if len > L
        return seq[1:L]  # slicing preserves LongDNA{4}
    elseif len < L
        padded_str = String(seq) * repeat("N", L - len)
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
    s = fix_length(seq, 37)
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

# Parse file name
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
