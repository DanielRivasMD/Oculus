# TODO: fixed output naming
# TODO: pass seq len as arg

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
using FilePathsBase: basename, splitext, joinpath, dirname

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
# Helper functions
####################################################################################################

"""
    fix_length(seq::LongDNA{4}, L::Int) -> LongDNA{4}

Truncate or pad a DNA sequence to exactly `L` nucleotides.
"""
function fix_length(seq::LongDNA{4}, L::Int)
    len = length(seq)
    if len > L
        return seq[1:L]
    elseif len < L
        padded_str = String(seq) * repeat("N", L - len)
        return LongDNA{4}(padded_str)
    else
        return seq
    end
end

"""
    detect_seq_length(seqs::Vector{LongDNA{4}}) -> Int

Return the length of the first sequence in the dataset.
"""
function detect_seq_length(seqs::Vector{LongDNA{4}})
    return length(seqs[1])
end

"""
    parse_model_seq_length(modelname::String) -> Int

Extract sequence length from model filename, e.g. "..._75nt..." → 75.
"""
function parse_model_seq_length(modelname::String)
    m = match(r"_(\d+)nt", modelname)
    m === nothing && error("Could not parse sequence length from model name: $modelname")
    return parse(Int, m.captures[1])
end

"""
    predict_one(model, seq::LongDNA{4}, L::Int) -> (pred, probs)

Run inference on a single sequence of length L.
"""
function predict_one(model, seq::LongDNA{4}, L::Int)
    s = fix_length(seq, L)
    X = onehot_encode(s)            # (L, 4)
    X = reshape(X, L, 4, 1)         # (length, channels, batch)
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

# Detect input length and model length
input_len = detect_seq_length(seqs)
model_len = parse_model_seq_length(modelname)

println("Input sequence length = $input_len, Model expects = $model_len")

if input_len != model_len
    error("Sequence length mismatch: input=$input_len, model=$model_len. Adjust fix_length or retrain.")
end

# Predict all
preds = Vector{Int}(undef, length(seqs))
probs = Matrix{Float32}(undef, 2, length(seqs))

for (i, seq) in enumerate(seqs)
    pred, prob = predict_one(model, seq, model_len)
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
