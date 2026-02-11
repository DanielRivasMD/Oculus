# TODO: fixed output naming
# TODO: pass seq len as arg

####################################################################################################
# cli args
####################################################################################################

begin
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.CONFIG, "args.jl"))   # infer_args()
end

# Parse CLI arguments
args = inference_args()

####################################################################################################
# Imports
####################################################################################################

using CUDA

redirect_stderr(devnull) do
  @eval using Flux
  @eval using Flux
end

using Flux: onehotbatch, onecold
using BSON
using BioSequences
using FASTX
using CodecZlib
using FilePathsBase: basename, splitext, joinpath, dirname

####################################################################################################
# Load configuration
####################################################################################################

begin
  include(joinpath(Paths.CONFIG, "hparams.jl"))  # CNNParams
  include(joinpath(Paths.CONFIG, "sparams.jl"))  # SampleParams
  include(joinpath(Paths.UTIL, "load.jl"))       # onehot_encode, etc.
end

####################################################################################################
# Helper functions
####################################################################################################

"""
    fix_length(seq::LongDNA{4}, L::Int)

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
    detect_seq_length(seqs)

Return the length of the first sequence.
"""
detect_seq_length(seqs) = length(seqs[1])

"""
    parse_model_seq_length(modelname)

Extract sequence length from model filename, e.g. "..._75nt..." → 75.
"""
function parse_model_seq_length(modelname::String)
  m = match(r"_(\d+)nt", modelname)
  m === nothing && error("Could not parse sequence length from model name: $modelname")
  return parse(Int, m.captures[1])
end

"""
    predict_one(model, seq, L)

Run inference on a single sequence of length L.
"""
function predict_one(model, seq::LongDNA{4}, L::Int)
  s = fix_length(seq, L)
  X = onehot_encode(s)            # (L, 4)
  X = reshape(X, L, 4, 1)         # (length, channels, batch)
  probs = model(X)                # (2, 1)
  pred = onecold(probs, 0:1)[1]
  return pred, probs[:, 1]
end

####################################################################################################
# Sequence loader
####################################################################################################

function load_sequences(path::String)
  open(path) do io
    stream = endswith(path, ".gz") ? GzipDecompressorStream(io) : io

    if endswith(path, ".fa") ||
       endswith(path, ".fasta") ||
       endswith(path, ".fa.gz") ||
       endswith(path, ".fasta.gz")
      reader = FASTA.Reader(stream)
      return [LongDNA{4}(sequence(record)) for record in reader]

    elseif endswith(path, ".fq") ||
           endswith(path, ".fastq") ||
           endswith(path, ".fq.gz") ||
           endswith(path, ".fastq.gz")
      reader = FASTQ.Reader(stream)
      return [LongDNA{4}(sequence(record)) for record in reader]

    else
      error("Unsupported file extension: $path")
    end
  end
end

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing

  # Load model
  bs = BSON.load(args["model"])
  model = bs[:model_cpu] |> cpu

  # Parse model name
  modelname = splitext(basename(args["model"]))[1]

  # Parse input file
  datapath = args["data"]
  samplefile = basename(datapath)
  samplebase = replace(samplefile, r"\.fastq.*" => "")
  rootdir = basename(dirname(datapath))

  # Load sequences
  seqs = load_sequences(datapath)

  # Detect lengths
  input_len = detect_seq_length(seqs)
  model_len = parse_model_seq_length(modelname)

  println("Input sequence length = $input_len, Model expects = $model_len")

  if input_len != model_len
    error(
      "Sequence length mismatch: input=$input_len, model=$model_len. Adjust fix_length or retrain.",
    )
  end

  # Predict all
  preds = Vector{Int}(undef, length(seqs))
  probs = Matrix{Float32}(undef, 2, length(seqs))

  for (i, seq) in enumerate(seqs)
    pred, prob = predict_one(model, seq, model_len)
    preds[i] = pred
    probs[:, i] = prob
  end

  # Write CSV
  open(args["out"], "w") do io
    println(io, "id,p0,p1")
    for i = 1:length(seqs)
      id_str = string(rootdir, "-", samplebase, "-", modelname, "-", i)
      println(io, "$id_str,$(probs[1,i]),$(probs[2,i])")
    end
  end

  println("Predictions written to $(args["out"])")
end
####################################################################################################
