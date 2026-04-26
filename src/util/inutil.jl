####################################################################################################

module INCore

####################################################################################################

using BSON
using BioSequences, FASTX, CodecZlib
using Flux
using FilePathsBase: basename, splitext, joinpath, dirname
using DelimitedFiles

import BSON: load

####################################################################################################

export load_model, load_sequences, predict_all, write_predictions

####################################################################################################

"""
    parse_model_seq_length(modelname::String) -> Int

Extract sequence length from a filename containing `_NNNnt`, e.g.
`myModel_75nt_timestamp.bson` → 75.
"""
function parse_model_seq_length(modelname::String)
  m = match(r"_(\d+)nt", modelname)
  m === nothing && error("Could not parse sequence length from model name: $modelname")
  return parse(Int, m.captures[1])
end

####################################################################################################

const nt2ix = Dict(DNA_A => 1, DNA_C => 2, DNA_G => 3, DNA_T => 4)

function onehot_encode(seq::LongDNA{4})
  L = length(seq)
  X = zeros(Float32, L, 4)
  @inbounds for (i, nt) in enumerate(seq)
    ix = get(nt2ix, nt, 0)
    if ix != 0
      X[i, ix] = 1.0f0
    end
  end
  return X
end

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

####################################################################################################

"""
    load_model(path::String) -> (model, hparams, sparams, model_len)

Load a trained CNN from a BSON file. Returns:
- `model`         : Flux model on CPU
- `hparams`       : HyperParams struct (if saved)
- `sparams`       : SampleParams struct (if saved)
- `model_len`     : inferred sequence length from filename
"""
function load_model(path::String)
  bs = BSON.load(path)
  if haskey(bs, :model_cpu)
    model = bs[:model_cpu] |> cpu
  elseif haskey(bs, :model)
    model = bs[:model] |> cpu
  else
    error("BSON file does not contain model_cpu or model")
  end

  hparams = get(bs, :hparams, nothing)
  sparams = get(bs, :sparams, nothing)

  modelname = splitext(basename(path))[1]
  model_len = parse_model_seq_length(modelname)

  return model, hparams, sparams, model_len
end

####################################################################################################

"""
    load_sequences(path::String) -> Vector{LongDNA{4}}

Load sequences from FASTA / FASTQ (plain or .gz). Returns a vector of `LongDNA{4}`.
"""
function load_sequences(path::String)
  open(path) do io
    stream = endswith(path, ".gz") ? GzipDecompressorStream(io) : io
    if occursin(r"\.fa$|\.fasta$|\.fa\.gz$|\.fasta\.gz$", path)
      reader = FASTA.Reader(stream)
      return [sequence(LongDNA{4}, record) for record in reader]
    elseif occursin(r"\.fq$|\.fastq$|\.fq\.gz$|\.fastq\.gz$", path)
      reader = FASTQ.Reader(stream)
      return [sequence(LongDNA{4}, record) for record in reader]
    else
      error("Unsupported file extension: $path")
    end
  end
end

####################################################################################################

"""
    predict_all(model, seqs::Vector{LongDNA{4}}, L::Int)
        -> (preds::Vector{Int}, probs::Matrix{Float32})

Run inference on a batch of sequences after fixing their length to `L`.
Returns hard predictions (0/1) and a 2×N probability matrix (row1=p0, row2=p1).
"""
function predict_all(model, seqs::Vector{LongDNA{4}}, L::Int)
  preds = Vector{Int}(undef, length(seqs))
  probs = Matrix{Float32}(undef, 2, length(seqs))
  for (i, seq) in enumerate(seqs)
    s = fix_length(seq, L)
    X = onehot_encode(s)               # (L,4)
    X = reshape(X, L, 4, 1)            # (L, channels, batch)
    p = model(X)                       # (2,1)
    probs[:, i] = p[:, 1]
    preds[i] = p[2, 1] >= 0.5 ? 1 : 0   # 1 if p1 >= 0.5 else 0
  end
  return preds, probs
end

####################################################################################################

"""
    write_predictions(path::String, ids::Vector{String}, probs::Matrix{Float32})

Write CSV with columns: `id, p0, p1`.
"""
function write_predictions(path::String, ids::Vector{String}, probs::Matrix{Float32})
  open(path, "w") do io
    println(io, "id,p0,p1")
    for i in eachindex(ids)
      println(io, "$(ids[i]),$(probs[1,i]),$(probs[2,i])")
    end
  end
end

####################################################################################################

end

####################################################################################################
