####################################################################################################

module FECore

####################################################################################################

using DataFrames
using DelimitedFiles
using SHA

####################################################################################################

export load_fasta, minimal_features, onehot_features, build_df, writedf, file_hash

####################################################################################################

"""
    load_fasta(path::String) -> Vector{String}

Read a FASTA file and return a vector of sequences (as plain strings).
Headers are discarded.
"""
function load_fasta(path::String)::Vector{String}
  seqs = String[]
  buf = IOBuffer()
  open(path) do f
    for line in eachline(f)
      if startswith(line, '>')
        if position(buf) > 0
          push!(seqs, String(take!(buf)))
        end
      else
        write(buf, strip(line))
      end
    end
    if position(buf) > 0
      push!(seqs, String(take!(buf)))
    end
  end
  return seqs
end

####################################################################################################

"""
    minimal_features(seq::String) -> Dict{Symbol,Float64}

Compute the minimal set of features:
- C→T fraction at 5' end (windows of 5,10,15)
- G→A fraction at 3' end (windows of 5,10,15)
- GC content
All values rounded to two decimals.
"""
function minimal_features(seq::String)::Dict{Symbol,Float64}
  L = length(seq)
  seq_up = uppercase(seq)

  # 5' C→T
  function ct5p(k)
    window = seq_up[1:min(k, L)]
    c = count(==('C'), window)
    t = count(==('T'), window)
    frac = c == 0 ? 0.0 : t / (c + t)
    return round(frac; digits = 2)
  end

  # 3' G→A
  function ga3p(k)
    window = seq_up[max(1, L - k + 1):L]
    g = count(==('G'), window)
    a = count(==('A'), window)
    frac = g == 0 ? 0.0 : a / (g + a)
    return round(frac; digits = 2)
  end

  # GC content
  gc_raw = count(c -> c == 'G' || c == 'C', seq_up) / L
  gc = round(gc_raw; digits = 2)

  return Dict(
    :ct5p_5 => ct5p(5),
    :ct5p_10 => ct5p(10),
    :ct5p_15 => ct5p(15),
    :ga3p_5 => ga3p(5),
    :ga3p_10 => ga3p(10),
    :ga3p_15 => ga3p(15),
    :gc_content => gc,
  )
end

####################################################################################################

"""
    onehot_features(seq::String) -> Dict{Symbol,Int}

One‑hot encode each position: for each base A,T,G,C at position i,
create a feature `a_i`, `t_i`, `g_i`, `c_i` with value 1 if the base matches.
"""
function onehot_features(seq::String)::Dict{Symbol,Int}
  seq_up = uppercase(seq)
  L = length(seq_up)

  bases = ['A', 'T', 'G', 'C']
  feats = Dict{Symbol,Int}()

  for (i, b) in enumerate(seq_up)
    for base in bases
      key = Symbol(lowercase(base), i)   # e.g. a1, t7
      feats[key] = (b == base ? 1 : 0)
    end
  end

  return feats
end

####################################################################################################

"""
    build_df(seqs::Vector{String}, label::Int; onehot::Bool) -> DataFrame

Convert a list of sequences into a DataFrame where each row corresponds to one sequence,
and columns are the features (plus a `label` column). The label column is placed last.
"""
function build_df(seqs::Vector{String}, label::Int; onehot::Bool)::DataFrame
  rows = Vector{Dict}()
  for seq in seqs
    if onehot
      feats = onehot_features(seq)
    else
      feats = minimal_features(seq)
    end
    feats[:label] = label

    row = Dict{String,Any}(string(k) => v for (k, v) in feats)
    push!(rows, row)
  end

  df = DataFrame(rows)

  # Ensure label is last column
  cols = [name for name in names(df) if name != "label"]
  push!(cols, "label")
  df = df[:, cols]

  return df
end

####################################################################################################

"""
    writedf(path::String, df::DataFrame; sep::Char=',')

Write a DataFrame to a CSV file.
"""
function writedf(path::String, df::DataFrame; sep::Char = ',')
  open(path, "w") do io
    println(io, join(names(df), sep))
    for row in eachrow(df)
      println(io, join(row, sep))
    end
  end
end

####################################################################################################

"""
    file_hash(path::String) -> String

Compute the SHA‑256 hash of a file’s contents. Used for cache invalidation.
"""
function file_hash(path::String)::String
  open(path, "r") do io
    return bytes2hex(sha256(io))
  end
end

####################################################################################################

end

####################################################################################################
