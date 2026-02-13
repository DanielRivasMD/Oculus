####################################################################################################
# cli args
####################################################################################################

begin
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.UTIL, "args.jl"))
end

# Parse CLI arguments
args = feature_args()

####################################################################################################
# Imports
####################################################################################################

using DataFrames
using DelimitedFiles
using FilePathsBase: basename, splitext
using BioSequences

####################################################################################################
# Load configuration
####################################################################################################

begin
  include(joinpath(Paths.CONFIG, "feparams.jl"))
  include(joinpath(Paths.UTIL, "ioDataFrame.jl"))
end;

####################################################################################################
# FASTA loader
####################################################################################################

feature_config = args["config"]
feature_params = loadFeatureParams(feature_config, args)

@info feature_params
@info args

function load_fasta(path::String)
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
# Feature extraction
####################################################################################################

function minimal_features(seq::String)
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

  return (
    ct5p_5 = ct5p(5),
    ct5p_10 = ct5p(10),
    ct5p_15 = ct5p(15),
    ga3p_5 = ga3p(5),
    ga3p_10 = ga3p(10),
    ga3p_15 = ga3p(15),
    gc_content = gc,
  )
end

####################################################################################################
# Feature extraction
####################################################################################################

"One-hot encode every position: a1,a2,... t1,t2,... g1,... c1,..."
function onehot_features(seq::String)
  seq_up = uppercase(seq)
  L = length(seq_up)

  bases = ['A', 'T', 'G', 'C']
  feats = Dict{Symbol,Int}()

  for (i, b) in enumerate(seq_up)
    for base in bases
      key = Symbol(lowercase(base), i)   # e.g. "a1", "t7"
      feats[key] = (b == base ? 1 : 0)
    end
  end

  return feats
end

####################################################################################################
# Build DataFrame for a set of sequences
####################################################################################################

function build_df(seqs::Vector{String}, label::Int; onehot::Bool)
  rows = Vector{Dict}()

  for seq in seqs
    if onehot
      feats = onehot_features(seq)
    else
      feats = Dict(pairs(minimal_features(seq)))
    end

    feats[:label] = label
    push!(rows, feats)
  end

  df = DataFrame(rows)

  # Ensure label is last
  newcols = filter(!=(string(:label)), names(df))
  push!(newcols, string(:label))
  df = df[:, newcols]

  return df
end


####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing

  modern_seqs = load_fasta(feature_params.modern)
  ancient_seqs = load_fasta(feature_params.ancient)

  println("Loaded $(length(modern_seqs)) modern sequences")
  println("Loaded $(length(ancient_seqs)) ancient sequences")

  df_mod = build_df(modern_seqs, 1; onehot = feature_params.onehot)
  df_anc = build_df(ancient_seqs, 0; onehot = feature_params.onehot)

  df = vcat(df_mod, df_anc)

  writedf(feature_params.out, df; sep = ',')
  println("Feature matrix written to $feature_params.out")
end

####################################################################################################
