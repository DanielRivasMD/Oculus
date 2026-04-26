####################################################################################################

module USCore

####################################################################################################

using BioSequences
using DataFrames
using DelimitedFiles
using Random

####################################################################################################

export onehot_encode,
  readdf,
  load_data,
  load_fasta,
  split_data,
  writedf,
  write_predictions,
  confusion_matrix,
  accuracy,
  balancedAccuracy,
  FDR,
  FNR,
  FOR,
  FPR,
  fScore,
  MCC,
  NPV,
  PPV,
  sensitivity,
  specificity,
  performance

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
    readdf(path; sep=',') -> DataFrame

Read CSV file onto DataFrame
"""
function readdf(path::String; sep::Char = ',')
  data, header = readdlm(path, sep, header = true)
  return DataFrame(data, vec(header))
end

####################################################################################################

"""
    load_data(path::String; label_col="label") -> DataFrame

Read CSV file onto DataFrame
Assumes the label column is named "label"
"""
function load_data(path::String; label_col = "label")::DataFrame
  df = readdf(path)
  if !(label_col in names(df))
    error("DataFrame does not contain column '$label_col'")
  end
  return df
end

####################################################################################################

"""
    load_fasta(path::String) -> Vector{String}

Read FASTA file onto vector of sequences
Headers are discarded
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
    split_data(df::DataFrame, split_frac::Float64, seed::Int) -> (train_df, test_df)

Stratified split into train and test sets based on the label column
If split_frac <= 0, returns the full DataFrame as train and an empty test
"""
function split_data(df::DataFrame, split_frac::Float64, seed::Int)
  Random.seed!(seed)
  labels = df.label
  if split_frac <= 0.0
    return df, DataFrame()
  end
  # Stratified split per class
  train_idx = Int[]
  test_idx = Int[]
  for class in unique(labels)
    idx = findall(labels .== class)
    n_test = Int(round(length(idx) * split_frac))
    shuffle!(idx)
    append!(test_idx, idx[1:n_test])
    append!(train_idx, idx[n_test+1:end])
  end
  return df[train_idx, :], df[test_idx, :]
end

####################################################################################################

"write dataframe"
function writedf(path, df::DataFrame; sep = ',')
  header = permutedims(names(df))  # 1×N matrix of strings
  data = Matrix(df)
  writedlm(path, vcat(header, data), sep)
end

####################################################################################################

"""
    write_predictions(path::String, predictions::Vector{Int}, test_indices::Vector{Int}, truth::Vector{Int})

Write predictions to CSV with columns: sample, truth, prediction.
"""
function write_predictions(
  path::String,
  predictions::Vector{Int},
  test_indices::Vector{Int},
  truth::Vector{Int},
)
  df = DataFrame(sample = test_indices, truth = truth, prediction = predictions)
  writedf(path, df)
end

####################################################################################################

"""
    confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int}) -> Matrix{Int}

Build a 2×2 confusion matrix from vectors of 0/1 labels.
Rows    = actual    (0 = negative, 1 = positive)
Columns = predicted (0 = negative, 1 = positive)
"""
function confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int})
  cm = zeros(Int, 2, 2)
  for (t, p) in zip(y_true, y_pred)
    cm[t+1, p+1] += 1
  end
  return cm
end

####################################################################################################

function accuracy(cm::Matrix)
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return (cm[2, 2] + cm[1, 1]) / sum(cm)
end

function sensitivity(cm::Matrix)  # True Positive Rate, Recall for positive class (1)
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[2, 2] / (cm[2, 2] + cm[2, 1])
end

function specificity(cm::Matrix)  # True Negative Rate
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[1, 1] / (cm[1, 1] + cm[1, 2])
end

function PPV(cm::Matrix)          # Positive Predictive Value, Precision
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[2, 2] / (cm[2, 2] + cm[1, 2])
end

function NPV(cm::Matrix)          # Negative Predictive Value
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[1, 1] / (cm[1, 1] + cm[2, 1])
end

function FPR(cm::Matrix)          # False Positive Rate, Fall‑out
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[1, 2] / (cm[1, 2] + cm[1, 1])
end

function FNR(cm::Matrix)          # False Negative Rate, Miss Rate
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[2, 1] / (cm[2, 1] + cm[2, 2])
end

function FDR(cm::Matrix)          # False Discovery Rate
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[1, 2] / (cm[1, 2] + cm[2, 2])
end

function FOR(cm::Matrix)          # False Omission Rate
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return cm[2, 1] / (cm[2, 1] + cm[1, 1])
end

function fScore(cm::Matrix)       # F1 score
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  p = PPV(cm)
  s = sensitivity(cm)
  return 2 * p * s / (p + s)
end

function MCC(cm::Matrix)          # Matthews Correlation Coefficient
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  tp, fp = cm[2, 2], cm[1, 2]
  fn, tn = cm[2, 1], cm[1, 1]
  num = tp * tn - fp * fn
  den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  return num / den
end

function balancedAccuracy(cm::Matrix)
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return (sensitivity(cm) + specificity(cm)) / 2
end

####################################################################################################

"""
    performance(cm::Matrix{Int})

Return a dictionary of all performance metrics for a given 2×2 confusion matrix
"""
function performance(cm::Matrix{Int})
  size(cm) == (2, 2) || error("Confusion matrix must be 2×2")
  return Dict(
    "Sensitivity" => sensitivity(cm),
    "Specificity" => specificity(cm),
    "Accuracy" => accuracy(cm),
    "BalancedAccuracy" => balancedAccuracy(cm),
    "F1Score" => fScore(cm),
    "Precision" => PPV(cm),
    "NPV" => NPV(cm),
    "FPR" => FPR(cm),
    "FNR" => FNR(cm),
    "FDR" => FDR(cm),
    "FOR" => FOR(cm),
    "MCC" => MCC(cm),
    "ConfusionMatrix" => cm,
  )
end

"""
    performance(y_true::Vector{Int}, y_pred::Vector{Int})

Calculate all metrics directly from label vectors (0 = negative, 1 = positive)
"""
function performance(y_true::Vector{Int}, y_pred::Vector{Int})
  cm = confusion_matrix(y_true, y_pred)
  return performance(cm)
end

####################################################################################################

end

####################################################################################################
