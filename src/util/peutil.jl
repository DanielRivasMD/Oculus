####################################################################################################

module PECore

####################################################################################################

using DataFrames
using DelimitedFiles
using FreqTables

####################################################################################################

export accuracy,
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

"""
    accuracy(a::Matrix)
Calculate accuracy from a 2×2 confusion matrix.
"""
function accuracy(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return (a[1, 1] + a[2, 2]) / sum(a)
end

####################################################################################################

"""
    balancedAccuracy(a::Matrix)
Calculate balanced accuracy from a 2×2 confusion matrix.
"""
function balancedAccuracy(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return (sensitivity(a) + specificity(a)) / 2
end

####################################################################################################

"""
    FDR(a::Matrix)
Calculate False Discovery Rate from a 2×2 confusion matrix.
"""
function FDR(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[1, 2] / (a[1, 1] + a[1, 2])
end

####################################################################################################

"""
    FNR(a::Matrix)
Calculate False Negative Rate from a 2×2 confusion matrix.
"""
function FNR(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[2, 1] / (a[1, 1] + a[2, 1])
end

####################################################################################################

"""
    FOR(a::Matrix)
Calculate False Omission Rate from a 2×2 confusion matrix.
"""
function FOR(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[2, 1] / (a[2, 1] + a[2, 2])
end

####################################################################################################

"""
    FPR(a::Matrix)
Calculate False Positive Rate from a 2×2 confusion matrix.
"""
function FPR(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[1, 2] / (a[1, 2] + a[2, 2])
end

####################################################################################################

"""
    fScore(a::Matrix)
Calculate F1 score from a 2×2 confusion matrix.
"""
function fScore(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return 2 * (PPV(a) * sensitivity(a)) / (PPV(a) + sensitivity(a))
end

####################################################################################################

"""
    MCC(a::Matrix)
Calculate Matthews Correlation Coefficient from a 2×2 confusion matrix.
"""
function MCC(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  num = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
  den = sqrt(
    (a[1, 1] + a[1, 2]) * (a[1, 1] + a[2, 1]) * (a[2, 2] + a[1, 2]) * (a[2, 2] + a[2, 1]),
  )
  return num / den
end

####################################################################################################

"""
    NPV(a::Matrix)
Calculate Negative Predictive Value from a 2×2 confusion matrix.
"""
function NPV(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[2, 2] / (a[2, 2] + a[2, 1])
end

####################################################################################################

"""
    PPV(a::Matrix)
Calculate Positive Predictive Value (precision) from a 2×2 confusion matrix.
"""
function PPV(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[1, 1] / (a[1, 1] + a[1, 2])
end

####################################################################################################

"""
    sensitivity(a::Matrix)
Calculate Sensitivity (True Positive Rate, Recall) from a 2×2 confusion matrix.
"""
function sensitivity(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[1, 1] / (a[1, 1] + a[2, 1])
end

####################################################################################################

"""
    specificity(a::Matrix)
Calculate Specificity (True Negative Rate) from a 2×2 confusion matrix.
"""
function specificity(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return a[2, 2] / (a[2, 2] + a[1, 2])
end

####################################################################################################

"""
    performance(tb::Vector, label::Vector)
Calculate performance metrics from prediction and true label vectors.
"""
function performance(tb::Vector, label::Vector)
  # Convert to 1/2 labels for FreqTables
  tb_copied = copy(tb)
  # labels = [1, 2]
  tb_copied[tb_copied.>1] .= 2
  # Build confusion matrix
  pos = freqtable(tb_copied[label.==1])
  neg = freqtable(tb_copied[label.==0])
  # Convert to 2×2 matrix
  cm = [pos[1] pos[2]; neg[1] neg[2]]
  return performance(cm)
end

####################################################################################################

"""
    performance(a::Matrix)
Calculate all metrics from a 2×2 confusion matrix and return a dictionary.
"""
function performance(a::Matrix)
  if size(a) != (2, 2)
    error("Confusion matrix must be 2×2")
  end
  return Dict(
    "Sensitivity" => sensitivity(a),
    "Specificity" => specificity(a),
    "Accuracy" => accuracy(a),
    "BalancedAccuracy" => balancedAccuracy(a),
    "F1Score" => fScore(a),
    "Precision" => PPV(a),
    "NPV" => NPV(a),
    "FPR" => FPR(a),
    "FNR" => FNR(a),
    "FDR" => FDR(a),
    "FOR" => FOR(a),
    "MCC" => MCC(a),
    "ConfusionMatrix" => a,
  )
end

####################################################################################################

function load_predictions(path::String)
  data, header = readdlm(path, ',', header = true)
  df = DataFrame(data, vec(header))
  # Ensure required columns exist
  for col in ["truth", "prediction"]
    if !(col in names(df))
      error("CSV must contain column '$col'")
    end
  end
  return df
end

####################################################################################################

end

####################################################################################################
