####################################################################################################

module RGCore

####################################################################################################

using DataFrames
using DelimitedFiles
using GLM
using GLMNet
using Random
using StatsBase
using LinearAlgebra

####################################################################################################

export readdf,
  load_data,
  split_data,
  train_logistic,
  train_glmnet,
  predict_glmnet,
  evaluate,
  write_predictions

####################################################################################################

"""
    readdf(path; sep=',') -> DataFrame

Read a CSV file with a header and return a DataFrame.
"""
function readdf(path::String; sep::Char = ',')
  data, header = readdlm(path, sep, header = true)
  return DataFrame(data, vec(header))
end

####################################################################################################

"""
    load_data(path::String; label_col="label") -> DataFrame

Read a CSV file and return the DataFrame. Assumes the label column is named "label".
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
    split_data(df::DataFrame, split_frac::Float64, seed::Int) -> (train_df, test_df)

Stratified split into train and test sets based on the label column.
If split_frac <= 0, returns the full DataFrame as train and an empty test.
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

"""
    train_logistic(df_train::DataFrame) -> GLM.LogitModel

Fit a simple logistic regression model using all columns except "label".
"""
function train_logistic(df_train::DataFrame)
  f = Term(:label) ~ sum(Term.(Symbol.(setdiff(names(df_train), ["label"]))))
  return glm(f, df_train, Binomial(), LogitLink())
end

####################################################################################################

"""
    train_glmnet(df_train::DataFrame, reg::String, alpha::Float64, nfolds::Int, seed::Int) -> (intercept, betas, best_lambda)

Fit a regularized logistic model using GLMNet. Returns intercept, coefficient vector, and the best lambda.
- reg: "ridge" (alpha=0), "lasso" (alpha=1), "elasticnet" (0<alpha<1)
- alpha: mixing parameter (ignored for ridge/lasso)
- nfolds: number of cross‑validation folds to choose lambda
- seed: random seed
"""
function train_glmnet(
  df_train::DataFrame,
  reg::String,
  alpha::Float64,
  nfolds::Int,
  seed::Int,
)
  Random.seed!(seed)
  # Prepare feature matrix X and label vector y (0/1)
  feature_cols = setdiff(names(df_train), ["label"])
  X = Matrix(df_train[:, feature_cols])
  y = Int.(df_train.label)  # should already be 0/1
  # Determine alpha based on reg
  if reg == "ridge"
    alpha = 0.0
  elseif reg == "lasso"
    alpha = 1.0
  elseif reg == "elasticnet"
    if alpha <= 0 || alpha >= 1
      error("Elastic Net requires 0 < alpha < 1")
    end
  else
    error("Unknown reg method: $reg")
  end
  # Perform cross‑validated fit
  fit_cv = glmnetcv(X, y; alpha = alpha, nfolds = nfolds)
  idx = argmin(fit_cv.meanloss)
  best_lambda = fit_cv.lambda[idx]
  intercept = fit_cv.path.a0[idx]
  betas = fit_cv.path.betas[:, idx]
  return intercept, betas, best_lambda
end

####################################################################################################

"""
    predict_glmnet(intercept::Float64, betas::Vector{Float64}, X_test::Matrix) -> Vector{Float64}

Return predicted probabilities (0..1) for test set.
"""
function predict_glmnet(intercept, betas, X_test)
  linpred = intercept .+ X_test * betas
  return 1 ./ (1 .+ exp.(-linpred))
end

####################################################################################################

"""
    evaluate(y_true::Vector{Int}, y_pred::Vector{Int}) -> Dict

Compute basic classification metrics: accuracy, sensitivity, specificity, etc.
Expects 0/1 labels.
"""
function evaluate(y_true, y_pred)
  cm = zeros(Int, 2, 2)
  for (t, p) in zip(y_true, y_pred)
    cm[t+1, p+1] += 1   # because labels 0/1 -> indices 1/2
  end
  # Compute metrics
  tp, fp = cm[1, 1], cm[1, 2]
  fn, tn = cm[2, 1], cm[2, 2]
  accuracy = (tp + tn) / sum(cm)
  sensitivity = tp / (tp + fn)  # recall for positive class (modern = 1)
  specificity = tn / (tn + fp)  # true negative rate
  precision = tp / (tp + fp)
  f1 = 2 * precision * sensitivity / (precision + sensitivity)
  return Dict(
    "accuracy" => accuracy,
    "sensitivity" => sensitivity,
    "specificity" => specificity,
    "precision" => precision,
    "f1" => f1,
    "confusion_matrix" => cm,
  )
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
  # Write DataFrame using writedlm (CSV format)
  header = permutedims(names(df))
  data = Matrix(df)
  writedlm(path, vcat(header, data), ',')
end

####################################################################################################

end

####################################################################################################
