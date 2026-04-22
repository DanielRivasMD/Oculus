####################################################################################################

module DTCore

####################################################################################################

using DataFrames
using DelimitedFiles
using Random
using DecisionTree
using XGBoost
using StatsBase
using LinearAlgebra

####################################################################################################

export load_data,
  split_data,
  train_decision_tree,
  train_random_forest,
  train_xgboost,
  predict,
  evaluate,
  write_predictions

####################################################################################################

"""
    load_data(path::String; label_col="label") -> DataFrame

Read a CSV file and return the DataFrame. Assumes the label column is named "label".
"""
function load_data(path::String; label_col = "label")::DataFrame
  data, header = readdlm(path, ',', header = true)
  df = DataFrame(data, vec(header))
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
    train_decision_tree(X_train::Matrix, y_train::Vector{Int}, max_depth::Int, min_samples_leaf::Int) -> DecisionTreeClassifier

Train a Decision Tree classifier.
"""
function train_decision_tree(
  X_train::Matrix,
  y_train::Vector{Int},
  max_depth::Int,
  min_samples_leaf::Int,
)
  # DecisionTree expects labels 1,2
  y_train_dt = y_train .+ 1
  model = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
  DecisionTree.fit!(model, X_train, y_train_dt)   # qualified call
  return model
end

####################################################################################################

"""
    train_random_forest(X_train::Matrix, y_train::Vector{Int}, max_depth::Int, min_samples_leaf::Int, n_trees::Int, partial_sampling::Float64) -> RandomForestClassifier

Train a Random Forest classifier.
"""
function train_random_forest(
  X_train::Matrix,
  y_train::Vector{Int},
  max_depth::Int,
  min_samples_leaf::Int,
  n_trees::Int,
  partial_sampling::Float64,
)
  y_train_dt = y_train .+ 1
  model = RandomForestClassifier(
    n_trees = n_trees,
    max_depth = max_depth,
    min_samples_leaf = min_samples_leaf,
    partial_sampling = partial_sampling,
  )
  DecisionTree.fit!(model, X_train, y_train_dt)   # qualified call
  return model
end

####################################################################################################

"""
    train_xgboost(X_train::Matrix, y_train::Vector{Int}, num_rounds::Int, eta::Float64, max_depth::Int, subsample::Float64, colsample_bytree::Float64, seed::Int) -> XGBoost.Booster

Train an XGBoost classifier.
"""
function train_xgboost(
  X_train::Matrix,
  y_train::Vector{Int},
  num_rounds::Int,
  eta::Float64,
  max_depth::Int,
  subsample::Float64,
  colsample_bytree::Float64,
  seed::Int,
)
  dtrain = DMatrix(X_train, label = Float32.(y_train))
  # DOC: hardcoded values on this call
  return XGBoost.xgboost(
    dtrain;
    num_round = num_rounds,
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    eval_metric = "logloss",
    seed = seed,
    watchlist = Dict(),
  )
end

####################################################################################################

"""
    predict(model, X_test::Matrix, model_type::String) -> Vector{Int}

Predict labels (0/1) for the test set.
"""
function predict(model, X_test::Matrix, model_type::String)
  if model_type == "tree" || model_type == "forest"
    # DecisionTree models predict 1,2 -> convert to 0,1
    pred_dt = DecisionTree.predict(model, X_test)
    return pred_dt .- 1
  elseif model_type == "xgboost"
    dtest = DMatrix(X_test)
    probs = XGBoost.predict(model, dtest)
    return Int.(probs .>= 0.5)
  else
    error("Unknown model_type: $model_type")
  end
end

####################################################################################################

"""
    evaluate(y_true::Vector{Int}, y_pred::Vector{Int}) -> Dict

Compute basic classification metrics: accuracy, sensitivity, specificity, precision, f1, confusion matrix.
Expects 0/1 labels.
"""
function evaluate(y_true, y_pred)
  cm = zeros(Int, 2, 2)
  for (t, p) in zip(y_true, y_pred)
    cm[t+1, p+1] += 1   # because labels 0/1 -> indices 1/2
  end
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
