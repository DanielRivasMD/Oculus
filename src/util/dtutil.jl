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
  train_decision_tree,
  train_random_forest,
  train_xgboost,
  predict

####################################################################################################

"""
    train_decision_tree(X_train::Matrix, y_train::Vector{Int}, max_depth::Int, min_samples_leaf::Int) -> DecisionTreeClassifier

Train a Decision Tree classifier
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

Train a Random Forest classifier
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

Train an XGBoost classifier
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

Predict labels (0/1) for the test set
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

end

####################################################################################################
