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

export
  train_logistic,
  train_glmnet,
  predict_glmnet,
  predict_logistic


####################################################################################################

"""
    train_logistic(df_train::DataFrame) -> GLM.LogitModel

Fit a simple logistic regression model using all columns except "label"
"""
function train_logistic(df_train::DataFrame)
  f = Term(:label) ~ sum(Term.(Symbol.(setdiff(names(df_train), ["label"]))))
  return glm(f, df_train, Binomial(), LogitLink())
end

####################################################################################################

"""
    train_glmnet(df_train::DataFrame, reg::String, alpha::Float64, nfolds::Int, seed::Int) -> (intercept, betas, best_lambda)

Fit a regularized logistic model using GLMNet. Returns intercept, coefficient vector, and the best lambda
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
    predict_glmnet(intercept::Float64, betas::Vector{Float64}, X_test::Matrix; threshold::Float64 = 0.5) -> Vector{Float64}

Return predicted probabilities (0..1) for test set
"""
function predict_glmnet(intercept, betas, X_test; threshold = 0.5)
  linpred = intercept .+ X_test * betas
  probs = 1 ./ (1 .+ exp.(-linpred))
  preds = Int.(probs .>= threshold)
  return (probs, preds)
end

####################################################################################################

"""
    predict_logistic(model::GLM.LogitModel, test_df::DataFrame; threshold::Float64 = 0.5) -> (probs::Vector{Float64}, preds::Vector{Int})

Return predicted probabilities and binary predictions (0/1) for logistic regression model
"""
function predict_logistic(model, test_df::DataFrame; threshold = 0.5)
  probs = GLM.predict(model, test_df)
  preds = Int.(probs .>= threshold)
  return probs, preds
end

####################################################################################################

end

####################################################################################################
