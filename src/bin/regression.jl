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
args = regression_args()

####################################################################################################
# Imports
####################################################################################################

using DataFrames
using FilePathsBase: basename, splitext
using GLM
using GLMNet
using LinearAlgebra
using StatsModels
using Random

####################################################################################################
# Load configuration
####################################################################################################

include(joinpath(Paths.UTIL, "params.jl"))
include(joinpath(Paths.CONFIG, "rparams.jl"))
include(joinpath(Paths.UTIL, "cross_validation.jl"))
include(joinpath(Paths.UTIL, "ioDataFrame.jl"))

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing

  infile = args["in"]
  outfile = args["out"]
  reg_method = args["reg"]
  nfolds = args["nfolds"]
  alpha = args["alpha"]
  split_frac = args["split"]

  println("Loading dataframe from $infile")
  df = readdf(infile; sep = ',')

  # Extract features and label
  label_col = "label"
  feature_cols = setdiff(names(df), [label_col])

  X = Matrix(df[:, feature_cols])
  y = Vector(df[:, label_col])

  # Validate label values are 0/1
  if !(all(x -> x in (0, 1), y))
    error(
      "Label column must contain only 0 (ancient) or 1 (modern) for logistic regression",
    )
  end

  ####################################################################################################
  # Train/Test Split
  ####################################################################################################

  if split_frac > 0
    println("Performing stratified split with test fraction = $split_frac")

    test_idx = perClassSplits(y, split_frac)
    train_idx = setdiff(collect(1:length(y)), test_idx)

    X_train = X[train_idx, :]
    y_train = y[train_idx]

    X_test = X[test_idx, :]
    y_test = y[test_idx]

  else
    println("No split requested. Training and predicting on full dataset.")

    train_idx = collect(1:length(y))
    test_idx = train_idx

    X_train = X
    y_train = y

    X_test = X
    y_test = y
  end

  ####################################################################################################
  # Logistic baseline (GLM)
  ####################################################################################################

  println("\nFitting logistic regression (binomial, logit link) baseline...")

  df_train = df[train_idx, :]
  f = Term(:label) ~ sum(Term.(Symbol.(feature_cols)))
  model_glm = glm(f, df_train, Binomial(), LogitLink())

  # GLM predicted probabilities
  probs_glm = GLM.predict(model_glm, df[test_idx, :])
  preds_glm_class = Int.(probs_glm .>= 0.5)

  ####################################################################################################
  # Regularized Models (Ridge, LASSO, Elastic Net) using GLMNet
  ####################################################################################################

  if reg_method == "none"
    println("\nNo regularization selected. Using GLM baseline only.")

    if outfile !== nothing
      outdf =
        DataFrame(sample = test_idx, truth = Int.(y_test), prediction = preds_glm_class)
      writedf(outfile, outdf; sep = ',')
      println("Predictions written to $outfile")
    end

    return
  end

  println("\nRunning regularized model: $reg_method (GLMNet logistic)")

  # Determine alpha
  if reg_method == "lasso"
    alpha = 1.0
  elseif reg_method == "ridge"
    alpha = 0.0
  elseif reg_method == "elasticnet"
    if alpha <= 0 || alpha >= 1
      error("Elastic Net requires 0 < --alpha < 1")
    end
  else
    error("Unknown regularization method: $reg_method")
  end

  println("Using alpha = $alpha")
  println("Cross-validation folds = $nfolds")

  # GLMNet logistic regression (binomial inferred automatically)
  y_train_int = Int.(y_train)
  fit_cv = glmnetcv(X_train, y_train_int; alpha = alpha, nfolds = nfolds)

  idx = argmin(fit_cv.meanloss)
  best_lambda = fit_cv.lambda[idx]

  println("Best λ from cross-validation: $best_lambda")

  # Extract coefficients
  intercept = fit_cv.path.a0[idx]
  betas = fit_cv.path.betas[:, idx]

  # Logistic probabilities on test set
  linpred = intercept .+ X_test * betas
  probs_reg = 1 ./ (1 .+ exp.(-linpred))
  preds_reg_class = Int.(probs_reg .>= 0.5)

  ####################################################################################################
  # Predictions output (standardized format)
  ####################################################################################################

  if outfile !== nothing
    preds = preds_reg_class  # regularized model output

    outdf = DataFrame(sample = test_idx, truth = Int.(y_test), prediction = preds)

    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

####################################################################################################
