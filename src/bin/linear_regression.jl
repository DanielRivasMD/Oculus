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
using StatsModels

####################################################################################################
# Load configuration
####################################################################################################

begin
  include(joinpath(Paths.UTIL, "ioDataFrame.jl"))
end;

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing

  infile = args["in"]
  outfile = args["out"]
  reg_method = args["reg"]
  nfolds = args["nfolds"]
  alpha = args["alpha"]

  println("Loading dataframe from $infile")
  df = readdf(infile; sep = ',')

  # Extract features and label
  label_col = "label"
  feature_cols = setdiff(names(df), [label_col])

  X = Matrix(df[:, feature_cols])
  y = Vector(df[:, label_col])

  ##################################################################################################
  # Linear Regression (no regularization)
  ##################################################################################################

  println("\nFitting linear regression model...")

  f = Term(:label) ~ sum(Term.(Symbol.(feature_cols)))
  model_lm = lm(f, df)

  println("\nLinear model coefficients:")
  println(DataFrame(term = coefnames(model_lm), coef = GLM.coef(model_lm)))

  ##################################################################################################
  # Regularized Models (Ridge, LASSO, Elastic Net)
  ##################################################################################################

  if reg_method == "none"
    println("\nNo regularization selected. Skipping GLMNet models.")

    if outfile !== nothing
      preds_lm = GLM.predict(model_lm)
      outdf = DataFrame(pred_lm = preds_lm)
      writedf(outfile, outdf; sep = ',')
      println("Predictions written to $outfile")
    end

    return
  end

  println("\nRunning regularized model: $reg_method")

  # Determine alpha
  if reg_method == "lasso"
    alpha = 1.0
  elseif reg_method == "ridge"
    alpha = 0.0
  elseif reg_method == "elasticnet"
    # alpha is passed by user
    if alpha <= 0 || alpha >= 1
      error("Elastic Net requires 0 < --alpha < 1")
    end
  else
    error("Unknown regularization method: $reg_method")
  end

  println("Using alpha = $alpha")
  println("Cross-validation folds = $nfolds")

  fit_cv = glmnetcv(X, y; alpha = alpha, nfolds = nfolds)

  idx = argmin(fit_cv.meanloss)
  best_lambda = fit_cv.lambda[idx]

  println("Best λ from cross-validation: $best_lambda")

  # Extract coefficients from GLMNet path
  intercept = fit_cv.path.a0[idx]
  betas = fit_cv.path.betas[:, idx]

  reg_terms = vcat(:Intercept, Symbol.(feature_cols))
  reg_coefs = vcat(intercept, betas)

  println("\nRegularized model coefficients:")
  println(DataFrame(term = reg_terms, coef = reg_coefs))

  ##################################################################################################
  # Predictions
  ##################################################################################################

  if outfile !== nothing
    preds_lm = GLM.predict(model_lm)
    preds_reg = intercept .+ X * betas

    outdf = DataFrame(pred_lm = preds_lm, pred_reg = preds_reg)

    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

##################################################################################################
