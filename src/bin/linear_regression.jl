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
using DelimitedFiles
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

  println("Loading dataframe from $infile")
  df = readdf(infile; sep = ',')

  ################################################################################################
  # FEATURE / LABEL EXTRACTION
  ################################################################################################

  label_col = "label"
  feature_cols = setdiff(names(df), [label_col])

  ################################################################################################
  # LINEAR REGRESSION (GLM)
  ################################################################################################

  println("\nFitting linear regression model...")

  feature_syms = Symbol.(feature_cols)
  f = Term(:label) ~ sum(Term.(feature_syms))
  model_lm = lm(f, df)

  println("\nLinear model coefficients:")
  println(DataFrame(term = coefnames(model_lm), coef = GLM.coef(model_lm)))

  ################################################################################################
  # LASSO REGRESSION WITH CROSS‑VALIDATION (GLMNet)
  ################################################################################################

  println("\nRunning LASSO cross‑validation (GLMNet)...")

  X = Matrix(df[:, feature_cols])
  y = Vector(df[:, label_col])

  # α = 1 → Lasso
  fit_cv = glmnetcv(X, y; alpha = 1.0, nfolds = 10)

  # Best λ index
  idx = argmin(fit_cv.meanloss)
  best_lambda = fit_cv.lambda[idx]

  println("Best λ from cross‑validation: $best_lambda")

  # Extract coefficients directly from the GLMNet path
  intercept = fit_cv.path.a0[idx]
  betas = fit_cv.path.betas[:, idx]

  lasso_terms = vcat(:Intercept, Symbol.(feature_cols))
  lasso_coefs = vcat(intercept, betas)

  println("\nLASSO coefficients:")
  println(DataFrame(term = lasso_terms, coef = lasso_coefs))

  ################################################################################################
  # OPTIONAL PREDICTIONS
  ################################################################################################

  if outfile !== nothing
    preds_lm = GLM.predict(model_lm)
    preds_lasso = intercept .+ X * betas

    outdf = DataFrame(pred_lm = preds_lm, pred_lasso = preds_lasso)

    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

####################################################################################################
