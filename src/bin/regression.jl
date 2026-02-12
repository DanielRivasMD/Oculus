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
  # Classification: Logistic baseline + Regularized (ridge/lasso/elasticnet)
  ##################################################################################################


  # Validate label values are 0/1
  if !(all(x -> x in (0, 1), y))
    error(
      "Label column must contain only 0 (ancient) or 1 (modern) for logistic regression",
    )
  end

  ##################################################################################################
  # Logistic baseline (GLM)
  ##################################################################################################

  println("\nFitting logistic regression (binomial, logit link) baseline...")

  # Build formula and fit GLM logistic model
  f = Term(:label) ~ sum(Term.(Symbol.(feature_cols)))
  model_glm = glm(f, df, Binomial(), LogitLink())

  # Coefficients (intercept + features)
  println("\nLogistic model coefficients:")
  println(DataFrame(term = coefnames(model_glm), coef = GLM.coef(model_glm)))

  # Predicted probabilities from GLM (response scale)
  # GLM.predict returns fitted values on response scale for GLM.jl
  probs_glm = GLM.predict(model_glm)

  # Class predictions (threshold 0.5)
  preds_glm_class = Int.(probs_glm .>= 0.5)

  ##################################################################################################
  # Regularized Models (Ridge, LASSO, Elastic Net) using GLMNet (binomial)
  ##################################################################################################

  if reg_method == "none"
    println("\nNo regularization selected. Skipping GLMNet models.")

    if outfile !== nothing
      outdf = DataFrame(prob_glm = probs_glm, pred_glm = preds_glm_class)
      writedf(outfile, outdf; sep = ',')
      println("Predictions written to $outfile")
    end

    return
  end

  println("\nRunning regularized model: $reg_method (GLMNet binomial)")

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

  # GLMNet expects y as numeric 0/1 for binomial
  y_int = Int.(y)   # ensure 0/1 integers
  fit_cv = glmnetcv(X, y_int; alpha = alpha, nfolds = nfolds)

  idx = argmin(fit_cv.meanloss)
  best_lambda = fit_cv.lambda[idx]

  println("Best λ from cross-validation: $best_lambda")

  # Extract coefficients from GLMNet path
  intercept = fit_cv.path.a0[idx]
  betas = fit_cv.path.betas[:, idx]

  reg_terms = vcat(:Intercept, Symbol.(feature_cols))
  reg_coefs = vcat(intercept, betas)

  println("\nRegularized model coefficients (at best λ):")
  println(DataFrame(term = reg_terms, coef = reg_coefs))

  # Compute predicted probabilities from GLMNet (logistic)
  linpred = intercept .+ X * betas
  probs_reg = 1 ./ (1 .+ exp.(-linpred))
  preds_reg_class = Int.(probs_reg .>= 0.5)

  ##################################################################################################
  # Metrics
  ##################################################################################################

  function confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int})
    classes = sort(unique(vcat(y_true, y_pred)))
    n = length(classes)
    cm = zeros(Int, n, n)
    # map class value to index (should be 0/1 -> 1/2)
    idxmap = Dict(c => i for (i, c) in enumerate(classes))
    for (t, p) in zip(y_true, y_pred)
      cm[idxmap[t], idxmap[p]] += 1
    end
    return cm, classes
  end

  # True labels
  y_int = Int.(y)   # 0/1

  # GLM metrics
  cm_glm, classes = confusion_matrix(y_int, preds_glm_class)
  acc_glm = sum(cm_glm[i, i] for i = 1:size(cm_glm, 1)) / sum(cm_glm)

  # Regularized metrics
  cm_reg, _ = confusion_matrix(y_int, preds_reg_class)
  acc_reg = sum(cm_reg[i, i] for i = 1:size(cm_reg, 1)) / sum(cm_reg)

  println("\nLogistic baseline metrics")
  println("Accuracy: $(round(acc_glm, digits=4))")
  println("Confusion matrix (rows=true, cols=pred):")
  println(cm_glm)

  println("\nRegularized model metrics")
  println("Accuracy: $(round(acc_reg, digits=4))")
  println("Confusion matrix (rows=true, cols=pred):")
  println(cm_reg)

  ##################################################################################################
  # Predictions output
  ##################################################################################################

  if outfile !== nothing
    outdf = DataFrame(
      prob_glm = probs_glm,
      pred_glm = preds_glm_class,
      prob_reg = probs_reg,
      pred_reg = preds_reg_class,
    )
    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

##################################################################################################
