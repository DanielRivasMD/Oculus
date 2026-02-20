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
args = decisiontree_args()

####################################################################################################
# Imports
####################################################################################################

using ArgParse
using DataFrames
using Random
using DecisionTree
using LinearAlgebra
using XGBoost
using FilePathsBase: basename, splitext

####################################################################################################
# Load configuration
####################################################################################################

include(joinpath(Paths.UTIL, "params.jl"))
include(joinpath(Paths.CONFIG, "dtparams.jl"))
include(joinpath(Paths.UTIL, "ioDataFrame.jl"))

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing

  infile = args["in"]
  outfile = args["out"]
  max_depth = args["max_depth"]
  min_samples_leaf = args["min_samples_leaf"]
  test_frac = args["test_frac"]
  seed = args["seed"]

  # Random forest specific
  n_trees = args["n_trees"]
  rf_partial = args["rf_partial_sampling"]

  # XGBoost specific
  xgb_rounds = args["xgb_rounds"]
  xgb_eta = args["xgb_eta"]
  xgb_max_depth = args["xgb_max_depth"]
  xgb_subsample = args["xgb_subsample"]
  xgb_colsample_bytree = args["xgb_colsample_bytree"]

  model_choice = args["model"]

  Random.seed!(seed)

  println("Loading dataframe from $infile")
  df = readdf(infile; sep = ',')

  # Ensure labels are 0 or 1
  raw_labels = df.label
  if !(all(x -> x in (0, 1), raw_labels))
    error("Label column must contain only 0 (ancient) or 1 (modern)")
  end

  # Features (all columns except label)
  feature_cols = filter(c -> c != :label, names(df))
  X = Matrix(df[:, feature_cols])

  n = size(X, 1)
  if test_frac <= 0.0 || test_frac >= 0.5
    error("--test_frac must be between 0.0 and 0.5")
  end

  ####################################################################################################
  # Stratified train/test split
  ####################################################################################################

  byclass = Dict{Int,Vector{Int}}()
  for (i, lab) in enumerate(raw_labels)
    push!(get!(byclass, lab, Int[]), i)
  end

  train_idx = Int[]
  test_idx = Int[]

  for (lab, inds) in byclass
    shuffle!(inds)
    k = Int(round(test_frac * length(inds)))
    if k > 0
      append!(test_idx, inds[1:k])
      append!(train_idx, inds[(k+1):end])
    else
      append!(train_idx, inds)
    end
  end

  # Fallback if test set is empty
  if isempty(test_idx)
    all_idx = collect(1:n)
    shuffle!(all_idx)
    test_n = Int(round(test_frac * n))
    test_idx = all_idx[1:test_n]
    train_idx = setdiff(all_idx, test_idx)
  end

  X_train = X[train_idx, :]
  X_test = X[test_idx, :]

  ####################################################################################################
  # Model branches
  ####################################################################################################

  if model_choice == "tree"

    y_train = Int.(raw_labels[train_idx]) .+ 1
    y_test = Int.(raw_labels[test_idx]) .+ 1

    println("Training Decision Tree Classifier...")
    model =
      DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    fit!(model, X_train, y_train)

    y_pred_test = DecisionTree.predict(model, X_test)

  elseif model_choice == "forest"

    y_train = Int.(raw_labels[train_idx]) .+ 1
    y_test = Int.(raw_labels[test_idx]) .+ 1

    println("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
      n_trees = n_trees,
      max_depth = max_depth,
      min_samples_leaf = min_samples_leaf,
      partial_sampling = rf_partial,
    )

    fit!(rf_model, X_train, y_train)
    y_pred_test = DecisionTree.predict(rf_model, X_test)

  elseif model_choice == "xgboost"

    y_train_xgb = Float32.(raw_labels[train_idx])
    y_test_xgb = Float32.(raw_labels[test_idx])

    println("Training XGBoost Classifier...")

    dtrain = DMatrix(X_train, label = y_train_xgb)
    dtest = DMatrix(X_test, label = y_test_xgb)

    params = Dict(
      "objective" => "binary:logistic",
      "eta" => xgb_eta,
      "max_depth" => xgb_max_depth,
      "subsample" => xgb_subsample,
      "colsample_bytree" => xgb_colsample_bytree,
      "eval_metric" => "logloss",
      "seed" => seed,
    )

    bst = xgboost(dtrain, num_round = xgb_rounds, params = params)

    prob_test = XGBoost.predict(bst, dtest)
    y_pred_test = Int.(prob_test .>= 0.5) .+ 1

    y_test = Int.(raw_labels[test_idx]) .+ 1

  else
    error("Unknown model choice: $model_choice. Use 'tree', 'forest', or 'xgboost'.")
  end

  ####################################################################################################
  # Standardized Output: sample, truth, prediction
  ####################################################################################################

  if outfile !== nothing
    truth = y_test .- 1
    pred = y_pred_test .- 1

    outdf = DataFrame(sample = test_idx, truth = truth, prediction = pred)

    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

####################################################################################################
