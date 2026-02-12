#!/usr/bin/env julia

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

begin
  include(joinpath(Paths.UTIL, "ioDataFrame.jl"))
end;

####################################################################################################
# Helpers
####################################################################################################

function confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int}, nclasses::Int)
  cm = zeros(Int, nclasses, nclasses)
  for (t, p) in zip(y_true, y_pred)
    cm[t, p] += 1
  end
  return cm
end

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

  # # Validate label column
  # if !(:label in names(df))
  #   error("Input CSV must contain a 'label' column with values 0 (ancient) or 1 (modern)")
  # end

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

  # Stratified train/test split
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

  # If test set ended up empty (small classes), fall back to random sampling
  if isempty(test_idx)
    all_idx = collect(1:n)
    shuffle!(all_idx)
    test_n = Int(round(test_frac * n))
    test_idx = all_idx[1:test_n]
    train_idx = setdiff(all_idx, test_idx)
  end

  X_train = X[train_idx, :]
  X_test = X[test_idx, :]

  ################################################################################################
  # Model branches
  ################################################################################################

  if model_choice == "tree"
    # DecisionTree.jl expects labels starting at 1
    y_train = Int.(raw_labels[train_idx]) .+ 1
    y_test = Int.(raw_labels[test_idx]) .+ 1

    println(
      "Training Decision Tree Classifier with max_depth=$max_depth min_samples_leaf=$min_samples_leaf",
    )
    model =
      DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    fit!(model, X_train, y_train)

    println("\nTrained tree structure:")
    print_tree(model)

    y_pred_train = DecisionTree.predict(model, X_train)
    y_pred_test = DecisionTree.predict(model, X_test)

    print_tree(model)

  elseif model_choice == "random_forest"
    # DecisionTree.jl RandomForestClassifier expects labels starting at 1
    y_train = Int.(raw_labels[train_idx]) .+ 1
    y_test = Int.(raw_labels[test_idx]) .+ 1

    println(
      "Training Random Forest Classifier with n_trees=$n_trees max_depth=$max_depth min_samples_leaf=$min_samples_leaf partial_sampling=$rf_partial",
    )
    rf_model = RandomForestClassifier(
      n_trees = n_trees,
      max_depth = max_depth,
      min_samples_leaf = min_samples_leaf,
      partial_sampling = rf_partial,
    )

    fit!(rf_model, X_train, y_train)

    println("Random Forest trained. Forest size: $(rf_model.n_trees) trees")

    y_pred_train = DecisionTree.predict(rf_model, X_train)
    y_pred_test = DecisionTree.predict(rf_model, X_test)

    # Optional feature importance
    try
      if hasproperty(rf_model, :feature_importance)
        fi = rf_model.feature_importance
        println("\nFeature importance (top 10):")
        idxs = sortperm(fi, rev = true)[1:min(10, length(fi))]
        for i in idxs
          println("  $(feature_cols[i]) => $(round(fi[i], digits=6))")
        end
      end
    catch
      # ignore if not present
    end

  elseif model_choice == "xgboost"
    # XGBoost expects labels as 0/1 for binary:logistic
    y_train_xgb = Float32.(raw_labels[train_idx])
    y_test_xgb = Float32.(raw_labels[test_idx])

    println(
      "Training XGBoost classifier with rounds=$xgb_rounds eta=$xgb_eta max_depth=$xgb_max_depth subsample=$xgb_subsample colsample_bytree=$xgb_colsample_bytree",
    )

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

    watchlist = [(dtrain, "train"), (dtest, "eval")]
    bst = xgboost(dtrain, num_round = xgb_rounds, params = params, evals = watchlist)

    # Predict probabilities and threshold at 0.5
    prob_train = XGBoost.predict(bst, dtrain)
    prob_test = XGBoost.predict(bst, dtest)

    y_pred_train = Int.(prob_train .>= 0.5) .+ 1   # convert to 1/2 for metrics
    y_pred_test = Int.(prob_test .>= 0.5) .+ 1

    # Optionally print top features by importance
    try
      fmap = xgboost_feature_score(bst)
      if !isempty(fmap)
        println("\nXGBoost feature importance (top 10):")
        # fmap is Dict{String,Float64} with "f0","f1",...
        pairs_sorted = sort(collect(fmap), by = x -> x[2], rev = true)
        for (i, (fname, score)) in enumerate(pairs_sorted[1:min(10, length(pairs_sorted))])
          # fname like "f0" -> index
          idx = parse(Int, replace(fname, "f" => "")) + 1
          println("  $(feature_cols[idx]) => $(round(score, digits=6))")
        end
      end
    catch
      # ignore if feature importance not available
    end

    # For metrics we need y_train/y_test in 1/2 form
    y_train = Int.(raw_labels[train_idx]) .+ 1
    y_test = Int.(raw_labels[test_idx]) .+ 1

  else
    error("Unknown model choice: $model_choice. Use 'tree', 'random_forest', or 'xgboost'.")
  end

  ################################################################################################
  # Metrics
  ################################################################################################

  nclasses = length(unique(y_test))
  cm_train = confusion_matrix(y_train, y_pred_train, nclasses)
  cm_test = confusion_matrix(y_test, y_pred_test, nclasses)

  acc_train = sum(diag(cm_train)) / sum(cm_train)
  acc_test = sum(diag(cm_test)) / sum(cm_test)

  println("\nTraining metrics")
  println("Accuracy: $(round(acc_train, digits=4))")
  println("Confusion matrix:")
  println(cm_train)

  println("\nTest metrics")
  println("Accuracy: $(round(acc_test, digits=4))")
  println("Confusion matrix:")
  println(cm_test)

  ################################################################################################
  # Write predictions if requested (convert back to 0/1 labels)
  ################################################################################################

  if outfile !== nothing
    # y_pred_test currently in 1/2 form; convert back to 0/1
    pred_labels = Int.(y_pred_test) .- 1
    true_labels = Int.(y_test) .- 1
    outdf = DataFrame(index = test_idx, truth = true_labels, pred = pred_labels)
    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

####################################################################################################
