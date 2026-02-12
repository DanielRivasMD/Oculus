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

  Random.seed!(seed)

  println("Loading dataframe from $infile")
  df = readdf(infile; sep = ',')

  # Validate label column
  if !(:label in names(df))
    error("Input CSV must contain a 'label' column with values 0 (ancient) or 1 (modern)")
  end

  # Ensure labels are 0 or 1
  raw_labels = df.label
  if !(all(x -> x in (0, 1), raw_labels))
    error("Label column must contain only 0 (ancient) or 1 (modern)")
  end

  # DecisionTree.jl expects integer class ids starting at 1.
  # Convert 0->1, 1->2 for training, and convert back when writing predictions.
  y = Int.(raw_labels) .+ 1

  # Features (all columns except label)
  feature_cols = filter(c -> c != :label, names(df))
  X = Matrix(df[:, feature_cols])

  n = size(X, 1)
  if test_frac <= 0.0 || test_frac >= 0.5
    error("--test_frac must be between 0.0 and 0.5")
  end

  # Stratified train/test split
  byclass = Dict{Int,Vector{Int}}()
  for (i, lab) in enumerate(y)
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
  y_train = y[train_idx]
  X_test = X[test_idx, :]
  y_test = y[test_idx]

  println(
    "Training Decision Tree Classifier with max_depth=$max_depth min_samples_leaf=$min_samples_leaf",
  )
  model = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
  fit!(model, X_train, y_train)

  println("\nTrained tree structure:")
  print_tree(model)

  # Predictions
  y_pred_train = predict(model, X_train)
  y_pred_test = predict(model, X_test)

  # Metrics: confusion matrix and accuracy only
  nclasses = length(unique(y))
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

  # Write predictions if requested (convert back to 0/1 labels)
  if outfile !== nothing
    pred_labels = Int.(y_pred_test) .- 1
    true_labels = Int.(y_test) .- 1
    outdf = DataFrame(index = test_idx, true = true_labels, pred = pred_labels)
    writedf(outfile, outdf; sep = ',')
    println("Predictions written to $outfile")
  end
end

####################################################################################################
