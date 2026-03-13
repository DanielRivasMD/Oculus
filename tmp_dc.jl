include("src/config/paths.jl")
using .Paths

using ArgParse
using DataFrames
using Random
using DecisionTree
using LinearAlgebra
using XGBoost
using FilePathsBase: basename, splitext

include(joinpath(Paths.UTIL, "params.jl"))
include(joinpath(Paths.UTIL, "ioDataFrame.jl"))

infile = "tmp/features_engineered.csv"
outfile = "tmp/repl_decision_tree.csv"
max_depth = 6
min_samples_leaf = 5
test_frac = 0.2
seed = 1


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

y_train = Int.(raw_labels[train_idx]) .+ 1
y_test = Int.(raw_labels[test_idx]) .+ 1

println("Training Decision Tree Classifier...")
model = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)
fit!(model, X_train, y_train)

y_pred_test = DecisionTree.predict(model, X_test)

truth = y_test .- 1
pred = y_pred_test .- 1

outdf = DataFrame(sample = test_idx, truth = truth, prediction = pred)

writedf(outfile, outdf; sep = ',')
println("Predictions written to $outfile")
