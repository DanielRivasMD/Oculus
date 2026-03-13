include("src/config/paths.jl")
using .Paths

using DataFrames
using FilePathsBase: basename, splitext
using GLM
using GLMNet
using LinearAlgebra
using StatsModels
using Random

include(joinpath(Paths.UTIL, "params.jl"))
include(joinpath(Paths.UTIL, "cross_validation.jl"))
include(joinpath(Paths.UTIL, "ioDataFrame.jl"))
include(joinpath(Paths.UTIL, "performance.jl"))

infile = "tmp/features_engineered.csv"
outfile = "tmp/repl_regression.csv"
reg_method = "none"
split_frac = 0.6

println("Loading dataframe from $infile")
df = readdf(infile; sep = ',')

# Extract features and label
label_col = "label"
feature_cols = setdiff(names(df), [label_col])

X = Matrix(df[:, feature_cols])
y = Vector(df[:, label_col])

# Validate label values are 0/1
if !(all(x -> x in (0, 1), y))
  error("Label column must contain only 0 (ancient) or 1 (modern) for logistic regression")
end

if split_frac > 0
  println("Performing stratified split with test fraction = $split_frac")

  test_idx = per_class_splits(y, split_frac)
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

println("\nFitting logistic regression (binomial, logit link) baseline...")

df_train = df[train_idx, :]
f = Term(:label) ~ sum(Term.(Symbol.(feature_cols)))
model_glm = glm(f, df_train, Binomial(), LogitLink())

# GLM predicted probabilities
probs_glm = GLM.predict(model_glm, df[test_idx, :])
preds_glm_class = Int.(probs_glm .>= 0.5)

####################################################################################################
# R equivalent
####################################################################################################

# # Fit the model
# model_glm <- glm(label ~ ., data = df_train, family = binomial(link = "logit"))

# # Predicted probabilities on test set
# probs_glm <- predict(model_glm, newdata = df_test, type = "response")

# # Convert to class predictions (threshold 0.5)
# preds_glm_class <- as.integer(probs_glm >= 0.5)

####################################################################################################

outdf = DataFrame(sample = test_idx, truth = Int.(y_test), prediction = preds_glm_class)

pf = performance(outdf.prediction .+ 1, outdf.truth .+ 1)
println(pf)

writedf(outfile, outdf; sep = ',')
println("Predictions written to $outfile")

