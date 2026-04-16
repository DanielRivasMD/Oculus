module RGFlow

using Avicenna.Flow: Stage, Config
using GLM
using ..RGCore

export regression_flow

const regression_flow = Config(
  "regression_analysis",
  [
    Stage("load_data", (config, _) -> RGCore.load_data(config["infile"]), "1.0"),
    Stage(
      "split_data",
      (config, prev) ->
        RGCore.split_data(prev["load_data"], config["split"], config["seed"]),
      "1.0",
    ),
    Stage(
      "train",
      (config, prev) -> begin
        train_df = prev["split_data"][1]
        if config["reg"] == "none"
          return RGCore.train_logistic(train_df)
        else
          return RGCore.train_glmnet(
            train_df,
            config["reg"],
            config["alpha"],
            config["nfolds"],
            config["seed"],
          )
        end
      end,
      "1.0",
    ),
    Stage(
      "predict",
      (config, prev) -> begin
        test_df = prev["split_data"][2]
        if isempty(test_df)
          return Float64[]
        end
        if config["reg"] == "none"
          model = prev["train"]
          # Predict probabilities
          probs = GLM.predict(model, test_df)
          preds = Int.(probs .>= 0.5)
          return (probs, preds)
        else
          intercept, betas, _ = prev["train"]
          feature_cols = setdiff(names(test_df), ["label"])
          X_test = Matrix(test_df[:, feature_cols])
          probs = RGCore.predict_glmnet(intercept, betas, X_test)
          preds = Int.(probs .>= 0.5)
          return (probs, preds)
        end
      end,
      "1.0",
    ),
    Stage("evaluate", (config, prev) -> begin
      test_df = prev["split_data"][2]
      if isempty(test_df)
        return Dict()
      end
      truth = Int.(test_df.label)
      preds = prev["predict"][2]
      return RGCore.evaluate(truth, preds)
    end, "1.0"),
    Stage(
      "write_output",
      (config, prev) -> begin
        if config["out"] === nothing
          return nothing
        end
        test_df = prev["split_data"][2]
        if isempty(test_df)
          return nothing
        end
        # We need original indices; we didn't preserve them. We'll just output row numbers from the test set.
        # Alternative: we could have stored indices in split_data; but for simplicity, we'll use row numbers (1-based).
        test_indices = collect(1:size(test_df, 1))  # placeholder – not the original indices.
        truth = Int.(test_df.label)
        preds = prev["predict"][2]
        RGCore.write_predictions(config["out"], preds, test_indices, truth)
        return nothing
      end,
      "1.0",
    ),
  ],
  "1.0",
)

end
