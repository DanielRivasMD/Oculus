####################################################################################################

module RGFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..RGCore
using ..USCore

####################################################################################################

export flow

####################################################################################################

# TODO: verify which section of the split is used for training & for testing
const flow = Config(
  "regression_analysis",
  [
    Stage("01_load_data", (config, _) -> USCore.load_data(config["infile"]), "1.0"),
    Stage(
      "02_split_data",
      (config, prev) ->
        USCore.split_data(prev["01_load_data"], config["split"], config["seed"]),
      "1.0",
    ),
    Stage(
      "03_train",
      (config, prev) -> begin
        train_df = prev["02_split_data"][1]
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
      "04_predict",
      (config, prev) -> begin
        test_df = prev["02_split_data"][2]
        if isempty(test_df)
          return Float64[]
        end
        if config["reg"] == "none"
          model = prev["03_train"]
          return RGCore.predict_logistic(model, test_df)
        else
          intercept, betas, _ = prev["03_train"]
          feature_cols = setdiff(names(test_df), ["label"])
          X_test = Matrix(test_df[:, feature_cols])
          return RGCore.predict_glmnet(intercept, betas, X_test)
        end
      end,
      "1.0",
    ),
    Stage("05_evaluate", (config, prev) -> begin
      test_df = prev["02_split_data"][2]
      if isempty(test_df)
        return Dict()
      end
      truth = Int.(test_df.label)
      preds = prev["04_predict"][2]
      return USCore.performance(truth, preds)
    end, "1.0"),
    Stage(
      "06_write_output",
      (config, prev) -> begin
        if config["out"] === nothing
          return nothing
        end
        test_df = prev["02_split_data"][2]
        if isempty(test_df)
          return nothing
        end
        # TODO: we need original indices, not preserved. output row numbers from the test set for now. store indices in split_data?
        test_indices = collect(1:size(test_df, 1))  # placeholder - not the original indices.
        truth = Int.(test_df.label)
        preds = prev["04_predict"][2]
        USCore.write_predictions(config["out"], preds, test_indices, truth)
        return nothing
      end,
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
