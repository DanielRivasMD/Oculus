####################################################################################################

module DTFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..DTCore
using ..USCore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "decision_tree_analysis",
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
        if isempty(train_df)
          error("Training set is empty")
        end
        feature_cols = setdiff(names(train_df), ["label"])
        X_train = Matrix(train_df[:, feature_cols])
        y_train = Int.(train_df.label)

        model_type = config["model"]

        if model_type == "tree"
          return DTCore.train_decision_tree(
            X_train,
            y_train,
            config["max_depth"],
            config["min_samples_leaf"],
          )
        elseif model_type == "forest"
          return DTCore.train_random_forest(
            X_train,
            y_train,
            config["max_depth"],
            config["min_samples_leaf"],
            config["n_trees"],
            config["rf_partial_sampling"],
          )
        elseif model_type == "xgboost"
          return DTCore.train_xgboost(
            X_train,
            y_train,
            config["xgb_rounds"],
            config["xgb_eta"],
            config["xgb_max_depth"],
            config["xgb_subsample"],
            config["xgb_colsample_bytree"],
            config["seed"],
          )
        else
          error("Unknown model_type: $model_type")
        end
      end,
      "1.0",
    ),
    Stage(
      "04_predict",
      (config, prev) -> begin
        test_df = prev["02_split_data"][2]
        if isempty(test_df)
          return Int[]
        end
        feature_cols = setdiff(names(test_df), ["label"])
        X_test = Matrix(test_df[:, feature_cols])
        model = prev["03_train"]
        model_type = config["model"]
        return DTCore.predict(model, X_test, model_type)
      end,
      "1.0",
    ),
    Stage("05_evaluate", (config, prev) -> begin
      test_df = prev["02_split_data"][2]
      if isempty(test_df)
        return Dict()
      end
      truth = Int.(test_df.label)
      preds = prev["04_predict"]
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
        test_indices = collect(1:size(test_df, 1))  # placeholder - not the original indices.
        truth = Int.(test_df.label)
        preds = prev["04_predict"]
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
