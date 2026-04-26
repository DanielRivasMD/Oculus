module INFlow

using Avicenna.Flow: Stage, Config
using ..INCore

export flow

const flow = Config(
  "inference",
  [
    Stage(
      "01_load_model",
      (config, _) -> begin
        model, hparams, sparams, model_len = INCore.load_model(config["model"])
        return (model = model, model_len = model_len)
      end,
      "1.0",
    ),
    Stage("02_load_sequences", (config, _) -> INCore.load_sequences(config["data"]), "1.0"),
    Stage(
      "03_predict",
      (config, prev) -> begin
        model = prev["01_load_model"].model
        model_len = prev["01_load_model"].model_len
        seqs = prev["02_load_sequences"]
        preds, probs = INCore.predict_all(model, seqs, model_len)

        # Build sample IDs: basename + model name + index
        datafile = basename(config["data"])
        sample_base = replace(datafile, r"\.fastq.*" => "")
        rootdir = basename(dirname(config["data"]))
        modelname = basename(config["model"]) |> splitext |> first
        ids = ["$rootdir-$sample_base-$modelname-$i" for i = 1:length(seqs)]

        return (preds = preds, probs = probs, ids = ids)
      end,
      "1.0",
    ),
    Stage(
      "04_write_csv",
      (config, prev) -> begin
        ids = prev["03_predict"].ids
        probs = prev["03_predict"].probs
        INCore.write_predictions(config["out"], ids, probs)
        return nothing
      end,
      "1.0",
    ),
  ],
  "1.0",
)

end
