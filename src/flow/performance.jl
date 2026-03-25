module PEFlow

using Avicenna.Flow: Stage, Config
using DataFrames
using DelimitedFiles
using ..PECore

export performance_flow

function load_predictions(path::String)
  data, header = readdlm(path, ',', header = true)
  df = DataFrame(data, vec(header))
  # Ensure required columns exist
  for col in ["truth", "prediction"]
    if !(col in names(df))
      error("CSV must contain column '$col'")
    end
  end
  return df
end

const performance_flow = Config(
  "performance_evaluation",
  [
    Stage("load_predictions", (config, _) -> load_predictions(config["infile"]), "1.0"),
    Stage("compute_metrics", (config, prev) -> begin
      df = prev["load_predictions"]
      truth = Int.(df.truth)
      pred = Int.(df.prediction)
      # Build confusion matrix (truth rows, pred columns)
      cm = zeros(Int, 2, 2)
      for (t, p) in zip(truth, pred)
        cm[t+1, p+1] += 1   # because labels 0,1 -> indices 1,2
      end
      return PECore.performance(cm)
    end, "1.0"),
  ],
  "1.0",
)

end
