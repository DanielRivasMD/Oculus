####################################################################################################

module PEFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..PECore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "performance_evaluation",
  [
    Stage("01_load_predictions", (config, _) -> PECore.load_predictions(config["infile"]), "1.0"),
    Stage("02_compute_metrics", (config, prev) -> begin
      df = prev["01_load_predictions"]
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

####################################################################################################

end

####################################################################################################
