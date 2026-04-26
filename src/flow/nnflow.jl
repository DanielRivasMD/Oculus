####################################################################################################

module NNFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..NNCore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "neural_network_training",
  [
    Stage(
      "01_load_configs",
      (config, _) -> begin
        hparams = NNCore.loadHparams(config["cnn"])
        sparams = NNCore.loadSparams(config["sample"])
        return (hparams = hparams, sparams = sparams)
      end,
      "1.0",
    ),
    Stage(
      "02_train",
      (config, prev) -> begin
        hparams = prev["01_load_configs"].hparams
        sparams = prev["01_load_configs"].sparams
        out_base = config["out"]
        return NNCore.train_and_save(hparams, sparams, out_base)
      end,
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
