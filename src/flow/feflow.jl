####################################################################################################

module FEFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..FECore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "feature_extraction",
  [
    Stage("01_load_ancient", (config, _) -> FECore.load_fasta(config["ancient"]), "1.0"),
    Stage("01_load_modern", (config, _) -> FECore.load_fasta(config["modern"]), "1.0"),
    Stage(
      "02_build_ancient",
      (config, prev) ->
        FECore.build_df(prev["01_load_ancient"], 0; onehot = config["onehot"]),
      "1.0",
    ),
    Stage(
      "02_build_modern",
      (config, prev) ->
        FECore.build_df(prev["01_load_modern"], 1; onehot = config["onehot"]),
      "1.0",
    ),
    Stage("03_merge", (_, prev) -> vcat(prev["02_build_ancient"], prev["02_build_modern"]), "1.0"),
    Stage(
      "04_write",
      (config, prev) -> FECore.writedf(config["out"], prev["03_merge"]; sep = ','),
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
