####################################################################################################

module FEFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..FECore

####################################################################################################

export features_flow

####################################################################################################

const features_flow = Config(
  "feature_extraction",
  [
    Stage("load_modern", (config, _) -> FECore.load_fasta(config["modern"]), "1.0"),
    Stage("load_ancient", (config, _) -> FECore.load_fasta(config["ancient"]), "1.0"),
    Stage(
      "build_modern",
      (config, prev) ->
        FECore.build_df(prev["load_modern"], 1; onehot = config["onehot"]),
      "1.0",
    ),
    Stage(
      "build_ancient",
      (config, prev) ->
        FECore.build_df(prev["load_ancient"], 0; onehot = config["onehot"]),
      "1.0",
    ),
    Stage("merge", (_, prev) -> vcat(prev["build_modern"], prev["build_ancient"]), "1.0"),
    Stage(
      "write",
      (config, prev) -> FECore.writedf(config["out"], prev["merge"]; sep = ','),
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
