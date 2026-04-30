####################################################################################################

module DAFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..DACore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "deamination_analysis",
  [
    Stage(
      "01_load",
      (config, _) -> begin
        if haskey(config, "single")
          seqs = DACore.load_fasta(config["single"])
          name = DACore.fname(config["single"])
          return (mode = :single, name = name, seqs = seqs)
        elseif haskey(config, "ancient") && haskey(config, "modern")
          seqs_ancient = DACore.load_fasta(config["ancient"])
          seqs_modern = DACore.load_fasta(config["modern"])
          return (
            mode = :dual,
            ancient_name = config["ancient_name"],
            modern_name = config["modern_name"],
            seqs_ancient = seqs_ancient,
            seqs_modern = seqs_modern,
          )
        else
          error("Provide either --single or both --ancient and --modern")
        end
      end,
      "1.0",
    ),
    Stage(
      "02_compute",
      (config, prev) -> begin
        if prev.mode == :single
          comp = DACore.position_composition(prev.seqs)
          return (mode = :single, comp = comp)
        else
          comp_ancient = DACore.position_composition(prev.seqs_ancient)
          comp_modern = DACore.position_composition(prev.seqs_modern)
          return (mode = :dual, comp_ancient = comp_ancient, comp_modern = comp_modern)
        end
      end,
      "1.0",
    ),
    Stage(
      "03_write_csv",
      (config, prev) -> begin
        if prev.mode == :single
          DACore.write_csv_single(config["csv"], prev.comp)
        else
          DACore.write_csv_dual(
            config["csv"],
            prev.modern_name,
            prev.ancient_name,
            prev.comp_modern,
            prev.comp_ancient,
          )
        end
      end,
      "1.0",
    ),
    Stage(
      "04_plot",
      (config, prev) -> begin
        DACore.plot_composition(config["csv"]; outfile = config["png"])
      end,
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
