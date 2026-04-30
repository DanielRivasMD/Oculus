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
        # @info prev
        if prev["01_load"].mode == :single
          comp = DACore.position_composition(prev["01_load"].seqs)
          return (mode = :single, comp = comp)
        else
          comp_ancient = DACore.position_composition(prev["01_load"].seqs_ancient)
          comp_modern = DACore.position_composition(prev["01_load"].seqs_modern)
          return (mode = :dual, comp_ancient = comp_ancient, comp_modern = comp_modern)
        end
      end,
      "1.0",
    ),
    Stage(
      "03_write_csv",
      (config, prev) -> begin
        if prev["02_compute"].mode == :single
          DACore.write_csv_single(config["csv"], prev["02_compute"].comp)
        else
          DACore.write_csv_dual(
            config["csv"],
            prev["01_load"].modern_name,
            prev["01_load"].ancient_name,
            prev["02_compute"].comp_modern,
            prev["02_compute"].comp_ancient,
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
