using DataFrames
using DelimitedFiles
using Plots

function readdf(path; sep = ',')
  data, header = readdlm(path, sep, header = true)
  return DataFrame(data, vec(header))
end

input_path = "data/cache/features/engineered.csv"
output_dir = "data/graph/png"

df = readdf(input_path)

# Split by label (0 = ancient, 1 = modern)
ancient = df[df.label.==0, :]
modern = df[df.label.==1, :]

features = ["ga3p_5", "ct5p_5"]

for feat in features
  p_modern = histogram(
    modern[:, feat],
    bins = 0:0.1:1,
    alpha = 0.7,
    color = :red,
    label = "Modern",
    xlabel = feat,
    ylabel = "Frequency",
    title = "Distribution of $feat (Modern)",
    legend = :topright,
    normalize = :pdf,   # use density to compare shapes
  )
  savefig(p_modern, joinpath(output_dir, "hist_modern_$feat.png"))

  p_ancient = histogram(
    ancient[:, feat],
    bins = 0:0.1:1,
    alpha = 0.7,
    color = :blue,
    label = "Ancient",
    xlabel = feat,
    ylabel = "Frequency",
    title = "Distribution of $feat (Ancient)",
    legend = :topright,
    normalize = :pdf,
  )
  savefig(p_ancient, joinpath(output_dir, "hist_ancient_$feat.png"))

  p_combined = histogram(
    modern[:, feat],
    bins = 0:0.1:1,
    alpha = 0.5,
    color = :red,
    label = "Modern",
    normalize = :pdf,
  )
  histogram!(
    p_combined,
    ancient[:, feat],
    bins = 0:0.1:1,
    alpha = 0.5,
    color = :blue,
    label = "Ancient",
  )
  title!(p_combined, "Distribution of $feat")
  savefig(p_combined, joinpath(output_dir, "hist_combined_$feat.png"))
end

println("All histograms saved to $output_dir")
