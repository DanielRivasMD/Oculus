#!/usr/bin/env julia

using DelimitedFiles
using Plots

# Helper to extract file names from headers
fname(path::String) = splitpath(path)[end]

function plot_composition(csvfile::String; outfile::Union{Nothing,String} = nothing)
  # Read the CSV file into a matrix of strings
  raw = readdlm(csvfile, ',', String)

  header = raw[1, :]
  data = raw[2:end, :]

  positions = parse.(Int, data[:, 1])

  # Extract file names from headers
  file1 = split(header[2], "_")[1]
  file2 = split(header[6], "_")[1]

  # Colors for bases
  colors = Dict("A" => "green", "T" => "red", "G" => "orange", "C" => "blue")

  plt = plot(
    size = (1000, 600),
    xlabel = "Position",
    ylabel = "Percentage (%)",
    title = "Base Composition Along Sequence",
  )

  # Plot file1 (solid lines)
  for (j, base) in enumerate(["A", "T", "G", "C"])
    col = parse.(Float64, data[:, 1+j])
    plot!(
      plt,
      positions,
      col,
      label = "$(file1) $base",
      color = colors[base],
      linestyle = :solid,
    )
  end

  # Plot file2 (dashed lines)
  for (j, base) in enumerate(["A", "T", "G", "C"])
    col = parse.(Float64, data[:, 5+j])
    plot!(
      plt,
      positions,
      col,
      label = "$(file2) $base",
      color = colors[base],
      linestyle = :dash,
    )
  end

  if outfile !== nothing
    savefig(plt, outfile)
    println("Plot saved to $outfile")
  else
    display(plt)
  end
end

# --- CLI ---
if abspath(PROGRAM_FILE) == @__FILE__
  if length(ARGS) < 1
    println("Usage: julia plot_composition.jl deamination.csv [output.png]")
    exit(1)
  end
  csvfile = ARGS[1]
  outfile = length(ARGS) >= 2 ? ARGS[2] : nothing
  plot_composition(csvfile; outfile = outfile)
end
