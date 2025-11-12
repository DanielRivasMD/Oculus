####################################################################################################
# Imports
####################################################################################################

using CUDA                   # GPU acceleration

redirect_stderr(devnull) do
  @eval using Flux           # Core deep learning library
  @eval using Flux           # DOC: loaded twice to avoid dependency data race issues
end

using BSON

####################################################################################################
# Load configuration
####################################################################################################

begin
  # Load path definitions
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  # Load configuration structs
  include(joinpath(Paths.CONFIG, "sample.jl"))    # SampleParams (data config)
  include(joinpath(Paths.CONFIG, "params.jl"))    # CNNParams (hyperparameters)
  include(joinpath(Paths.CONFIG, "args.jl"))      # Args API (now includes infer_args)
end;

####################################################################################################
# Plotting utilities
####################################################################################################

using UnicodePlots
using Plots
using Weave

"""
    plot_metrics(bs; mode=:term, outfile="metrics.html")

Render performance metrics from a BSON checkpoint.

# Arguments
- `bs` : Dict from BSON.load
- `mode` : `:term` for terminal plots (UnicodePlots), `:html` for HTML report (Plots.jl)
- `outfile` : output HTML file if `mode=:html`

# Notes
- Always prints model architecture, CNNParams, and SampleParams.
- If `mode=:html`, a temporary .jmd file is created and deleted after weaving.

# Example
```julia
using BSON
bs = BSON.load("model.bson")

# Terminal plots
plot_metrics(bs; mode=:term)

# HTML report
plot_metrics(bs; mode=:html, outfile="reports/cnn3.html")
"""
function plot_metrics(bs::Dict{Symbol,Any}; mode = :term, outfile = "metrics.html")
  train_losses = bs[:train_losses]
  val_losses = bs[:val_losses]
  train_accs = bs[:train_accs]
  val_accs = bs[:val_accs]
  epochs = 1:length(train_losses)

  model_cpu = bs[:model_cpu]
  hparams = bs[:hparams]
  sparams = bs[:sparams]

  println("=== Model ===")
  println(model_cpu)
  println("\n=== Hyperparameters (CNNParams) ===")
  println(hparams)
  println("\n=== Sample parameters (SampleParams) ===")
  println(sparams)

  if mode == :term
    # Terminal plots with UnicodePlots, fixed y-axis
    plt1 = lineplot(
      epochs,
      train_losses;
      name = "train_loss",
      xlabel = "epoch",
      ylabel = "loss",
      ylim = (0, 1),
    )
    lineplot!(plt1, epochs, val_losses; name = "val_loss")
    println(plt1)

    plt2 = lineplot(
      epochs,
      train_accs;
      name = "train_acc",
      xlabel = "epoch",
      ylabel = "accuracy",
      ylim = (0, 1),
    )
    lineplot!(plt2, epochs, val_accs; name = "val_acc")
    println(plt2)

  elseif mode == :html
    io = IOBuffer()
    show(io, MIME"text/plain"(), model_cpu)
    model_str = String(take!(io))

    md = """
    # Training Report

    ## Model Summary
    ```
    $model_str
    ```

    ## Hyperparameters
    ```
    $hparams
    ```

    ## Sample Parameters
    ```
    $sparams
    ```

    ## Final Metrics
    - Final validation loss: $(last(val_losses))
    - Final validation accuracy: $(last(val_accs))

    ## Loss Curves
    ```julia
    using Plots
    plot($epochs, $train_losses, label="train_loss", ylim=(0,1))
    plot!($epochs, $val_losses, label="val_loss")
    ```

    ## Accuracy Curves
    ```julia
    using Plots
    plot($epochs, $train_accs, label="train_acc", ylim=(0,1))
    plot!($epochs, $val_accs, label="val_acc")
    ```
    """

    tmpfile = "metrics.jmd"
    write(tmpfile, md)
    Weave.weave(tmpfile, out_path = outfile, doctype = "md2html")
    rm(tmpfile; force = true)
    println("HTML report written to $outfile")
  else
    error("Unknown mode: $mode (use :term or :html)")
  end
end

####################################################################################################
# Run reports (CLI only)
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing
  args = perf_args()
  bs = BSON.load(args["model"])
  mode = get(args, "mode", "html") == "term" ? :term : :html
  plot_metrics(bs; mode = mode, outfile = args["out"])
end

####################################################################################################
