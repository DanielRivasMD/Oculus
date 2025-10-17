
using BSON

using CUDA
using Flux
using Flux

include("src/config/params.jl")
include("src/config/sample.jl")

bs = BSON.load("model/fold1_2025-10-17_123329.bson")


using UnicodePlots
using Weave

"""
    plot_metrics(bs; backend=:unicode, outfile="metrics.html")

Plot training/validation losses and accuracies from a BSON checkpoint dictionary.
Also logs model architecture (`model_cpu`), hyperparameters (`hparams`), and sample parameters (`sparams`).

- `backend = :unicode` → terminal plots with UnicodePlots
- `backend = :weave`   → generate an HTML report with Weave
"""
function plot_metrics(bs::Dict{Symbol,Any}; backend=:unicode, outfile="metrics.html")
    train_losses = bs[:train_losses]
    val_losses   = bs[:val_losses]
    train_accs   = bs[:train_accs]
    val_accs     = bs[:val_accs]
    epochs       = 1:length(train_losses)

    model_cpu = bs[:model_cpu]
    hparams   = bs[:hparams]
    sparams   = bs[:sparams]

    if backend == :unicode
        # Log model + params
        println("=== Model ===")
        println(model_cpu)
        println("\n=== Hyperparameters (CNNParams) ===")
        println(hparams)
        println("\n=== Sample parameters (SampleParams) ===")
        println(sparams)

        # Loss plot
        plt1 = lineplot(epochs, train_losses; name="train_loss", xlabel="epoch", ylabel="loss")
        lineplot!(plt1, epochs, val_losses; name="val_loss")
        println(plt1)

        # Accuracy plot
        plt2 = lineplot(epochs, train_accs; name="train_acc", xlabel="epoch", ylabel="accuracy")
        lineplot!(plt2, epochs, val_accs; name="val_acc")
        println(plt2)

elseif backend == :weave
    # Convert model summary to string
    io = IOBuffer()
    show(io, MIME"text/plain"(), bs[:model_cpu])
    model_str = String(take!(io))

    # Build a markdown report with model + params + plots
    md = """
    # Training Report

    ## Model Summary
    ```
    $model_str
    ```

    ## Hyperparameters
    ```
    $(bs[:hparams])
    ```

    ## Sample Parameters
    ```
    $(bs[:sparams])
    ```

    ## Final Metrics
    - Final validation loss: $(bs[:final_val_loss])
    - Final validation accuracy: $(bs[:final_val_acc])

    ## Loss Curves
    ```julia
    using Plots
    plot($(1:length(bs[:train_losses])), $(bs[:train_losses]), label="train_loss")
    plot!($(1:length(bs[:val_losses])), $(bs[:val_losses]), label="val_loss")
    ```

    ## Accuracy Curves
    ```julia
    using Plots
    plot($(1:length(bs[:train_accs])), $(bs[:train_accs]), label="train_acc")
    plot!($(1:length(bs[:val_accs])), $(bs[:val_accs]), label="val_acc")
    ```
    """
    tmpfile = "metrics.jmd"
    write(tmpfile, md)
    Weave.weave(tmpfile, out_path=outfile, doctype="md2html")
    println("HTML report written to $outfile")
    else
        error("Unknown backend: $backend")
    end
end

# Terminal plots
plot_metrics(bs; backend=:unicode)

# HTML report
plot_metrics(bs; backend=:weave, outfile="metrics.html")

