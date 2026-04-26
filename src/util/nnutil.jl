module NNCore

####################################################################################################

using Parameters: @with_kw
using BSON
using Dates
using TOML
using FASTX, BioSequences, Random, StatsBase, Flux, CUDA, BSON, Dates
using Flux: DataLoader, onehotbatch, crossentropy
using Optimisers
using MLUtils: DataLoader
using Zygote

import BSON: @save

####################################################################################################

export HyperParams,
  SampleParams,
  loadHparams,
  loadSparams,
  make_dataset,
  buildCNN,
  trainCNN!,
  train_and_save,
  save_checkpoint,
  onehot_encode,
  onehot_batch,
  load_sequences_fasta

####################################################################################################

# ----- Structs --------------------------------------------------------------

@with_kw mutable struct HyperParams
  batchsize::Int = 64
  shuffle::Bool = true
  device::Function = gpu
  kernelsize::Int = 5
  epochs::Int = 100
  train_frac::Float64 = 0.8
  k::Int = 0

  σ::Function = relu
  maxpool::Int = 2
  η::Float64 = 1e-3
  momentum::Float64 = 0.9

  dropouts::Vector{Float64} = [0.2, 0.3, 0.4, 0.5]
  layerouts::Vector{Int} = [32, 64, 128]

  λ::Float64 = NaN
  batchnorm::Bool = true
  dropout::Bool = true
end

@with_kw struct SampleParams
  seqlen::Int
  seed::Int = 42
  modern::String
  ancient::String
end

# ----- Configuration loaders ------------------------------------------------

function symbolise_keys(tbl::Dict)
  Dict(Symbol(k) => v for (k, v) in tbl)
end

function struct_to_dict(s)
  Dict(f => getfield(s, f) for f in fieldnames(typeof(s)))
end

function loadHparams(path::String)::HyperParams
  params = HyperParams()
  cnn_cfg = !isempty(path) ? symbolise_keys(TOML.parsefile(path)["cnn"]) : Dict()
  if haskey(cnn_cfg, :device)
    cnn_cfg[:device] = cnn_cfg[:device] == "gpu" ? gpu : cpu
  end
  if haskey(cnn_cfg, :σ)
    cnn_cfg[:σ] = cnn_cfg[:σ] == "relu" ? relu : tanh
  end
  return HyperParams(; merge(struct_to_dict(params), cnn_cfg)...)
end

function loadSparams(path::String)::SampleParams
  cfg = TOML.parsefile(path)["sample"]
  return SampleParams(;
    seqlen = cfg["seqlen"],
    seed = get(cfg, "seed", 42),
    modern = cfg["modern"],
    ancient = cfg["ancient"],
  )
end

# ----- Data loading ---------------------------------------------------------

nt2ix = Dict(DNA_A => 1, DNA_C => 2, DNA_G => 3, DNA_T => 4)

function onehot_encode(seq::LongDNA{4})
  L = length(seq)
  X = zeros(Float32, L, 4)
  @inbounds for (i, nt) in enumerate(seq)
    ix = get(nt2ix, nt, 0)
    if ix != 0
      X[i, ix] = 1.0f0
    end
  end
  return X
end

function onehot_batch(seqs::Vector{LongDNA{4}})
  maxL = maximum(length, seqs)
  B = length(seqs)
  X = zeros(Float32, maxL, 4, B)
  @inbounds for (b, s) in enumerate(seqs)
    Xi = onehot_encode(s)
    L = size(Xi, 1)
    X[1:L, :, b] = Xi
  end
  return X
end

function load_sequences_fasta(path::AbstractString)
  FASTA.Reader(open(path)) do reader
    [sequence(LongDNA{4}, record) for record in reader]
  end
end

function load_balanced_data(params::SampleParams)
  modern = load_sequences_fasta(params.modern)
  ancient = load_sequences_fasta(params.ancient)
  println("French:     $(length(modern)) reads")
  println("Neandertal: $(length(ancient)) reads")

  modern_filt = filter(seq -> length(seq) == params.seqlen, modern)
  ancient_filt = filter(seq -> length(seq) == params.seqlen, ancient)
  minN = min(length(modern_filt), length(ancient_filt))
  Random.seed!(params.seed)
  modern_bal = sample(modern_filt, minN; replace = false)
  ancient_bal = sample(ancient_filt, minN; replace = false)
  println("Balanced: $minN per class")
  all_seqs = vcat(modern_bal, ancient_bal)
  labels = vcat(zeros(Int, minN), ones(Int, minN))
  return all_seqs, labels
end

# ----- Split indices --------------------------------------------------------

function split_indices(B::Int, hparams::HyperParams, sparams::SampleParams)
  Random.seed!(sparams.seed)
  idx = shuffle(1:B)
  if hparams.k == 0
    ntrain = round(Int, hparams.train_frac * B)
    return [(train = idx[1:ntrain], val = idx[ntrain+1:end])]
  else
    foldsize = ceil(Int, B / hparams.k)
    return [
      (
        train = setdiff(idx, idx[((i-1)*foldsize+1):min(i * foldsize, B)]),
        val = idx[((i-1)*foldsize+1):min(i * foldsize, B)],
      ) for i = 1:hparams.k
    ]
  end
end

# ----- Dataset assembly -----------------------------------------------------

function make_dataset(sparams::SampleParams, hparams::HyperParams)
  all_seqs, labels = load_balanced_data(sparams)
  X = onehot_batch(all_seqs)
  Y = onehotbatch(labels, 0:1)
  B = size(X, 3)
  @assert size(Y, 2) == B

  folds = split_indices(B, hparams, sparams)
  datasets = []
  for fold in folds
    train_idx, val_idx = fold.train, fold.val
    Xtrain, Ytrain = X[:, :, train_idx], Y[:, train_idx]
    Xval, Yval = X[:, :, val_idx], Y[:, val_idx]
    push!(datasets, ((Xtrain, Ytrain), (Xval, Yval)))
  end
  return datasets, (B = B, L = size(X, 1))
end

# ----- Model architecture ---------------------------------------------------

pool_out_len(seqlen::Int, p::Int, n::Int) = begin
  L = seqlen
  for _ = 1:n
    L = fld(L, p)
  end
  L
end

maybe_bn(channels, args) = args.batchnorm ? BatchNorm(channels) : identity
maybe_do(p, args) = args.dropout ? Dropout(p) : identity

function buildCNN(args::HyperParams, sparams::SampleParams)
  nblocks = length(args.layerouts)
  @assert length(args.dropouts) == nblocks + 1

  Lout = pool_out_len(sparams.seqlen, args.maxpool, nblocks)
  fin = Lout * args.layerouts[end]

  layers = Any[]

  # first conv block
  push!(
    layers,
    Conv(
      (args.kernelsize,),
      4 => args.layerouts[1],
      pad = SamePad(),
      init = Flux.kaiming_uniform,
    ),
    maybe_bn(args.layerouts[1], args),
    args.σ,
    Conv(
      (args.kernelsize,),
      args.layerouts[1] => args.layerouts[1],
      pad = SamePad(),
      init = Flux.kaiming_uniform,
    ),
    maybe_bn(args.layerouts[1], args),
    args.σ,
    MaxPool((args.maxpool,)),
    maybe_do(args.dropouts[1], args),
  )

  # remaining conv blocks
  for i = 2:nblocks
    push!(
      layers,
      Conv(
        (args.kernelsize,),
        args.layerouts[i-1] => args.layerouts[i],
        pad = SamePad(),
        init = Flux.kaiming_uniform,
      ),
      maybe_bn(args.layerouts[i], args),
      args.σ,
      Conv(
        (args.kernelsize,),
        args.layerouts[i] => args.layerouts[i],
        pad = SamePad(),
        init = Flux.kaiming_uniform,
      ),
      maybe_bn(args.layerouts[i], args),
      args.σ,
      MaxPool((args.maxpool,)),
      maybe_do(args.dropouts[i], args),
    )
  end

  # dense head
  push!(
    layers,
    Flux.flatten,
    Dense(fin, args.layerouts[end], init = Flux.kaiming_uniform),
    maybe_bn(args.layerouts[end], args),
    args.σ,
    maybe_do(args.dropouts[end], args),
    Dense(args.layerouts[end], 2),
    softmax,
  )

  return Chain(layers...) |> args.device
end

# ----- Training loop --------------------------------------------------------

function trainCNN!(
  model,
  train_loader::DataLoader,
  val_loader::Union{DataLoader,Nothing};
  hparams::HyperParams,
)
  opt = Optimisers.OptimiserChain(
    Optimisers.Descent(hparams.η),
    Optimisers.Momentum(hparams.momentum),
  )
  opt_state = Optimisers.setup(opt, model)

  function lossfn(m, xb, yb)
    ce = Flux.crossentropy(m(xb), yb)
    if !isnan(hparams.λ)
      reg = hparams.λ * Zygote.ignore() do
        sum(p -> sum(abs2, p), Flux.params(m))
      end
      return ce + reg
    else
      return ce
    end
  end

  train_losses = Float32[]
  val_losses = Float32[]
  train_accs = Float32[]
  val_accs = Float32[]

  for epoch = 1:hparams.epochs
    eloss = 0.0f0
    ecorr = 0
    ecnt = 0
    for (xb, yb) in train_loader
      xb = hparams.device(xb)
      yb = hparams.device(yb)
      loss_val, grads = Flux.withgradient(m -> lossfn(m, xb, yb), model)
      opt_state, model = Optimisers.update!(opt_state, model, grads[1])
      eloss += loss_val
      ŷ = model(xb)
      ecorr += sum(Flux.onecold(ŷ) .== Flux.onecold(yb))
      ecnt += size(yb, 2)
    end
    push!(train_losses, eloss / max(ecnt, 1))
    push!(train_accs, ecorr / max(ecnt, 1))

    if val_loader !== nothing
      vloss = 0.0f0
      vcorr = 0
      vcnt = 0
      for (xb, yb) in val_loader
        xb = hparams.device(xb)
        yb = hparams.device(yb)
        ŷ = model(xb)
        vloss += Flux.crossentropy(ŷ, yb)
        vcorr += sum(Flux.onecold(ŷ) .== Flux.onecold(yb))
        vcnt += size(yb, 2)
      end
      push!(val_losses, vloss / max(vcnt, 1))
      push!(val_accs, vcorr / max(vcnt, 1))
    else
      push!(val_losses, NaN32)
      push!(val_accs, NaN32)
    end
    @info "epoch $epoch" train_loss = train_losses[end] train_acc = train_accs[end] val_loss =
      val_losses[end] val_acc = val_accs[end]
  end
  return (; train_losses, val_losses, train_accs, val_accs, model, opt_state)
end

# ----- Checkpoint save ------------------------------------------------------

function save_checkpoint(model, hparams, sparams, train_result, path)
  @save path model hparams sparams train_losses = train_result.train_losses val_losses =
    train_result.val_losses train_accs = train_result.train_accs val_accs =
    train_result.val_accs
end

# ----- Full training + saving wrapper (used by flow) ------------------------

function train_and_save(hparams::HyperParams, sparams::SampleParams, out_base::String)
  datasets, meta = make_dataset(sparams, hparams)
  model_paths = String[]
  last_metrics = Dict()
  # strip potential extension to build base name
  base = replace(out_base, r"\.bson$" => "")
  for (i, ((Xtrain, Ytrain), (Xval, Yval))) in enumerate(datasets)
    train_loader =
      DataLoader((Xtrain, Ytrain); batchsize = hparams.batchsize, shuffle = hparams.shuffle)
    val_loader = DataLoader((Xval, Yval); batchsize = hparams.batchsize, shuffle = false)
    model = buildCNN(hparams, sparams)
    result = trainCNN!(model, train_loader, val_loader; hparams = hparams)
    stamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    suffix = length(datasets) > 1 ? "_fold$(i)" : ""
    fname = "$(base)$(suffix)_$(stamp).bson"
    model_cpu = cpu(model)
    save_checkpoint(model_cpu, hparams, sparams, result, fname)
    push!(model_paths, fname)
    last_metrics = Dict(
      "fold" => i,
      "val_loss" => last(result.val_losses),
      "val_acc" => last(result.val_accs),
      "train_loss" => last(result.train_losses),
      "train_acc" => last(result.train_accs),
    )
  end
  return (metrics = last_metrics, model_paths = model_paths)
end

####################################################################################################

end
