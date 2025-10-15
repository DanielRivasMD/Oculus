####################################################################################################
# Inference script
####################################################################################################

# Load candidate sequences from a new FASTA file
newseqs = load_sequences_fasta("")

# One-hot encode into tensor (maxL, 4, N)
# - onehot_batch automatically right-pads shorter sequences
# - maxL = length of the longest sequence in this batch
Xnew = onehot_batch(newseqs)

# Run model forward pass
# - Because the model ends with `softmax`, this returns probabilities
#   of shape (2, N), where each column is [p(class=0), p(class=1)]
ŷ = model(Xnew)

####################################################################################################
# Helper function for reuse
####################################################################################################

"""
    predict_class(seqs::Vector{LongDNA{4}}) -> (preds, probs)

Run inference on a batch of DNA sequences.

# Arguments
- `seqs` : Vector of DNA sequences (`LongDNA{4}`), any length.

# Returns
- `preds::Vector{Int}` : Hard class labels (0 = French, 1 = Neandertal).
- `probs::Array{Float32,2}` : Softmax probabilities of shape (2, N).

# Notes
- Use `probs` if you need confidence scores, ROC curves, or calibration.
- Use `preds` if you only need discrete class assignments.
- Sequences are automatically padded to the longest sequence in the batch.
"""
function predict_class(seqs::Vector{LongDNA{4}})
    X = onehot_batch(seqs)
    probs = model(X)              # (2, N) probabilities
    preds = onecold(probs, 0:1)   # vector of 0s and 1s
    return preds, probs
end

####################################################################################################
# Run inference using helper
####################################################################################################

pred_labels, probs = predict_class(newseqs)

for (i, seq) in enumerate(newseqs)
    println("Sequence $i classified as: ",
            pred_labels[i] == 0 ? "French" : "Neandertal",
            " (probabilities = ", probs[:, i], ")")
end

####################################################################################################
