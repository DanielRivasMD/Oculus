####################################################################################################

# new FASTA file with candidate sequences
newseqs = load_sequences_fasta("data/new_samples.fa")

# Filter to 50 nt
newseqs_50 = filter(seq -> length(seq) == 50, newseqs)

# One-hot encode
Xnew = onehot_batch(newseqs_50)   # (50, 4, N)

ŷ = model(Xnew)   # (2, N) softmax probabilities

####################################################################################################

using Flux: onecold

pred_labels = onecold(ŷ, 0:1)   # returns vector of 0s and 1s

for (i, seq) in enumerate(newseqs_50)
    println("Sequence $i classified as: ",
            pred_labels[i] == 0 ? "French" : "Neandertal",
            " (probabilities = ", ŷ[:, i], ")")
end

####################################################################################################

function predict_class(seqs::Vector{LongDNA{4}})
    X = onehot_batch(seqs)
    probs = model(X)
    preds = onecold(probs, 0:1)
    return preds, probs
end

####################################################################################################
