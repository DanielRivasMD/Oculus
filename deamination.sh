#!/usr/bin/env bash
set -euo pipefail

MODERN="data/cache/fasta/simulation_modern_100nt.fasta"
ANCIENT="data/cache/fasta/simulation_ancient_100nt.fasta"
OUT_DIR="data/cache"
GRAPH_DIR="data/graph/png"

mkdir -p "$OUT_DIR/fasta" "$GRAPH_DIR"

total=$(grep -c '^>' "$MODERN")
echo "Total sequences: $total"

sizes=()

for ((size = 400; size <= 3600 && size <= total; size += 400)); do
    sizes+=($size)
done

for ((size = 4000; size <= total; size += 4000)); do
  sizes+=($size)
done

subsample_fasta() {
  local infile=$1
  local outfile=$2
  local n=$3
  local seed=42

  awk -v n="$n" -v seed="$seed" '
    BEGIN {
        srand(seed);
        record = "";
        in_record = 0;
    }
    /^>/ {
        if (in_record) {
            records[++total] = record;
        }
        record = $0 "\n";
        in_record = 1;
        next;
    }
    {
        if (in_record) record = record $0 "\n";
    }
    END {
        if (in_record) records[++total] = record;
        if (n > total) n = total;
        for (i = 1; i <= total; i++) idx[i] = i;
        for (i = total; i > 1; i--) {
            j = int(rand() * i) + 1;
            tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        for (i = 1; i <= n; i++) {
            printf "%s", records[idx[i]];
        }
    }' "$infile" >"$outfile"
}

for n in "${sizes[@]}"; do
  echo "=== Subsample size: $n ==="

  modern_sub="$OUT_DIR/fasta/modern_sub_${n}.fasta"
  ancient_sub="$OUT_DIR/fasta/ancient_sub_${n}.fasta"

  echo "  Subsampling modern..."
  subsample_fasta "$MODERN" "$modern_sub" "$n"
  echo "  Subsampling ancient..."
  subsample_fasta "$ANCIENT" "$ancient_sub" "$n"

  echo "  Running deamination analysis..."
  julia --project bin/deamination.jl \
    --ancient "$ancient_sub" \
    --modern "$modern_sub" \
    --csv "$OUT_DIR/fasta/deamination_${n}.csv" \
    --png "$GRAPH_DIR/deamination_${n}.png"

  echo "  Done for size $n."
done

echo "All subsample analyses finished."
