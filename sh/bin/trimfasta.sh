#!/bin/sh
# Extract fixed-length reads from Neandertal & French FASTA files
# Compatible with whole-genome or per-chromosome FASTA outputs
# Computes N per-pair from the matching Neandertal file

####################################################################################################

show_help() {
  cat <<EOF
Usage: $0 <read_length> [--whole | --per-chrom]

Extract fixed-length reads from Neandertal & French FASTA files.

Arguments:
  <read_length>   Desired sequence length (e.g. 37, 50, 128)

Options:
  --whole         Process whole-genome FASTA files (default)
  --per-chrom     Process per-chromosome FASTA files (expects *_chrN.fasta)
EOF
}

####################################################################################################

if [ $# -lt 1 ]; then
  show_help; exit 1
fi

LEN="$1"
MODE="whole"
if [ $# -gt 1 ]; then
  case "$2" in
    --whole) MODE="whole" ;;
    --per-chrom) MODE="perchrom" ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Error: unexpected option '$2'"; show_help; exit 1 ;;
  esac
fi

DATA="data"
FASTADIR="${DATA}/fasta"

# Count how many reads in FASTA have exactly length LEN
count_len() {
  fasta="$1"
  seqtk seq -l0 "$fasta" \
    | awk '!/^>/ {print length($0)}' \
    | awk -v L="$LEN" '$1==L' \
    | wc -l
}

process_neandertal() {
  infile="$1"; outfile="$2"
  awk -v L="$LEN" '
    /^>/ {header=$0; next}
    { if (length($0)==L) {print header; print $0} }
  ' "$infile" > "$outfile"
  echo "$outfile: $(grep -c '^>' "$outfile") reads"
}

process_french() {
  infile="$1"; outfile="$2"; N="$3"
  awk -v L="$LEN" -v N="$N" '
    /^>/ {header=$0; getline seq;
          if (count < N) {
              print header
              print substr(seq,1,L)
              count++
          }
    }
  ' "$infile" > "$outfile"
  echo "$outfile: $(grep -c '^>' "$outfile") reads"
}

if [ "$MODE" = "whole" ]; then
  N=$(count_len "$FASTADIR/Neandertal.fasta")
  echo "Detected $N Neandertal reads of length $LEN (whole-genome)"

  process_neandertal "$FASTADIR/Neandertal.fasta" "$FASTADIR/Neandertal_${LEN}nt.fasta"
  process_french     "$FASTADIR/French.fasta"     "$FASTADIR/French_${LEN}nt.fasta" "$N"

else
  for nf in "$FASTADIR"/Neandertal_chr*.fasta; do
    [ -e "$nf" ] || continue
    chr=$(basename "$nf" .fasta | cut -d_ -f2)
    ff="$FASTADIR/French_${chr}.fasta"
    [ -f "$ff" ] || { echo "Missing French counterpart for $chr"; continue; }

    N=$(count_len "$nf")
    echo "Detected $N Neandertal reads of length $LEN in $nf"

    process_neandertal "$nf" "$FASTADIR/Neandertal_${chr}_${LEN}nt.fasta"
    process_french     "$ff" "$FASTADIR/French_${chr}_${LEN}nt.fasta" "$N"
  done
fi

echo "Extraction complete"

####################################################################################################
