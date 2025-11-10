#!/bin/sh
# Extract fixed-length reads from Neandertal & French FASTA files
# Compatible with whole-genome or per-chromosome FASTA outputs

####################################################################################################
# Usage:
#   ./extract_reads.sh <LENGTH> [--whole | --per-chrom]
#
# Example:
#   ./extract_reads.sh 37 --whole
#   ./extract_reads.sh 50 --per-chrom
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
  -h, --help      Show this help message

Examples:
  $0 37 --whole
  $0 50 --per-chrom
EOF
}

if [ $# -lt 1 ]; then
  show_help
  exit 1
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

process_pair() {
  sample="$1"
  infile="$2"
  outfile="$3"

  # Determine N = number of Neandertal reads of length LEN
  if [ "$sample" = "Neandertal" ]; then
    N=$(seqtk seq -l0 "$infile" \
        | awk '!/^>/ {print length($0)}' \
        | sort -n | uniq -c \
        | awk -v L="$LEN" '$2==L {print $1}')
    echo "Detected $N Neandertal reads of length $LEN in $infile"
  fi

  if [ "$sample" = "Neandertal" ]; then
    # Keep only reads of exactly LEN nt
    awk -v L="$LEN" '
      /^>/ {header=$0; next}
      { if (length($0)==L) {print header; print $0} }
    ' "$infile" > "$outfile"
  else
    # Truncate to LEN nt, downsample to N reads
    awk -v L="$LEN" -v N="$N" '
      /^>/ {header=$0; getline seq;
            if (count < N) {
                print header
                print substr(seq,1,L)
                count++
            }
      }
    ' "$infile" > "$outfile"
  fi

  echo "$outfile: $(grep -c '^>' "$outfile") reads"
}

if [ "$MODE" = "whole" ]; then
  process_pair Neandertal "$FASTADIR/Neandertal.fasta" "$FASTADIR/Neandertal_${LEN}nt.fasta"
  process_pair French     "$FASTADIR/French.fasta"     "$FASTADIR/French_${LEN}nt.fasta"
else
  # Loop over per-chromosome FASTAs
  for f in "$FASTADIR"/Neandertal_chr*.fasta; do
    chr=$(basename "$f" .fasta | cut -d_ -f2)
    process_pair Neandertal "$f" "$FASTADIR/Neandertal_${chr}_${LEN}nt.fasta"
  done
  for f in "$FASTADIR"/French_chr*.fasta; do
    chr=$(basename "$f" .fasta | cut -d_ -f2)
    process_pair French "$f" "$FASTADIR/French_${chr}_${LEN}nt.fasta"
  done
fi

echo "Extraction complete"

####################################################################################################
