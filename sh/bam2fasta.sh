#!/bin/sh
# Convert BAM → FASTA using samtools + seqtk

####################################################################################################
# Usage: ./bam2fasta.sh
####################################################################################################

show_help() {
  cat <<EOF
Usage: $0

Convert BAM files to FASTA using samtools + seqtk.

Options:
  -h, --help   Show this help message and exit

This script expects:
  data/bam/French.bam
  data/bam/Neandertal.bam

It will produce:
  data/fasta/French.fasta
  data/fasta/Neandertal.fasta
EOF
}

# Argument checks
if [ $# -gt 0 ]; then
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: unexpected argument '$1'"
      show_help
      exit 1
      ;;
  esac
fi

DATA="data"
BAMDIR="${DATA}/bam"
FASTADIR="${DATA}/fasta"

# Ensure output directory exists
mkdir -p "$FASTADIR"

# French
samtools fastq "$BAMDIR/French.bam" \
  | seqtk seq -A \
  > "$FASTADIR/French.fasta"

# Neandertal
samtools fastq "$BAMDIR/Neandertal.bam" \
  | seqtk seq -A \
  > "$FASTADIR/Neandertal.fasta"

echo "Conversion complete: FASTA files written to $FASTADIR"

####################################################################################################
