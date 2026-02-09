#!/bin/sh
# Convert BAM → FASTA using samtools + seqtk
# Supports whole-genome or per-chromosome extraction
# Filters out non-canonical contigs (keeps only chr1–chr22, chrX, chrY, chrM)

####################################################################################################

show_help() {
  cat <<EOF
Usage: $0 [--whole | --per-chrom]

Convert BAM files to FASTA using samtools + seqtk.

Options:
  --whole       Convert entire BAM into one FASTA (default)
  --per-chrom   Split BAM by chromosome, appending _chrN to filenames
  -h, --help    Show this help message

This script expects BAMs in:
  data/bam/French.bam
  data/bam/Neandertal.bam

It will produce FASTAs in:
  data/fasta/
EOF
}

####################################################################################################

MODE="whole"

# Argument parsing
if [ $# -gt 0 ]; then
  case "$1" in
    --whole) MODE="whole" ;;
    --per-chrom) MODE="perchrom" ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Error: unexpected argument '$1'"; show_help; exit 1 ;;
  esac
fi

DATA="data"
BAMDIR="${DATA}/bam"
FASTADIR="${DATA}/fasta"

mkdir -p "$FASTADIR"

# for sample in French Neandertal; do
for sample in simulation_ancient simulation_modern; do
  bam="$BAMDIR/$sample.bam"

  if [ "$MODE" = "whole" ]; then
    echo "Converting whole genome for $sample..."
    samtools fastq "$bam" | seqtk seq -A > "$FASTADIR/${sample}.fasta"
  else
    echo "Converting per chromosome for $sample..."
    samtools index "$bam"
    # Only keep canonical chromosomes: chr1–chr22, chrX, chrY, chrM
    for chr in $(samtools idxstats "$bam" | cut -f1 \
                 | grep -E '^chr([0-9]+|X|Y|M)$'); do
      echo "  Extracting $chr..."
      samtools view -b "$bam" "$chr" \
        | samtools fastq - \
        | seqtk seq -A \
        > "$FASTADIR/${sample}_${chr}.fasta"
    done
  fi
done

echo "Conversion complete: FASTA files written to $FASTADIR"

####################################################################################################
