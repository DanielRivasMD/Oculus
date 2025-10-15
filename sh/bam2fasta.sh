#!/bin/sh
# Convert BAM → FASTA using samtools + seqtk

####################################################################################################

DATA="data"
BAMDIR="${DATA}/bam"
FASTADIR="${DATA}/fasta"

# French
samtools fastq "$BAMDIR/French.bam" \
  | seqtk seq -A \
  > "$FASTADIR/French.fasta"

# Neandertal
samtools fastq "$BAMDIR/Neandertal.bam" \
  | seqtk seq -A \
  > "$FASTADIR/Neandertal.fasta"

echo "Conversion complete"

####################################################################################################
