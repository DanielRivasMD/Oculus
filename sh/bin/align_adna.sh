#!/bin/sh
# Align ancient DNA reads to a reference genome
# Usage: ./align_adna.sh reference.fasta reads.fasta

if [ $# -lt 2 ]; then
  echo "Usage: $0 <reference.fasta> <reads.fasta>"
  exit 1
fi

REF="$1"
READS="$2"
BASENAME=$(basename "$READS" .fasta)
OUT="${BASENAME}.bam"

# Step 1: Index reference genome (only once per reference)
if [ ! -f "${REF}.bwt" ]; then
  echo "Indexing reference genome..."
  bwa index "$REF"
fi

# Step 2: Align reads with BWA aln (ancient DNA parameters)
echo "Aligning reads..."
bwa aln -l 1024 -n 0.01 -o 2 "$REF" "$READS" > "${BASENAME}.sai"

# Step 3: Generate SAM
bwa samse "$REF" "${BASENAME}.sai" "$READS" > "${BASENAME}.sam"

# Step 4: Convert SAM → BAM, sort, and index
echo "Converting to BAM..."
samtools view -Sb "${BASENAME}.sam" | samtools sort -o "$OUT"
samtools index "$OUT"

# Step 5: Cleanup intermediate files
rm "${BASENAME}.sai" "${BASENAME}.sam"

echo "Alignment complete → $OUT (sorted and indexed)"

