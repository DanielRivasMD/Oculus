#!/usr/bin/env bash
# Usage: ./fastq_trimmer.sh INPUT.fastq

# set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 input.fastq"
    exit 1
fi

infile="$1"
# Strip extension and add suffix
outfile="${infile%.fastq}_sample.fastq"

# Take first 100 reads, trim to 50 nt, drop qualities
seqtk seq -A "$infile" | \
    head -n 200 | \
    awk 'NR%2==1{print $0} NR%2==0{print substr($0,1,50)}' \
    > "$outfile"

echo "Wrote sample to $outfile"
