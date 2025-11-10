#!/bin/sh
# Compute read length distribution from a FASTA file

####################################################################################################
# Usage:
#   ./read_len_dist.sh <fasta_file> [--csv output.csv]
#
# Example:
#   ./read_len_dist.sh data/fasta/French.fasta
#   ./read_len_dist.sh data/fasta/French.fasta --csv lengths.csv
####################################################################################################

show_help() {
  cat <<EOF
Usage: $0 <fasta_file> [--csv output.csv]

Compute read length distribution from a FASTA file.

Arguments:
  <fasta_file>   Input FASTA file

Options:
  --csv FILE     Save distribution to CSV file (columns: length,count)
  -h, --help     Show this help message
EOF
}

if [ $# -lt 1 ]; then
  show_help
  exit 1
fi

FASTA="$1"
CSV=""

if [ $# -gt 1 ]; then
  case "$2" in
    --csv)
      if [ $# -lt 3 ]; then
        echo "Error: --csv requires a filename"
        exit 1
      fi
      CSV="$3"
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: unexpected option '$2'"
      show_help
      exit 1
      ;;
  esac
fi

# Compute distribution: lengths and counts
DIST=$(awk '
  /^>/ {next} {print length($0)}
' "$FASTA" | sort -n | uniq -c)

echo "Read length distribution for $FASTA:"
echo "$DIST" | awk '{printf "Length %s: %s reads\n", $2, $1}'

# Save to CSV if requested
if [ -n "$CSV" ]; then
  echo "length,count" > "$CSV"
  echo "$DIST" | awk '{printf "%s,%s\n", $2, $1}' >> "$CSV"
  echo "Distribution saved to $CSV"
fi

####################################################################################################
