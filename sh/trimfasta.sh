#!/bin/sh
# Extract fixed-length reads from Neandertal & French FASTA files

####################################################################################################
# Usage: ./extract_reads.sh <LENGTH>
####################################################################################################

show_help() {
  cat <<EOF
Usage: $0 <read_length>

Extract fixed-length reads from Neandertal & French FASTA files.

Arguments:
  <read_length>   Desired sequence length (e.g. 37, 50, 128)

Options:
  -h, --help      Show this help message and exit

Example:
  $0 37
EOF
}

# Argument checks
if [ $# -ne 1 ]; then
  show_help
  # exit 1
fi

case "$1" in
  -h|--help)
    show_help
    # exit 0
    ;;
esac

LEN="$1"

DATA="data"
FASTADIR="${DATA}/fasta"

# Determine N = number of Neandertal reads of length LEN
N=$(seqtk seq -l0 "$FASTADIR/Neandertal.fasta" \
    | awk '!/^>/ {print length($0)}' \
    | sort -n | uniq -c \
    | awk -v L="$LEN" '$2==L {print $1}')

echo "Detected $N Neandertal reads of length $LEN"

# Neandertal: keep only reads of exactly LEN nt
awk -v L="$LEN" '
  /^>/ {header=$0; next}
  { if (length($0)==L) {print header; print $0} }
' "$FASTADIR/Neandertal.fasta" > "$FASTADIR/Neandertal_${LEN}nt.fasta"

# French: truncate to LEN nt, downsample to N reads
awk -v L="$LEN" -v N="$N" '
  /^>/ {header=$0; getline seq;
        if (count < N) {
            print header
            print substr(seq,1,L)
            count++
        }
  }
' "$FASTADIR/French.fasta" > "$FASTADIR/French_${LEN}nt.fasta"

# Sanity check
echo "French_${LEN}nt.fasta:     $(grep -c '^>' "$FASTADIR/French_${LEN}nt.fasta") reads"
echo "Neandertal_${LEN}nt.fasta: $(grep -c '^>' "$FASTADIR/Neandertal_${LEN}nt.fasta") reads"

####################################################################################################
