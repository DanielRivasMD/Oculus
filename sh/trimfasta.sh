#!/bin/sh
# Extract 37nt reads from Neandertal & French BAMs

####################################################################################################

DATA="data"
FASTADIR="${DATA}/fasta"

# Neandertal
awk '
  /^>/ {header=$0; next}
  { if (length($0)==37) {print header; print $0} }
' "$FASTADIR/Neandertal.fasta" > "$FASTADIR/Neandertal_37nt.fasta"

# French
awk -v N=2964143 '
  /^>/ {header=$0; getline seq;
        if (count < N) {
            print header
            print substr(seq,1,37)
            count++
        }
  }
' "$FASTADIR/French.fasta" > "$FASTADIR/French_37nt.fasta"

# Sanity check
echo "French_37nt.fasta:     $(grep -c '^>' "$FASTADIR/French_37nt.fasta") reads"
echo "Neandertal_37nt.fasta: $(grep -c '^>' "$FASTADIR/Neandertal_37nt.fasta") reads"

####################################################################################################
