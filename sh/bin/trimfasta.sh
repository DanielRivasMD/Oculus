#!/bin/sh
# Extract fixed-length reads from Neandertal & French FASTA files
# Adds optional padding logic

# BUG: trimming for modern keeps on the identified read lenght, not the cummulative after padding

####################################################################################################

show_help() {
  cat <<EOF
Usage: $0 <read_length> [--whole | --per-chrom] [--padding] [--minpad <value>]

Extract fixed-length reads from Neandertal & French FASTA files.

Arguments:
  <read_length>   Desired sequence length (e.g. 37, 50, 128)

Options:
  --whole         Process whole-genome FASTA files (default)
  --per-chrom     Process per-chromosome FASTA files (expects *_chrN.fasta)
  --padding       Pad shorter reads with 'N' in the middle
  --minpad <val>  Minimal length of reads eligible for padding (implies --padding)
EOF
}

####################################################################################################

if [ $# -lt 1 ]; then
  show_help; exit 1
fi

LEN="$1"
MODE="whole"
PADDING="false"
MINPAD=0

shift
while [ $# -gt 0 ]; do
  case "$1" in
    --whole) MODE="whole" ;;
    --per-chrom) MODE="perchrom" ;;
    --padding) PADDING="true" ;;
    --minpad) MINPAD="$2"; PADDING="true"; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Error: unexpected option '$1'"; show_help; exit 1 ;;
  esac
  shift
done

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
  awk -v L="$LEN" -v pad="$PADDING" -v minpad="$MINPAD" '
    /^>/ {header=$0; next}
    {
      if (length($0)==L) {
        print header; print $0
      } else if (pad=="true" && length($0) < L && length($0) >= minpad) {
        padlen = L - length($0)
        half = int(length($0)/2)
        if (length($0) % 2 == 1) {
          left = substr($0,1,half)
          right = substr($0,half+1)
        } else {
          left = substr($0,1,half)
          right = substr($0,half+1)
        }
        padded = left sprintf("%"padlen"s","") right
        gsub(/ /,"N",padded)
        print header; print padded
      }
    }
  ' "$infile" > "$outfile"
  echo "$outfile: $(grep -c '^>' "$outfile") reads"
}

process_french() {
  infile="$1"; outfile="$2"; N="$3"
  awk -v L="$LEN" -v N="$N" -v pad="$PADDING" -v minpad="$MINPAD" '
    /^>/ {header=$0; getline seq;
          if (count < N) {
            if (length(seq) >= L) {
              print header
              print substr(seq,1,L)
              count++
            } else if (pad=="true" && length(seq) < L && length(seq) >= minpad) {
              padlen = L - length(seq)
              half = int(length(seq)/2)
              if (length(seq) % 2 == 1) {
                left = substr(seq,1,half)
                right = substr(seq,half+1)
              } else {
                left = substr(seq,1,half)
                right = substr(seq,half+1)
              }
              padded = left sprintf("%"padlen"s","") right
              gsub(/ /,"N",padded)
              print header; print padded
              count++
            }
          }
    }
  ' "$infile" > "$outfile"
  echo "$outfile: $(grep -c '^>' "$outfile") reads"
}

####################################################################################################

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
