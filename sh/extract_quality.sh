#!/bin/bash
# Usage: samtools view file.bam | ./extract_quality.sh [read_length] > qualities.csv
# Default read length = 76

LEN="${1:-76}"

awk -v L="$LEN" '
BEGIN {
    OFS = "\t";
    # Build lookup table for characters 33 to 73 (Phred 0-40)
    scores["!"]=0; scores["\""]=1; scores["#"]=2; scores["$"]=3; scores["%"]=4;
    scores["&"]=5; scores["'\''"]=6; scores["("]=7; scores[")"]=8; scores["*"]=9;
    scores["+"]=10; scores[","]=11; scores["-"]=12; scores["."]=13; scores["/"]=14;
    scores["0"]=15; scores["1"]=16; scores["2"]=17; scores["3"]=18; scores["4"]=19;
    scores["5"]=20; scores["6"]=21; scores["7"]=22; scores["8"]=23; scores["9"]=24;
    scores[":"]=25; scores[";"]=26; scores["<"]=27; scores["="]=28; scores[">"]=29;
    scores["?"]=30; scores["@"]=31; scores["A"]=32; scores["B"]=33; scores["C"]=34;
    scores["D"]=35; scores["E"]=36; scores["F"]=37; scores["G"]=38; scores["H"]=39;
    scores["I"]=40;
    # Print header
    printf "seq\tqual";
    for (i=1; i<=L; i++) printf "%s%d", OFS, i;
    printf "\n";
}
{
    if (NF < 11) next;
    seq = $10;
    qual = $11;
    if (length(seq) != L) next;
    if (length(qual) != L) next;
    printf "%s\t%s", $10, $11;
    for (i=1; i<=L; i++) {
        c = substr(qual, i, 1);
        q = (c in scores) ? scores[c] : 0;
        printf "%s%d", OFS, q;
    }
    printf "\n";
}' -
