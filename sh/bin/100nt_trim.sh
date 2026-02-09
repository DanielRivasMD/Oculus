#!/bin/sh
awk '
  /^>/ {
    if (seq && length(seq) == 100) {
      print header
      print seq
    }
    header = $0
    seq = ""
    next
  }
  {
    seq = seq $0
  }
  END {
    if (seq && length(seq) == 100) {
      print header
      print seq
    }
  }
' "$1"
