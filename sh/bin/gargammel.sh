#!/usr/bin/env bash

# --- conda initialize ---
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/drivas/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/drivas/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/drivas/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/drivas/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# --- end conda initialize ---

# --- variables ---
N=$1   # number of reads (-n)
L=$2   # read length (-l)

# --- change directory ---
cd simul || { echo "Directory 'simul' not found"; exit 1; }

# --- run gargammel ---
OUT="h38_ancient_${N}_${L}"
gargammel --comp 1,0,0 -n "$N" -l "$L" -o "$OUT" -mapdamage config/missincorporation.txt .
