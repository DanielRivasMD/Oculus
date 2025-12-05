# Ns=(10000 20000 30000 40000 50000 100000 200000 300000 400000 500000 1000000 5000000 10000000 50000000 100000000)
Ns=(100000)
L=76

for N in $Ns
do
echo "Running gargammel.sh with N=$N & L=$L"
./sh/bin/gargammel.sh "$N" "$L"
done

for f in simul/h38_ancient_*.b.fa.gz
do
 out="${f%.b.fa.gz}nt.fasta"
 echo "Unzipping $f -> $out"
 gunzip -c "$f" > "$out"
done

rm simul/h38_ancient_*.f[aq].gz

for N in $Ns
do
OUT="simul/h38_modern_${N}_${L}nt.fasta"

echo "Running genomeExtract.jl for N=$N and L=$L -> $OUT"
julia --project src/bin/genomeExtract.jl data/refseq/GCF_000001405.26_GRCh38_genomic.fna "$OUT" "$N" "$L"
done
