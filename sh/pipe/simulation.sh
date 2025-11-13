for N in 10000 20000 30000 40000 50000 100000 200000 300000 400000 500000
do
 echo "Running gargammel.sh with N=$N and L=75"
 ./sh/gargammel.sh "$N" 75
done

rm simul/h38_ancient_*~simul/h38_ancient_*b.fa.gz

for f in simul/h38_ancient_*_75.b.fa.gz
do

 out="${f%.b.fa.gz}.fasta"

 echo "Unzipping $f -> $out"
 gunzip -c "$f" > "$out"
done

rm simul/h38_ancient_*_75.b.fa.gz
