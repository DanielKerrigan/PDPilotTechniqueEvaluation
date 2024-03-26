for i in $(seq 0 17);
do
		python get_clusters.py -d -i $i -p ../data/results/debug -o results/debug
done

