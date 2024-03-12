# ./copy.sh [debug|actual]
rsync -av kerrigan.d@xfer.discovery.neu.edu:/scratch/kerrigan.d/pdpilot/"$1" ./results
