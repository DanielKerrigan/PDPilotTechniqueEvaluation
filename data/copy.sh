# Usage:
# ./copy.sh [small|big]
rsync -av kerrigan.d@xfer.discovery.neu.edu:/scratch/kerrigan.d/pdpilot/"$1" ./results
