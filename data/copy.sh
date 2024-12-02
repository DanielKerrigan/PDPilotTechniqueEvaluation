# Usage:
# ./copy.sh [small|big]
rsync -av kerrigan.d@xfer.discovery.neu.edu:/work/vis/users/kerrigan.d/PDPilot/PDPilotTechniqueEvaluation/data/results/"$1" ./results
