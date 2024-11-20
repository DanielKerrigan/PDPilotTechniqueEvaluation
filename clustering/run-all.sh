printf "combine ice plots into one file\n"
python combine-ice.py

printf "\ngenerate synthetic ice plots\n"
python generate-ice-plots.py

printf "\ncluster ice plots using diff\n"
python cluster-ice-plots.py -m diff

printf "\ncluster ice plots using cice\n"
python cluster-ice-plots.py -m cice

printf "\ncluster ice plots using mean\n"
python cluster-ice-plots.py -m mean

printf "\ncluster ice plots using none\n"
python cluster-ice-plots.py -m none

printf "\nscore methods\n"
python score-methods.py
