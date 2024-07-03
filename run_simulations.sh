#!/bin/bash

# Define the list of values for --num_accusations
num_accusations_list=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

# Loop through each value in the list
for num_accusations in "${num_accusations_list[@]}"
do
    # Execute the Python script with the current value of --num_accusations
    python simulation_metrics.py --epsilon 10 --num_accusations "$num_accusations"
done
