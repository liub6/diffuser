#!/bin/bash

train_lengths_list=("$@")
names=("front" "middle" "rear" "extremes")

# Get the union of all lengths
declare -A train_union_array

for lengths in "${train_lengths_list[@]}"; do
    read -a train_lengths <<< ${lengths}
    for length in "${train_lengths[@]}"; do
        train_union_array["$length"]=1
    done
done

train_union_lengths_list=("${!train_union_array[@]}")

# Combine the training datasets
declare -a train_dataset_name=()



for length in "${train_union_lengths_list[@]}"; do
    for i in {0..5}; do
        train_dataset_name+="dm-cartpole-test-length${length}-v${i} "
    done
    minari combine ${train_dataset_name[@]}--dataset-id=dm-cartpole-train-length${length}-all-v0
    yes | minari delete ${train_dataset_name[@]}
    train_dataset_name=()
done

for i in "${!names[@]}"; do
    name=${names[$i]}
    train_lengths=${train_lengths_list[$i]}
    read -a train_lengths <<< ${train_lengths}
    for length in "${train_lengths[@]}"; do
        train_dataset_name+="dm-cartpole-train-length${length}-all-v0 "
    done
    minari combine ${train_dataset_name[@]}--dataset-id=dm-cartpole-train-length-${name}-v0
    train_dataset_name=()
done

for length in "${train_union_lengths_list[@]}"; do
    yes | minari delete dm-cartpole-train-length${length}-all-v0
done