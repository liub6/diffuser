#!/bin/bash

# Viriables
names=("front" "middle" "rear" "extremes")

# names=("front")

train_lengths_list=(
    "0.15 0.25 0.35 0.45"
    "0.25 0.35 0.45 0.55"
    "0.35 0.45 0.55 0.65"
    "0.15 0.25 0.55 0.65"
)
test_lengths_list=(
    "0.5 0.6 0.7"
    "0.1 0.2 0.6 0.7"
    "0.1 0.2 0.3"
    "0.3 0.4 0.5"
)

file=${1:-'td3_bc'}

# test_lengths_list=(
#     "0.15 0.25 0.35 0.45"
#     "0.25 0.35 0.45 0.55"
#     "0.35 0.45 0.55 0.65"
#     "0.15 0.25 0.55 0.65"
# )

# train_lengths_list=(
#     "0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4"
#     "0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6"
#     "0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8"
#     "0.05 0.1 0.15 0.2 0.65 0.7 0.75 0.8"
# )
# test_lengths_list=(
#     "0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8"
#     "0.05 0.1 0.15 0.2 0.65 0.7 0.75 0.8"
#     "0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4"
#     "0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6"
# )

# train_lengths_list=(
#     "0.035 0.045 0.055 0.065 0.075 0.085"
#     "0.055 0.065 0.075 0.085 0.095"
#     "0.065 0.075 0.085 0.095 0.105 0.115"
#     "0.035 0.045 0.055 0.095 0.105 0.115"
# )
# test_lengths_list=(
#     "0.09 0.10 0.11 0.12"
#     "0.03 0.04 0.05 0.10 0.11 0.12"
#     "0.03 0.04 0.05 0.06"
#     "0.06 0.07 0.08 0.09"
# )

# train_lengths_list=(
#     "0.075 0.125 0.175 0.225 0.275"
#     "0.175 0.225 0.275 0.325"
#     "0.225 0.275 0.325 0.375 0.425"
#     "0.075 0.125 0.375 0.425"
# )
# test_lengths_list=(
#     "0.3 0.35 0.4 0.45"
#     "0.05 0.1 0.15 0.35 0.4 0.45"
#     "0.05 0.1 0.15 0.2"
#     "0.15 0.2 0.25 0.3 0.35"
# )

seeds=("0" "1" "2")
# seeds=("0" "1" "2" "3" "4")

# Get the union of all lengths
declare -A train_union_array
declare -A test_union_array
declare -A union_array

for lengths in "${train_lengths_list[@]}"; do
    read -a train_lengths <<< ${lengths}
    for length in "${train_lengths[@]}"; do
        train_union_array["$length"]=1
        union_array["$length"]=1
    done
done

for lengths in "${test_lengths_list[@]}"; do
    read -a test_lengths <<< ${lengths}
    for length in "${test_lengths[@]}"; do
        test_union_array["$length"]=1
        union_array["$length"]=1
    done
done

union_lengths_list=("${!union_array[@]}")
test_union_lengths_list=("${!test_union_array[@]}")
train_union_lengths_list=("${!train_union_array[@]}")

# # Get trainning data
# for length in "${train_union_lengths_list[@]}"; do
#     sbatch ./sac.sh ${length}
#     sleep 0.1
# done

# # Get trainning data
# for length in {1..800}
# do
#     sbatch ./sac.sh ${length}
#     sleep 0.1
# done

# # Combine the training datasets
# declare -a train_dataset_name=()
# for length in "${train_union_lengths_list[@]}"; do
#     for i in {0..5}; do
#         train_dataset_name+="dm-cartpole-test-length${length}-v${i} "
#     done
#     minari combine ${train_dataset_name[@]}--dataset-id=dm-cartpole-train-length${length}-all-v0
#     yes | minari delete ${train_dataset_name[@]}
#     train_dataset_name=()
# done

# Combine the training datasets
# declare -a train_dataset_name=()

# for i in {3..800}; do
#     train_dataset_name+="dm-cartpole-test-length${i}.0-v0 "
#     # yes | minari delete "dm-cartpole-test-length${i}.0-v0"
# done
# minari combine ${train_dataset_name[@]}--dataset-id=dm-cartpole-train-length-all-v0
# # yes | minari delete ${train_dataset_name[@]}
# train_dataset_name=()


# for i in "${!names[@]}"; do
#     name=${names[$i]}
#     train_lengths=${train_lengths_list[$i]}
#     read -a train_lengths <<< ${train_lengths}
#     for length in "${train_lengths[@]}"; do
#         train_dataset_name+="dm-cartpole-train-length${length}-all-v0 "
#     done
#     minari combine ${train_dataset_name[@]}--dataset-id=dm-cartpole-train-length-${name}-v0
#     train_dataset_name=()
# done

# for length in "${train_union_lengths_list[@]}"; do
#     yes | minari delete dm-cartpole-train-length${length}-all-v0
# done


# Train the diffuser
jobids=()
for i in "${!names[@]}"; do
    name=${names[$i]}
    read -a test_lengths <<< "${test_lengths_list[$i]}"

    # Train the diffuser
    # sbatch ./diffusion.sh \
    # 0.1 \                                        #cond for trianing could be any value
    # True \                                       #minari
    # "dm-cartpole-train-length-${name}-v0" \      #dataset
    # "./results-${name}" \                        #results_folder
    # False \                                       #save_samples
    # False                                        #load_checkpoint

    jobid=$(sbatch ./diffusion.sh \
    0.1 \
    1 \
    "dm-cartpole-train-length-all-v0" \
    "./results-${name}" \
    0 \
    0 \
    ${name} | awk '{print $4}')
    jobids+=($jobid)
    sleep 0.1
done
dependency_str="afterok"
for jobid in "${jobids[@]}"; do
    dependency_str="${dependency_str}:${jobid}"
done


# Generate the dataset
jobids=()
for i in "${!names[@]}"; do
    name=${names[$i]}

    for length in "${test_union_lengths_list[@]}"
    do
        jobid=$(sbatch --dependency=$dependency_str ./diffusion.sh \
        ${length} \
        1 \
        "dm-cartpole-train-length-all-v0" \
        "./results-${name}" \
        1 \
        1 \
        ${name} | awk '{print $4}')
        jobids+=($jobid)
        sleep 0.1
    done
done
dependency_str="afterok"
for jobid in "${jobids[@]}"; do
    dependency_str="${dependency_str}:${jobid}"
done


# Test the diffuser
jobids=()
for i in "${!names[@]}"; do
    name=${names[$i]}

    for length in "${test_union_lengths_list[@]}"
    do
        for seed in "${seeds[@]}"
        do
            :
            # Test the diffuser
            # sbatch ./td3_bc.sh \
            # ./corl/yaml/td3_bc/cartpole/cartpole_swingup.yaml \             #config
            # ./corl_logs/ \                                                  #checkpoints_path
            # "${name}-${length}-${seed}" \                                   #name
            # "./results-${name}/5m_samples.npz_${length}.npz" \              #diffusion.path
            # ${length} \                                                     #pole_length
            # "dm-cartpole-train-length-${name}-v0" \                         #dataset
            # False \                                                         #context_aware
            # ${seed}                                                         #seed

            # Diffuser
            sbatch --dependency=$dependency_str ./test_diffuser.sh \
            $file \
            ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
            ./corl_logs/ \
            "diffuser-${name}-${length}-${seed}" \
            "./results-${name}/5m_samples.npz_${length}.npz" \
            ${length} \
            "dm-cartpole-train-length-all-v0" \
            0 \
            ${seed} \
            ${name} 
            sleep 0.1

            # Context aware
            sbatch --dependency=$dependency_str ./test_diffuser.sh \
            $file \
            ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
            ./corl_logs/ \
            "context-aware-${name}-${length}-${seed}" \
            "None" \
            ${length} \
            "dm-cartpole-train-length-all-v0" \
            1 \
            ${seed} \
            ${name} 
            sleep 0.1

            # Diffuser + context aware
            sbatch --dependency=$dependency_str ./test_diffuser.sh \
            $file \
            ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
            ./corl_logs/ \
            "diffuser-context-aware-${name}-${length}-${seed}" \
            "./results-${name}/5m_samples.npz_${length}.npz" \
            ${length} \
            "dm-cartpole-train-length-all-v0" \
            1 \
            ${seed} \
            ${name} 
            sleep 0.1

            # True
            sbatch --dependency=$dependency_str ./test_diffuser.sh \
            $file \
            ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
            ./corl_logs/ \
            "true-${name}-${length}-${seed}" \
            "None" \
            ${length} \
            "dm-cartpole-train-length-all-v0" \
            0 \
            ${seed} \
            ${name} 
            sleep 0.1
        done
    done
done

# ********************************************************************************************************************

# # Test the diffuser
# jobids=()
# for i in "${!names[@]}"; do
#     name=${names[$i]}

#     for length in "${test_union_lengths_list[@]}"
#     do
#         for seed in "${seeds[@]}"
#         do
#             :
#             # Test the diffuser
#             # sbatch ./td3_bc.sh \
#             # ./corl/yaml/td3_bc/cartpole/cartpole_swingup.yaml \             #config
#             # ./corl_logs/ \                                                  #checkpoints_path
#             # "${name}-${length}-${seed}" \                                   #name
#             # "./results-${name}/5m_samples.npz_${length}.npz" \              #diffusion.path
#             # ${length} \                                                     #pole_length
#             # "dm-cartpole-train-length-${name}-v0" \                         #dataset
#             # False \                                                         #context_aware
#             # ${seed}                                                         #seed

#             # Diffuser
#             sbatch ./test_diffuser.sh \
#             $file \
#             ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
#             ./corl_logs/ \
#             "diffuser-${name}-${length}-${seed}" \
#             "./results-${name}/5m_samples.npz_${length}.npz" \
#             ${length} \
#             "dm-cartpole-train-length-all-v0" \
#             0 \
#             ${seed} \
#             ${name}
#             sleep 0.1

#             # Context aware
#             sbatch ./test_diffuser.sh \
#             $file \
#             ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
#             ./corl_logs/ \
#             "context-aware-${name}-${length}-${seed}" \
#             "None" \
#             ${length} \
#             "dm-cartpole-train-length-all-v0" \
#             1 \
#             ${seed} \
#             ${name}
#             sleep 0.1

#             # Diffuser + context aware
#             sbatch ./test_diffuser.sh \
#             $file \
#             ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
#             ./corl_logs/ \
#             "diffuser-context-aware-${name}-${length}-${seed}" \
#             "./results-${name}/5m_samples.npz_${length}.npz" \
#             ${length} \
#             "dm-cartpole-train-length-all-v0" \
#             1 \
#             ${seed} \
#             ${name}
#             sleep 0.1

#             # True
#             sbatch ./test_diffuser.sh \
#             $file \
#             ./corl/yaml/$file/cartpole/cartpole_swingup.yaml \
#             ./corl_logs/ \
#             "true-${name}-${length}-${seed}" \
#             "None" \
#             ${length} \
#             "dm-cartpole-train-length-all-v0" \
#             0 \
#             ${seed} \
#             ${name}
#             sleep 0.1
#         done
#     done
# done


