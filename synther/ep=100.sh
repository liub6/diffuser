num_episodes=10
dataset=train_dataset.npz
names=("front")

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
    ${dataset} \
    "./results-${name}_episodes=${num_episodes}" \
    0 \
    0 \
    ${name} | awk '{print $4}')
    jobids+=($jobid)
    sleep 0.1
done
echo "Training diffuser jobs: ${jobids[@]}"
dependency_str="afterok"
for jobid in "${jobids[@]}"; do
    dependency_str="${dependency_str}:${jobid}"
done