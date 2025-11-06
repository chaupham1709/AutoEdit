python run_training_wo_attention.py \
    --exp_name "exp/ddpm_ppo_sft" \
    --num_epochs 15 \
    --annotation_folder "data/EditBench/EditData" \
    --train_ppo \
    --checkpoint "path/to/checkpoint_sft"