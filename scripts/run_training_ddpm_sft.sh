python run_training_wo_attention.py \
    --exp_name "exp/ddpm_ppo_sft" \
    --num_epochs 5 \
    --low 0.05 \
    --range 0.6 \
    --annotation_folder "data/EditBench/EditData" \
    --train_sft