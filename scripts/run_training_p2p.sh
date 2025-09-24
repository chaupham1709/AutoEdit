python run_training.py \
    --checkpoint "exp/ddpm_ppo/checkpoint.ckpt" \
    --input_image_path "assets/cake.jpg" \
    --src_prompt "a round cake with orange frosting on a wooden plate" \
    --tgt_prompt "a square cake with orange frosting on a wooden plate" \
    --save_edit_path "output/edit_cake.png"