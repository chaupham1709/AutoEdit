# [NeurIPS 25] Official implementation of AutoEdit: Automatic Hyperparameter Tuning for Image Editing

This is the official implementation of AutoEdit: Automatic Hyperparameter Tuning for Image Editing (NeurIPS 2025). The paper can be found [here](https://arxiv.org/abs/2509.15031):

![Automatic tuning the hyperparameters with AutoEdit](figures/image.png)

### To-do list

- [ ] Release inference code.
- [ ] Release training code.

## Installation

For editing methods running with SD 1.4, please refer to the environment installation in ``python_env/environment.yml``:

```bash
conda env export -n p2p > python_env/environment.yml
```

or you can install the environment by pip on your own. Note the version of some important libraries:

```
accelerate==1.4.0
diffusers==0.12.0
torch==2.1.2
transformers==4.37.2
```

Note that the version of diffusers is low with the SD 1.4 model, I will try to update the code to make it compatible with recent diffusers version.

## Checkpoint

Checkpoint path can be downloaded [here](https://drive.google.com/drive/folders/1uA2EZ2RogMFQFzzIVf2MPusmTKAAwaF0?usp=sharing). Each folder will be the checkpoint of AutoEdit with the corresponding method (see the Running AutoEdit section).

## Running AutoEdit

You can run AutoEdit by following these scripts:

DDPM-Inversion: Search for inversion timestep:

```bash
python run_training_wo_attention.py \
    --checkpoint "exp/ddpm_ppo/checkpoint.ckpt" \
    --input_image_path "assets/cake.jpg" \
    --src_prompt "a round cake with orange frosting on a wooden plate" \
    --tgt_prompt "a square cake with orange frosting on a wooden plate" \
    --save_edit_path "output/edit_cake.png"
```

## Citation

If you find our work interesting, please considering cite:

```
@article{pham2025autoedit,
  title={AutoEdit: Automatic Hyperparameter Tuning for Image Editing},
  author={Pham, Chau and Dao, Quan and Bhosale, Mahesh and Tian, Yunjie and Metaxas, Dimitris and Doermann, David},
  journal={arXiv preprint arXiv:2509.15031},
  year={2025}
}
```