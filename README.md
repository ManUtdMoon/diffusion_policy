# Beyond Action Residuals: Steering Robot Manipulation Policies with Bottleneck Latent Reinforcement Learning (ZPRL)

Dongjie Yu<sup>&#42;,1,2</sup>,
Kun Lei<sup>&#42;,2,3</sup>,
Zhennan Jiang<sup>4</sup>,
Jia Pan<sup>&#35;,1</sup>,
Huazhe Xu<sup>&#35;,2,5</sup>

<sup>&#42;</sup> Equal contribution
<sup>&#35;</sup> Corresponding authors

<sup>1</sup> School of Computing and Data Science, HKU;
<sup>2</sup> Shanghai Qi Zhi Institute;
<sup>3</sup> Shanghai Jiao Tong University;
<sup>4</sup> Institute of Automation, CAS;
<sup>5</sup> IIIS, THU

TLDR: ZPRL is an RL finetuning framework that perturbs bottleneck latents to steer robot manipulation policies, achieving efficient steering and smooth robot actions.


## Installation
1. Clone the repository.
```bash
git clone
cd zprl
```

2. Create a virtual environment and install the required dependencies. (We used mamba for fast env management, but you can also use conda.)
```bash
mamba env create -f ./conda_environment.yaml
mamba activate zprl
mamba install -c conda-forge mesalib glew glfw # necessary for GPU rendering
```
Then add `export MUJOCO_GL=egl` to your `~/.bashrc`.

3. Install `robosuite` and `robomimic`.
```bash
mamba activate zprl
cd /your/path/to/dependencies/
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.4.1 # version matters to reproduce the results
pip install -e .

cd /your/path/to/dependencies/
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout 9273f9cc # commit matters to reproduce the results
pip install -e .

# some pre-setup to reduce warnings
python /your/path/to/dependencies/robomimic/robomimic/scripts/setup_macros.py
```

4. We made a small patch to `robosuite` to correct its GPU rendering on the specified device and enable faster parallel simulation. Replace `/your/path/to/dependencies/robosuite/robosuite/renderers/context/egl_context.py` with [this](patches/egl_context.py) and replace `/your/path/to/dependencies/robosuite/robosuite/utils/binding_utils.py` with [this](patches/binding_utils.py). You may need to change the `conversion_map` in `egl_context.py` because the map varies on different computers. (Credit to [Baiye Cheng](https://scholar.google.com/citations?user=YOvZXGAAAAAJ))


## Downloading datasets
We have uploaded the datasets for training robomimic tasks (can, square, transport) [here](https://huggingface.co/datasets/ManUtdMoon/robomimicv030). You can download the directory, put it anywhere you like and organize it as follows:
```bash
robomimicv030
├── can
│   └── mh
│       └── image_v141_subset_abs.hdf5
├── square
│   └── mh
│       └── image_v141_subset_abs.hdf5
└── transport
    └── mh
        └── image_v141_subset_abs.hdf5
```
Each `.hdf5` randomly samples 100 trajectories from the original Robomimic [MH dataset](https://robomimic.github.io/docs/v0.3/datasets/overview.html), renders the image observation following scripts [here](https://github.com/EricJin2002/SIME/blob/main/simulation/extract_obs_from_raw_datasets.sh), and turns the delta action into an absolute action with [this](zprl/scripts/robomimic_dataset_conversion.py). But downloading the dataset we uploads can save you all of these steps.

> After downloading, remember to change the `dataset_path` in `zprl/config/task/{task}_image_abs.yaml` to `/your/path/to/the_hdf5` per task.


## Offline Training
1. Activate the environment and login to W&B to track experiments if you have never done before.
```bash
mamba activate zprl
wandb login
```
2. Launch training on cuda:0 with seed 10.
```bash
export MUJOCO_EGL_DEVICE_ID=0 # make robomimic envs run on cuda:0
python train.py \
    --config-name=train_flow_match_vib_unet_image_workspace \
    training.seed=10 \
    task.dataset.seed=10 \
    exp_name=unet_vib_default \
    training.device=cuda:0 \
    task=square_image_abs \
    policy.vib_latent_dim=16 \
    policy.vib_beta=0.0002 \
    policy.vib_recon=0.01
```
> We only change `vib_latent_dim`, `vib_beta`, `vib_recon` during offline training. Specifically, we set `(16, 0.0002, 0.01)` for can and square and `(32, 0.0001, 0.005)` for transport. You can also try other values to see how they affect the performance.

After the training finishes, you will get a directory containing the results like this.
```bash
data/outputs/yyyy.mm.dd/hh.mm.ss_train_flow_match_vib_unet_image_square_image
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── checkpoints
│   ├── epoch_0000-score_0.000.ckpt
│   └── latest.ckpt
├── logs.json.txt
├── media
│   ├── ...
│   └── train_1.mp4
└── train.log
```

You can evaluate the checkpoint again with specified action chunk length (here we use 4) by
```bash
python eval_base.py \
    -c data/outputs/yyyy.mm.dd/hh.mm.ss_train_flow_match_vib_unet_image_square_image/checkpoints/latest.ckpt\
    -o data/eval/square/ \
    -t 4 \
    -d cuda:0
```

Then you will get a directory containing the results on 50 rollouts. You can edit `eval_base.py` to change configurations of evaluation.
```bash
data/eval/square
├── eval_log.json
└── media
    ├── test_100000.mp4
    ├── ...
    └── test_100009.mp4
```

The offline stack generally follows [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and the authors have built an incredible code base and tutorial. Remember to check it out if you are interested.


## Online RL
After the offline training, you can try ZPRL from a given base policy with
```bash
python train.py \                                          
    --config-name=train_online_vib_robomimic_workspace \     
    training.seed=10 \
    exp_name=zprl_default \
    training.device=cuda:0 \
    online_task=square_image_abs \
    online_task.base_ckpt=data/outputs/yyyy.mm.dd/hh.mm.ss_train_flow_match_vib_unet_image_square_image/checkpoints/latest.ckpt
```
Note that there will be 45 parallel environments running (20 for training and 25 for evaluation) and it will not cause OOM on RTX 4090. The structure of the resulting RL training directory under `data/outputs/` is similar as that of offline IL.

> An important hyperparameter in ZPRL is the scale of z-perturbation ($\lambda$ in our paper). We recommend starting from let $\mathrm{RMS}(\lambda\Delta z) \approx 0.15 \mathrm{RMS}(z)$. We track these values during online RL for tuning reference.

> When you summarize the RL results, remember to multiply the steps in w&b by `n_action_steps` (the length of action chunk) such that the actual environment steps are counted. 

## Acknowledgements
Our code base is built on the following repositories and the structure of this README borrows a lot from [DICE-RL](https://github.com/real-stanford/dice-rl). We thank the authors for open-sourcing their wonderful codes and clear documentation.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): our offline pipeline generally follows the Diffusion Policy workspace but replaces DDPM/DDIM with a rectified flow to reduce denoising steps during inference.
- [Policy Decorator](https://github.com/tongzhoumu/policy_decorator): our online workspace basically follows what policy decorator does. We make some optimization (such as next observation pre-encoding) to accelerate training.

## Contact
Feel free to contact [Dongjie Yu](mailto:djyu@connect.hku.hk) if you have any questions about the paper or the code base.