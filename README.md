# Beyond Action Residuals: Steering Robot Manipulation Policies with Bottleneck Latent Reinforcement Learning (ZPRL)

Dongjie Yu$^{*,1,2}$, Kun Lei$^{*,2,3}$, Zhennan Jiang$^{4}$, Jia Pan$^{\dagger,1}$, Huazhe Xu$^{\dagger,2,5}$

$^*$ Equal contribution
$^{\dagger}$ Corresponding authors

$^1$ School of Computing and Data Science, HKU
$^2$ Shanghai Qi Zhi Institute
$^3$ Shanghai Jiao Tong University
$^4$ Institute of Automation, CAS
$^5$ IIIS, THU

TLDR: ZPRL is an RL finetuning framework that perturbs bottleneck latents to steer robot manipulation policies, achieving efficient steering and smooth robot actions.


## Installation
1. Clone the repository.
```bash
git clone
cd zprl
```

2. Create a virtual environment and install the required dependencies. (We used mamba for fast env management, but you can also use conda.)
```bash
mamba create env -f ./conda_environment.yaml
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

4. We made a small patch to `robosuite` to correct its GPU rendering on the specified device and enable faster parallel simulation. Replace `/your/path/to/dependencies/robosuite/robosuite/renderers/context/egl_context.py` with [this](https://drive.google.com/file/d/1oxenFUt2E1uwEYaltP6bOZGFZgxxHef2/view?usp=sharing) and replace `/your/path/to/dependencies/robosuite/robosuite/utils/binding_utils.py` with [this](https://drive.google.com/file/d/1_AaxnOZVl629tBK-QVNkh9xzT8HObKat/view?usp=sharing). You may need to change the `conversion_map` in `egl_context.py` because the map varies on different computers.


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
Each `.hdf5` randomly samples 100 trajectories from the original Robomimic [MH dataset](https://robomimic.github.io/docs/v0.3/datasets/overview.html), renders the image observation following scripts [here](https://github.com/EricJin2002/SIME/blob/main/simulation/extract_obs_from_raw_datasets.sh), and turns the delta action into an absolute action with [this](diffusion_policy/scripts/robomimic_dataset_conversion.py). But downloading the dataset we uploads can save you all of these steps.

> After downloading, remember to change the `dataset_path` in `zprl/config/task/{task}_image_abs.yaml` to `/your/path/to/the_hdf5` per task.


## Offline Training


## Online RL


## Acknowledgements
Our code base is built on the following repositories and this README borrows a lot from [DICE-RL](https://github.com/real-stanford/dice-rl). We thank the authors for open-sourcing their wonderful codes and clear documentation.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): our offline pipeline generally follows the Diffusion Policy workspace but replaces DDPM/DDIM with a rectified flow to reduce denoising steps during inference.
- [Policy Decorator](https://github.com/tongzhoumu/policy_decorator): our online workspace basically follows what policy decorator does. We make some optimization (such as next observation pre-encoding) to accelerate training.

## Contact
Feel free to contact [Dongjie Yu](mailto:djyu@connect.hku.hk) if you have any questions about the paper or the code base.