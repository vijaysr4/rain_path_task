## RainPath Task - Virtual Staining
This project is a DDIM implementation for **virtual histological staining**, transforming standard brightfield images into Hematoxylin & Eosin (H\&E)–stained outputs using a Gaussian diffusion framework and a U-Net backbone.

### Task Description

* **Virtual Staining**: Replace chemical H\&E staining with a computational approach, generating realistic images from stained tissue scans.

### Challenges 
Compute power: Trained the entire model in `NVIDIA RTX A5000` with `23 GB VRAM`. So as a result had to simplify the UNet, IO resolution `512 -> 256`, batch size etc

### Architecture Overview

1. **U-Net Model** (`model.py`)

   * Encoder–decoder architecture with skip connections.
   * `in_channels=3`, initial `model_channels=32`, two down/up-sampling levels (`channel_mult=(1,2)`).
   * No attention modules for memory efficiency.

2. **Gaussian Diffusion** (`gaussion_d.py`)

   * Forward process: progressively add noise over `T` timesteps (`q_sample`).
   * Reverse process: denoise using learned U-Net to predict noise residuals (`p_sample`).
   * Supports both ancestral sampling and deterministic DDIM for sharper results.

3. **Inference Pipeline**

   * **Image-to-Image**: Inject noise at timestep `t_enc` into a real input, then reverse-diffuse to produce a stylized H\&E image.
   * **Unconditional Sampling**: Generate pure H\&E samples from random noise via ancestral or DDIM sampling.



## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Requirements

See [requirements.txt](./requirements.txt) for a full list of dependencies.

## Directory Structure

```
├── .gitignore
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── data.py               # Data loading and preprocessing
├── gaussion_d.py         # Gaussian diffusion model definitions
├── model.py              # Model architecture (Unet)
├── main.py               # Entry-point for training 
├── inference_DDIM.py     # Inference with DDIM sampling
├── inference_without_input.py # Raw Inference generation 
├── jpg_to_tiff.py        # Image format conversion utility
├── utils.py              # Helper functions 
├── view_patch.py         # View of .tiff patches
├── io_imgs/              # Input images directory
├── he_styled_results/    # Output styled results directory
└── model/                # Saved model checkpoints
```

## Usage

To run training or evaluation:

```bash
python main.py \
  --data_root . \
  --batch_size 2 \
  --crop_size 256 \
  --lr 1e-4 \
  --timesteps 1000 \
  --epochs 20 \
  --device cuda

```

To perform DDIM inference:

```bash
python inference_DDIM.py \
  --input_image io_imgs/ihc.tiff \
  --ckpt model/hemit_singlegpu.pth \
  --out_dir he_styled_results \
  --t_enc 50 --height 256 --width 256 \
  --sharpness 4
```

Inference without input (to view pure latent representation)

```bash
python inference_without_input.py \
  --ckpt model/hemit_singlegpu.pth \
  --timesteps 1000 \
  --height 256 \
  --width 256 \
  --batch 1 \
  --out_dir he_styled_results
```






