import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import tifffile
import numpy as np

from model import UNetModel
from gaussion_d import GaussianDiffusion

def load_model_and_diffusion(ckpt_path: Path, device: torch.device, timesteps: int):
    model = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        channel_mult=(1, 2),
        attention_resolutions=()
    ).to(device)
    diffusion = GaussianDiffusion(timesteps=timesteps)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, diffusion

@torch.no_grad()
def diffusion_style_transfer(model, diffusion, x, t_enc, device):
    """
    1) Encode x at timestep t_enc → x_noisy
    2) Reverse-diffuse x_noisy → x_styled,
       off-loading to CPU between steps.
    """
    # forward noising
    noise = torch.randn_like(x)
    t = torch.full((x.size(0),), t_enc, device=device, dtype=torch.long)
    with autocast():
        x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)

    # reverse diffusion one step at a time, CPU-offloading
    x_d = x_noisy
    for timestep in reversed(range(t_enc + 1)):
        ts = torch.full((x.size(0),), timestep, device=device, dtype=torch.long)
        with autocast():
            x_d = diffusion.p_sample(model, x_d, ts)
        x_d = x_d.cpu()
        torch.cuda.empty_cache()
        x_d = x_d.to(device)
    return x_d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_image", type=Path, required=True,
                   help="Path to your input image (JPEG, PNG, or TIFF)")
    p.add_argument("--ckpt", type=Path, default="hemit_singlegpu.pth",
                   help="Trained model checkpoint")
    p.add_argument("--out_dir", type=Path, default=Path("styled_outputs"),
                   help="Where to save H&E-styled TIFF")
    p.add_argument("--timesteps", type=int, default=500,
                   help="Diffusion timesteps (must match training)")
    p.add_argument("--t_enc", type=int, default=250,
                   help="Encoding timestep (0 = no change, higher = more stylized)")
    p.add_argument("--height", type=int, default=256,
                   help="Resize height (try 256 or lower if OOM persists)")
    p.add_argument("--width", type=int, default=256,
                   help="Resize width (try 256 or lower if OOM persists)")
    p.add_argument("--sharpness", type=float, default=1.0,
                   help="Sharpness factor for post-processing (1.0 = no change)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read & preprocess your input image
    suffix = args.input_image.suffix.lower()
    if suffix in ['.tif', '.tiff']:
        arr = tifffile.imread(str(args.input_image))
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        img = Image.open(args.input_image).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    x = preprocess(img).unsqueeze(0).to(device)

    # Load model & diffusion
    model, diffusion = load_model_and_diffusion(args.ckpt, device, args.timesteps)
    torch.cuda.empty_cache()

    # Style-transfer
    styled = diffusion_style_transfer(model, diffusion, x, args.t_enc, device)

    # Post-process and save as TIFF with optional sharpening
    # Convert to uint8 HxWxC array
    arr = ((styled[0].clamp(-1,1) + 1) * 127.5).round().byte().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    pil_img = Image.fromarray(arr)

    # Apply sharpness adjustment
    if args.sharpness != 1.0:
        pil_img = TF.adjust_sharpness(pil_img, args.sharpness)

    # Save result
    final_arr = np.array(pil_img)
    out_path = args.out_dir / f"styled_t{args.t_enc:03d}_s{args.sharpness:.1f}.tiff"
    tifffile.imwrite(str(out_path), final_arr)
    print(f"H&E-styled (sharpness={args.sharpness}) image saved to: {out_path}")

if __name__ == "__main__":
    main()


'''
python inference_DDIM.py \
  --input_image io_imgs/ihc.tiff \
  --ckpt model/hemit_singlegpu.pth \
  --out_dir he_styled_results \
  --t_enc 50 --height 256 --width 256 \
  --sharpness 4

'''