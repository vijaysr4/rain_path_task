import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast
import tifffile
import numpy as np

from model import UNetModel
from gaussion_d import GaussianDiffusion


def load_model(ckpt_path: Path, device: torch.device) -> UNetModel:
    """
    Load the UNetModel from checkpoint.

    Args:
        ckpt_path: Path to the .pth checkpoint file.
        device: Torch device where the model will be loaded.

    Returns:
        A loaded UNetModel in evaluation mode.
    """
    model = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        channel_mult=(1, 2),
        attention_resolutions=()
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def sample_image(
    model: UNetModel,
    diffusion: GaussianDiffusion,
    shape: tuple[int, int, int, int],
    device: torch.device
) -> torch.Tensor:
    """
    Generate images from pure noise via reverse diffusion.

    Args:
        model: The pre-loaded UNetModel.
        diffusion: GaussianDiffusion instance with timesteps configured.
        shape: Tuple(batch_size, channels, height, width).
        device: Torch device for computation.

    Returns:
        A tensor of shape `shape` with values in [-1, 1].
    """
    # Initialize noise
    x = torch.randn(shape, device=device)

    # Reverse diffusion loop
    for t in reversed(range(diffusion.timesteps)):
        batch_ts = torch.full(
            (shape[0],), t,
            device=device,
            dtype=torch.long
        )
        with autocast():
            x = diffusion.p_sample(model, x, batch_ts)

    return x


def save_tensor_as_tiff(x: torch.Tensor, out_path: Path) -> None:
    """
    Convert a [-1,1] tensor to uint8 and save as TIFF.

    Args:
        x: Tensor of shape (C, H, W) with values in [-1, 1].
        out_path: Destination path for the TIFF file.
    """
    # Scale to [0,255]
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    # Reorder to H x W x C
    arr = np.transpose(arr, (1, 2, 0))
    tifffile.imwrite(str(out_path), arr)


def parse_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.

    Returns:
        Namespace with parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Unconditional sampling via Gaussian diffusion"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Model checkpoint file (e.g. hemit_singlegpu.pth)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="Number of diffusion timesteps (must match training)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output image height in pixels"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Output image width in pixels"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("samples"),
        help="Directory to save generated TIFFs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and diffusion
    model = load_model(args.ckpt, device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps)

    # Generate samples
    shape = (args.batch, 3, args.height, args.width)
    generated = sample_image(model, diffusion, shape, device)

    # Save outputs
    for idx in range(args.batch):
        filename = f"gen_{args.height}x{args.width}_{idx:02d}.tiff"
        out_path = args.out_dir / filename
        save_tensor_as_tiff(generated[idx], out_path)
        print(f"Saved sample {idx} to {out_path}")


if __name__ == "__main__":
    main()

'''

python inference_without_input.py \
  --ckpt model/hemit_singlegpu.pth \
  --timesteps 1000 \
  --height 256 \
  --width 256 \
  --batch 1 \
  --out_dir he_styled_results
'''