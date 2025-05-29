from pathlib import Path
from PIL import Image

def convert_pair(input_path: Path, label_path: Path, output_dir: Path) -> None:
    """
    Convert one input–label .tif pair to .png, saving into output_dir.
    The output filenames are prefixed with 'input_' and 'label_' to avoid collisions.
    """
    stem = input_path.stem  # common basename without suffix
    mappings = {
        input_path: f"input_{stem}.png",
        label_path: f"label_{stem}.png",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for src, out_name in mappings.items():
        img = Image.open(src)
        dst = output_dir / out_name
        img.save(dst)
        print(f"Saved {dst}")

if __name__ == "__main__":
    # directory where this script resides
    script_dir = Path(__file__).parent.resolve()

    # ---- Specify exactly two pairs here ----
    pairs = [
        (
            Path("train/input/[10358,37400]_patch_0_0.tif"),
            Path("train/label/[10358,37400]_patch_0_0.tif"),
        ),
        (
            Path("train/input/[12819,37206]_patch_5_6.tif"),
            Path("train/label/[12819,37206]_patch_5_6.tif"),
        ),
    ]

    for inp, lbl in pairs:
        convert_pair(inp, lbl, script_dir)
