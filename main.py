import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import tifffile

from model import *
from gaussion_d import *
from utils import *

class HEMITTrainDataset(Dataset):
    def __init__(self, root: Path, crop_size=256):
        self.input_dir = root / "train" / "input"
        self.ids = [p.stem for p in self.input_dir.glob("*.tif")]
        self.to_pil = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tif = self.input_dir / f"{self.ids[idx]}.tif"
        arr = tifffile.imread(tif)            # H×W×C
        img = self.to_pil(arr)
        return self.transform(img)            # C×H×W in [-1,1]

def train_single_gpu(
    data_root: str = ".",
    batch_size: int = 1,
    crop_size: int = 256,
    lr: float = 5e-4,
    timesteps: int = 500,
    epochs: int = 50,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + loader
    root = Path(data_root)
    ds = HEMITTrainDataset(root, crop_size=crop_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    # Model (smaller UNet)
    model = UNetModel(
        in_channels=3,
        model_channels=32,         # narrower network
        out_channels=3,
        channel_mult=(1, 2),       # fewer levels
        attention_resolutions=()   # no attention
    ).to(device)

    # Diffusion, optimizer, loss, AMP scaler
    diffusion = GaussianDiffusion(timesteps=timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs+1):
        model.train()
        torch.cuda.empty_cache()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, x in enumerate(loader, 1):
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)
                pred = model(x_noisy, t)
                loss = mse(pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{epochs}  Batch {batch_idx}/{len(loader)}  Loss {loss.item():.4f}")

        print(f"Epoch {epoch}/{epochs} done in {time.time()-t0:.1f}s — Avg Loss {epoch_loss/len(loader):.4f}")

    # Save final weights
    torch.save(model.state_dict(), "model/hemit_singlegpu.pth")

if __name__ == "__main__":
    # run from directory containing train/ and this script
    train_single_gpu(data_root=".", batch_size=1, crop_size=256, epochs=10)
