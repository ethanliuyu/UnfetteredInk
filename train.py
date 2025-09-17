import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import handSvgDataset  # type: ignore
from models.model import TrajectoryFrequencyModel


# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = "model_last.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    if not os.path.isfile(checkpoint_path):
        print(f"[Checkpoint] No checkpoint found at {checkpoint_path}")
        return 0
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Loaded from {checkpoint_path} (epoch {start_epoch})")
    return start_epoch


# ------------------------------------------------------------
# Training script
# ------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # 1. Dataset & DataLoader
    # --------------------------------------------------------
    lmdb_path = "./DataPreparation/LMDB/lmdb"
    dataset = handSvgDataset(lmdb_path)
    batch_size = 2
    # macOS / Windows 默认采用 spawn，多进程下无法 pickle LMDB env → 设为 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # --------------------------------------------------------
    # 2. Model, Optimiser, Scheduler
    # --------------------------------------------------------
    seq_len = dataset.sqeLen
    model = TrajectoryFrequencyModel(seq_len=seq_len).to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # --------------------------------------------------------
    # 3. Resume from checkpoint (optional)
    # --------------------------------------------------------
    checkpoint_dir = "./TrajCheckpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "model_last.pth.tar")
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer) if os.path.isfile(checkpoint_path) else 0

    # --------------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------------
    epochs = 200
    log_interval = 1  # iterations
    save_interval = 1  # epochs

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100)
        for it, (inputs, targets, styles, images) in enumerate(pbar):
            # Move tensors to device
            inputs = inputs.to(device)        # (B,L,2)
            targets = targets.to(device)      # (B,L,2)
            styles = styles.to(device)        # (B,K,L,2)
            images = images.to(device)        # (B,K,256,256)

            # --------------------------------------------------
            # Forward pass (per-sample due to current API)
            # --------------------------------------------------

            # 直接整体送入模型，由模型内部提取图像风格特征
            recon_fft_batch = model(inputs, styles, images)  # (B,L,2) complex

            # --------------------------------------------------
            # Loss & optimisation
            # --------------------------------------------------
            loss_total, comps = model.prepare_and_compute_loss(
                recon_fft=recon_fft_batch,
                traj_target=targets,
                style_seq_ref=styles,
                ref_images=images,
                lambda1=1.0, lambda2=1.0, lambda3=0.5, lambda4=0.5,
            )

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item()
            pbar.set_postfix(Total=f'{loss_total.item():.4f}',
                             MSE=f"{comps['mse'].item():.4f}",
                             IMG=f"{comps['img'].item():.4f}",
                             RG=f"{comps['rg'].item():.4f}",
                             RT=f"{comps['rt'].item():.4f}")


        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f} ===")

        # ----------------------------------------------------
        # 5. Save checkpoint
        # ----------------------------------------------------
        if (epoch + 1) % save_interval == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_dir, filename=f"model_epoch_{epoch+1}.pth.tar")
            # Always update last
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_dir, filename="model_last.pth.tar")
