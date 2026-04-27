import os
import subprocess
import torch
from torch.utils.data import DataLoader

from src.custom_model import CustomModel
from src.clip_dl import CocoDataset, Flickr30kDataset


coco_dataset = False

# Load dataset
if coco_dataset:
    if "datasets" not in os.listdir():
        print("COCO dataset is not downloaded. Running download script...")
        subprocess.run(["python", "src/download_coco_data.py"])

    clip_dataset = CocoDataset(root_dir="datasets")
else:
    clip_dataset = Flickr30kDataset()


# Small batch for laptop/CPU testing
clip_dataloader = DataLoader(
    clip_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# Use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Create model
model = CustomModel().to(device)

# Optimizer
optimizer = torch.optim.Adam(
    [
        {"params": model.vision_encoder.parameters()},
        {"params": model.caption_encoder.parameters()},
    ],
    lr=model.lr,
)

# Quick test: run only ONE batch
num_epochs = 1

for epoch in range(num_epochs):
    model.train()

    for batch_idx, batch in enumerate(clip_dataloader):
        image = batch["image"].to(device)
        text = batch["caption"]

        loss, img_acc, cap_acc = model(image, text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Batch [{batch_idx + 1}]")
        print(f"Loss: {loss.item()}")
        print("One batch finished successfully ✅")

        break

print("Training test complete.")