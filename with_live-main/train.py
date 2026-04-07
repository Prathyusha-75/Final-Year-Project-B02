import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.zero_dce.model import enhance_net_nopool
import cv2
import numpy as np
import glob
import os
from pathlib import Path

class LowLightDataset(Dataset):
    def __init__(self, data_path):
        self.image_list = glob.glob(os.path.join(data_path, "*.*"))
    def __getitem__(self, index):
        img = cv2.imread(self.image_list[index])
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256))
        return torch.from_numpy(img.astype(np.float32)/255.0).permute(2, 0, 1)
    def __len__(self):
        return len(self.image_list)

def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "train"
    weights_path = base_dir / "models" / "zero_dce" / "weights.pth"

    model = enhance_net_nopool().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    if not data_dir.exists():
        print("Put training images in data/train/ first!"); return

    dataset = LowLightDataset(str(data_dir))
    if len(dataset) == 0:
        print("No training images found in data/train/."); return

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("Training started...")

    for epoch in range(30):
        for img in loader:
            img = img.to(device)
            enhanced = model(img)
            
            # Exposure Loss (Targets 0.6 brightness)
            loss = torch.mean(torch.abs(torch.mean(enhanced, dim=1) - 0.6))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Complete")
        torch.save(model.state_dict(), weights_path)

if __name__ == "__main__":
    train()
