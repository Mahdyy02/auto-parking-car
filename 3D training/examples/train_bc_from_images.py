<<<<<<< HEAD
"""Train a behavioural cloning model from the dataset created by `record_bc_dataset.py`.

Saves a PyTorch model `bc_model.pt` to the output folder.

Usage:
    python examples/train_bc_from_images.py --data data/bc_dataset/manual_drive --epochs 10
"""
import os
import argparse
import pickle
from typing import List, Dict

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class ImageBCDataset(Dataset):
    def __init__(self, index_path, root_dir):
        with open(index_path, 'rb') as f:
            idx = pickle.load(f)
        self.records = idx['records']
        self.root = root_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img_path = os.path.join(self.root, rec['image'])
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # C x H x W
        action = np.array([rec['steering'], rec['throttle'], rec['brake']], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(action)


class BCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    index_path = os.path.join(args.data, 'index.pkl')
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    ds = ImageBCDataset(index_path, args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, actions in loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        avg = total_loss / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs}, loss={avg:.6f}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'bc_model.pt')
    torch.save({'model_state_dict': model.state_dict()}, out_path)
    print(f"Saved model to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/bc_dataset/manual_drive')
    p.add_argument('--out', default='data/bc_models')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
=======
"""Train a behavioural cloning model from the dataset created by `record_bc_dataset.py`.

Saves a PyTorch model `bc_model.pt` to the output folder.

Usage:
    python examples/train_bc_from_images.py --data data/bc_dataset/manual_drive --epochs 10
"""
import os
import argparse
import pickle
from typing import List, Dict

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class ImageBCDataset(Dataset):
    def __init__(self, index_path, root_dir):
        with open(index_path, 'rb') as f:
            idx = pickle.load(f)
        self.records = idx['records']
        self.root = root_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img_path = os.path.join(self.root, rec['image'])
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # C x H x W
        action = np.array([rec['steering'], rec['throttle'], rec['brake']], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(action)


class BCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    index_path = os.path.join(args.data, 'index.pkl')
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    ds = ImageBCDataset(index_path, args.data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, actions in loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
        avg = total_loss / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs}, loss={avg:.6f}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'bc_model.pt')
    torch.save({'model_state_dict': model.state_dict()}, out_path)
    print(f"Saved model to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/bc_dataset/manual_drive')
    p.add_argument('--out', default='data/bc_models')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
