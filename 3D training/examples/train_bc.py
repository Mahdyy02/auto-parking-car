<<<<<<< HEAD
"""Train a deterministic policy by Behavioral Cloning from collected episodes.

Usage:
    python examples\train_bc.py --data_dir data/episodes --epochs 10 --batch 64
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from policy import DeterministicPolicy


class EpisodeDataset(Dataset):
    def __init__(self, files):
        states = []
        actions = []
        for f in files:
            d = np.load(f)
            states.append(d['states'])
            actions.append(d['actions'])
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx].astype(np.float32), self.actions[idx].astype(np.float32)


def train(data_dir, epochs=10, batch_size=64, lr=1e-3, device='cpu'):
    files = glob.glob(os.path.join(data_dir, '*.npz'))
    if not files:
        raise RuntimeError('No episode files found in ' + data_dir)

    ds = EpisodeDataset(files)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    state_dim = ds.states.shape[1]
    policy = DeterministicPolicy(state_dim)
    policy.to(device)

    opt = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for e in range(epochs):
        total_loss = 0.0
        for batch_states, batch_actions in tqdm(loader, desc=f'Epoch {e+1}/{epochs}'):
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            pred = policy(batch_states)
            loss = loss_fn(pred, batch_actions)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_states.size(0)

        avg_loss = total_loss / len(ds)
        print(f'Epoch {e+1} avg loss: {avg_loss:.6f}')

    # save model
    torch.save(policy.state_dict(), os.path.join(data_dir, 'policy_bc.pt'))
    print('Saved policy to', os.path.join(data_dir, 'policy_bc.pt'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/episodes')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
=======
"""Train a deterministic policy by Behavioral Cloning from collected episodes.

Usage:
    python examples\train_bc.py --data_dir data/episodes --epochs 10 --batch 64
"""
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from policy import DeterministicPolicy


class EpisodeDataset(Dataset):
    def __init__(self, files):
        states = []
        actions = []
        for f in files:
            d = np.load(f)
            states.append(d['states'])
            actions.append(d['actions'])
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx].astype(np.float32), self.actions[idx].astype(np.float32)


def train(data_dir, epochs=10, batch_size=64, lr=1e-3, device='cpu'):
    files = glob.glob(os.path.join(data_dir, '*.npz'))
    if not files:
        raise RuntimeError('No episode files found in ' + data_dir)

    ds = EpisodeDataset(files)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    state_dim = ds.states.shape[1]
    policy = DeterministicPolicy(state_dim)
    policy.to(device)

    opt = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for e in range(epochs):
        total_loss = 0.0
        for batch_states, batch_actions in tqdm(loader, desc=f'Epoch {e+1}/{epochs}'):
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            pred = policy(batch_states)
            loss = loss_fn(pred, batch_actions)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_states.size(0)

        avg_loss = total_loss / len(ds)
        print(f'Epoch {e+1} avg loss: {avg_loss:.6f}')

    # save model
    torch.save(policy.state_dict(), os.path.join(data_dir, 'policy_bc.pt'))
    print('Saved policy to', os.path.join(data_dir, 'policy_bc.pt'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/episodes')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.data_dir, epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
