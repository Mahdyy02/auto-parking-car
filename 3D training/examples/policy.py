<<<<<<< HEAD
import torch
import torch.nn as nn


class DeterministicPolicy(nn.Module):
    """Simple MLP policy that maps state vectors to continuous actions (steer, throttle, brake).

    Replace the image-embedding portion of the input with a ViT embedding when ready.
    """
    def __init__(self, state_dim: int, hidden_sizes=(1024, 512, 256)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        # output 3 dims: steering (-0.5..0.5), throttle (-1..1), brake (bool {0,1})
        layers.append(nn.Linear(last, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # steering to tanh (-1..1), throttle/brake to sigmoid (0..1)
        steer = torch.tanh(out[:, 0:1])
        thr = torch.sigmoid(out[:, 1:2])
        brk = torch.sigmoid(out[:, 2:3])
        return torch.cat([steer, thr, brk], dim=1)
=======
import torch
import torch.nn as nn


class DeterministicPolicy(nn.Module):
    """Simple MLP policy that maps state vectors to continuous actions (steer, throttle, brake).

    Replace the image-embedding portion of the input with a ViT embedding when ready.
    """
    def __init__(self, state_dim: int, hidden_sizes=(1024, 512, 256)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        # output 3 dims: steering (-0.5..0.5), throttle (-1..1), brake (bool {0,1})
        layers.append(nn.Linear(last, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # steering to tanh (-1..1), throttle/brake to sigmoid (0..1)
        steer = torch.tanh(out[:, 0:1])
        thr = torch.sigmoid(out[:, 1:2])
        brk = torch.sigmoid(out[:, 2:3])
        return torch.cat([steer, thr, brk], dim=1)
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
