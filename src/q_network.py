from typing import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP Q-network for Pong states."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
