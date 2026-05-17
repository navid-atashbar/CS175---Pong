from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2

# Map agent actions to Gymnasium ALE/Pong-v5 actions.
# ALE: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=DOWN, 5=UP
ALE_ACTION_MAP = {
    ACTION_STAY: 0,
    ACTION_UP: 5,
    ACTION_DOWN: 4,
}


class DQNAgent:
    """
    Uses epsilon-greedy exploration, experience replay, and a target network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        buffer_capacity: int = 100_000,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        min_epsilon: float = 0.05,
        target_update_freq: int = 1000,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_capacity)

        self.train_steps = 0
        self.losses: list[float] = []

    def to_env_action(self, action: int) -> int:
        return ALE_ACTION_MAP[int(action)]

    def select_action(
        self,
        state: np.ndarray,
        *,
        training: bool = True,
    ) -> int:
        """
        Epsilon-greedy action selection.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        return self.greedy_action(state)

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = self._state_tensor(state)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Sample a minibatch and perform one gradient step.

        Returns the loss if an update was performed, otherwise None.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        loss_value = float(loss.item())
        self.losses.append(loss_value)

        if self.train_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return loss_value

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "gamma": self.gamma,
                    "batch_size": self.batch_size,
                    "epsilon_decay": self.epsilon_decay,
                    "min_epsilon": self.min_epsilon,
                    "target_update_freq": self.target_update_freq,
                },
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.train_steps = int(checkpoint.get("train_steps", 0))

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32)
        return torch.as_tensor(state, device=self.device).unsqueeze(0)
