import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


class A2CAgent(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.001):
        super().__init__()

        # 共享网络层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.action_dim = action_dim  # 确保 action_dim 被正确初始化

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        shared_features = self.shared_layers(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def select_action(self,
                      state: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        """选择动作（商品组合）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_logits, _ = self.forward(state_tensor)

            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                action_logits = action_logits.masked_fill(mask_tensor == 1, -1e10)

            action_probs = torch.softmax(action_logits, dim=-1)

            # 采样动作
            action = torch.multinomial(action_probs, 1).item()  # 获取动作的索引

        return np.array([action])  # 返回动作，确保是 numpy 数组
