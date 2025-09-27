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

        # 确保action_dim正确设置
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

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

        # 初始化权重
        self._init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 确保输入是正确的形状
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        shared_features = self.shared_layers(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)

        return action_logits, value

    def select_action(self,
                      state: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """选择动作（商品组合）"""

        # 确保状态是正确的格式
        if isinstance(state, list):
            state = np.array(state)

        state_tensor = torch.FloatTensor(state)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            action_logits, value = self.forward(state_tensor)

            # 应用掩码（如果提供）
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask)
                if len(mask_tensor.shape) == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
                action_logits = action_logits.masked_fill(mask_tensor.bool(), -1e10)

            action_probs = torch.softmax(action_logits, dim=-1)

            if deterministic:
                # 确定性选择最高概率的动作
                action_idx = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs.gather(1, action_idx.unsqueeze(1))).squeeze()
            else:
                # 随机采样动作
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

            # 转换为二进制动作向量
            action = np.zeros(self.action_dim, dtype=np.float32)
            action_index = action_idx.item()
            if action_index < self.action_dim:
                action[action_index] = 1

        return action, log_prob.item() if hasattr(log_prob, 'item') else float(log_prob)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """评估动作（用于训练）"""
        action_logits, values = self.forward(states)

        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        with torch.no_grad():
            _, value = self.forward(state)
            return value.squeeze()

    def update_policy(self, states, actions, returns, old_log_probs=None):
        """更新策略（简化版）"""
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)

        # 获取当前策略的log概率和价值
        log_probs, values, entropy = self.evaluate_actions(states_tensor, actions_tensor)

        # 计算优势
        advantages = returns_tensor - values.detach()

        # Actor损失
        actor_loss = -(log_probs * advantages).mean()

        # Critic损失
        critic_loss = ((values - returns_tensor) ** 2).mean()

        # 熵奖励（鼓励探索）
        entropy_loss = -entropy.mean()

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_logits, _ = self.forward(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs.squeeze().numpy()