import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout_rate: float = 0.0):
        super().__init__()

        self.layers = nn.ModuleList()

        # 输入层到第一个隐藏层
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            # 添加激活函数
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())

            # 添加Dropout
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))

        # 输出层
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionNetwork(nn.Module):
    """注意力网络"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0

        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 计算Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        # 重塑回原始格式
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        # 输出投影
        output = self.output_projection(context)

        return output


class StateEncoder(nn.Module):
    """状态编码器 - 用于处理复杂状态表示"""

    def __init__(self, inventory_dim: int, customer_types: int,
                 feature_dim: int = 64, use_attention: bool = False):
        super().__init__()

        self.inventory_dim = inventory_dim
        self.customer_types = customer_types
        self.feature_dim = feature_dim

        # 库存编码器
        self.inventory_encoder = MLP(
            input_dim=inventory_dim,
            hidden_dims=[feature_dim, feature_dim],
            output_dim=feature_dim
        )

        # 客户类型编码器
        self.customer_encoder = nn.Embedding(customer_types, feature_dim)

        # 时间编码器
        self.time_encoder = MLP(
            input_dim=1,
            hidden_dims=[feature_dim // 2],
            output_dim=feature_dim // 2
        )

        # 特征融合
        if use_attention:
            self.feature_fusion = AttentionNetwork(
                input_dim=feature_dim,
                hidden_dim=feature_dim
            )
        else:
            self.feature_fusion = MLP(
                input_dim=feature_dim * 2 + feature_dim // 2,
                hidden_dims=[feature_dim, feature_dim],
                output_dim=feature_dim
            )

        self.use_attention = use_attention

    def forward(self, inventory: torch.Tensor, customer_type: torch.Tensor,
                time_remaining: torch.Tensor) -> torch.Tensor:

        # 编码库存
        inventory_features = self.inventory_encoder(inventory)

        # 编码客户类型
        customer_features = self.customer_encoder(customer_type.long())

        # 编码时间
        time_features = self.time_encoder(time_remaining.unsqueeze(-1))

        if self.use_attention:
            # 使用注意力机制融合特征
            combined_features = torch.stack([
                inventory_features,
                customer_features,
                torch.cat([time_features, torch.zeros_like(time_features)], dim=-1)
            ], dim=1)

            fused_features = self.feature_fusion(combined_features)
            fused_features = fused_features.mean(dim=1)  # 平均池化

        else:
            # 简单拼接
            combined_features = torch.cat([
                inventory_features,
                customer_features,
                time_features
            ], dim=-1)

            fused_features = self.feature_fusion(combined_features)

        return fused_features


class DuelingNetwork(nn.Module):
    """Dueling网络架构 - 分离状态价值和动作优势"""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # 共享特征提取器
        self.feature_extractor = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=hidden_dim
        )

        # 状态价值分支
        self.value_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim // 2],
            output_dim=1
        )

        # 动作优势分支
        self.advantage_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim // 2],
            output_dim=action_dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)

        # 计算状态价值
        value = self.value_head(features)

        # 计算动作优势
        advantage = self.advantage_head(features)

        # Dueling聚合：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values, value


class NoiseNetwork(nn.Module):
    """噪声网络 - 用于探索"""

    def __init__(self, input_dim: int, output_dim: int, std_init: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        # 权重参数
        self.weight_mu = nn.Parameter(torch.empty((output_dim, input_dim)))
        self.weight_sigma = nn.Parameter(torch.empty((output_dim, input_dim)))

        # 偏置参数
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """重置参数"""
        mu_range = 1.0 / np.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.input_dim))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.output_dim))

    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        x = x + residual
        x = self.norm(x)
        return F.relu(x)


class DeepResidualNetwork(nn.Module):
    """深度残差网络"""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 256, num_blocks: int = 3):
        super().__init__()

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


def create_network(network_type: str, **kwargs) -> nn.Module:
    """网络工厂函数"""

    if network_type == 'mlp':
        return MLP(**kwargs)
    elif network_type == 'attention':
        return AttentionNetwork(**kwargs)
    elif network_type == 'dueling':
        return DuelingNetwork(**kwargs)
    elif network_type == 'residual':
        return DeepResidualNetwork(**kwargs)
    elif network_type == 'state_encoder':
        return StateEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# 权重初始化函数
def init_weights(module: nn.Module, method: str = 'xavier'):
    """初始化网络权重"""
    if isinstance(module, nn.Linear):
        if method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif method == 'normal':
            nn.init.normal_(module.weight, 0, 0.1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)