# adapters/kuaisim_adapter.py
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, Any
import sys
import os
from pathlib import Path
import logging

from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class KuaiSimAdapter(BaseAdapter):
    """KuaiSim环境适配器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 基本参数
        self.num_products = config.get('num_products', 50)
        self.cardinality = config.get('cardinality', 6)
        self.max_episodes = config.get('max_episodes', 100)
        self.max_steps = config.get('max_steps', 20)

        # KuaiSim路径配置
        self.kuaisim_path = config.get('kuaisim_path', None)
        self.data_path = config.get('data_path', 'dataset/Kuairand-Pure/')

        # 状态和动作维度
        self._state_dim = None
        self._action_dim = self.num_products

        # 初始化KuaiSim或模拟组件
        self._init_components()

        # 环境状态
        self.current_episode = 0
        self.reset()

    def _init_components(self):
        """初始化KuaiSim组件或模拟组件"""

        if self.kuaisim_path and self._try_init_kuaisim():
            logger.info("KuaiSim组件初始化成功")
            self.use_kuaisim = True
        else:
            logger.info("使用模拟数据模式")
            self.use_kuaisim = False
            self._init_mock_components()

    def _try_init_kuaisim(self) -> bool:
        """尝试初始化KuaiSim"""
        try:
            # 添加KuaiSim路径
            if self.kuaisim_path not in sys.path:
                sys.path.append(str(self.kuaisim_path))

            # 导入KuaiSim模块
            from reader import KRMBSeqReader
            from model.simulator import KRMBUserResponse

            # 创建参数对象
            self.args = self._create_kuaisim_args()

            # 初始化组件
            self.reader = KRMBSeqReader(self.args)
            self.user_model = KRMBUserResponse(
                self.args,
                self.reader.get_statistics(),
                self.device
            )

            # 加载预训练模型（如果存在）
            model_path = self.config.get('model_path')
            if model_path and os.path.exists(model_path):
                self.user_model.load_from_checkpoint(with_optimizer=False)
                logger.info(f"加载预训练模型: {model_path}")

            return True

        except Exception as e:
            logger.warning(f"KuaiSim初始化失败: {e}")
            return False

    def _create_kuaisim_args(self):
        """创建KuaiSim参数对象"""

        class Args:
            def __init__(self, config):
                # 数据路径
                self.data_path = config.get('data_path', 'dataset/Kuairand-Pure/')
                self.train_file = config.get('train_file', 'log_session_4_08_to_5_08_Pure.csv')
                self.user_meta_file = config.get('user_meta_file', 'user_features_Pure_fillna.csv')
                self.item_meta_file = config.get('item_meta_file', 'video_features_basic_Pure_fillna.csv')

                # 数据处理参数
                self.max_hist_seq_len = config.get('max_hist_seq_len', 100)
                self.data_separator = ','
                self.meta_file_separator = ','
                self.val_holdout_per_user = 5
                self.test_holdout_per_user = 5
                self.n_worker = 4
                self.device = config.get('device', 'cpu')

                # 模型参数
                self.user_latent_dim = config.get('user_latent_dim', 32)
                self.item_latent_dim = config.get('item_latent_dim', 32)
                self.enc_dim = config.get('enc_dim', 64)
                self.n_ensemble = config.get('n_ensemble', 2)
                self.attn_n_head = config.get('attn_n_head', 4)
                self.transformer_d_forward = config.get('transformer_d_forward', 64)
                self.transformer_n_layer = config.get('transformer_n_layer', 2)
                self.state_hidden_dims = config.get('state_hidden_dims', [128])
                self.scorer_hidden_dims = config.get('scorer_hidden_dims', [128, 32])
                self.dropout_rate = config.get('dropout_rate', 0.1)
                self.l2_coef = config.get('l2_coef', 0.01)

        return Args(self.config)

    def _init_mock_components(self):
        """初始化模拟组件"""
        # 模拟物品特征
        self.mock_item_features = np.random.rand(self.num_products, 32)

        # 模拟用户偏好（4种用户类型）
        self.mock_user_preferences = np.random.dirichlet(
            np.ones(self.num_products), 4
        )

        # 模拟价格
        self.mock_prices = np.random.uniform(1.0, 5.0, self.num_products)

        # 模拟用户历史
        self.mock_user_history = {
            'recent_items': np.random.choice(self.num_products, 10),
            'preferences': np.random.rand(self.num_products)
        }

    @property
    def state_dim(self) -> int:
        """状态维度"""
        if self._state_dim is None:
            # 计算状态维度
            temp_state = self._get_state()
            self._state_dim = len(temp_state)
        return self._state_dim

    @property
    def action_dim(self) -> int:
        """动作维度"""
        return self._action_dim

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        # 重置环境状态
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_history = []

        # 初始化用户状态
        self.current_user_type = np.random.randint(0, 4)

        # 初始化"库存"（在推荐场景中可以理解为物品的可推荐状态）
        self.inventory = np.ones(self.num_products) * 10
        self.initial_inventory = self.inventory.copy()

        # 生成用户历史（模拟或从KuaiSim获取）
        if self.use_kuaisim:
            self._init_kuaisim_user_state()
        else:
            self._init_mock_user_state()

        # 获取初始状态
        state = self._get_state()
        info = self._get_info()

        return state, info

    def _init_kuaisim_user_state(self):
        """初始化KuaiSim用户状态"""
        try:
            # 从KuaiSim数据中采样用户会话
            # 这里需要根据实际的KuaiSim数据格式调整
            self.current_user_session = self._sample_user_session()

        except Exception as e:
            logger.warning(f"KuaiSim用户状态初始化失败: {e}")
            self._init_mock_user_state()

    def _init_mock_user_state(self):
        """初始化模拟用户状态"""
        self.current_user_session = {
            'user_id': np.random.randint(0, 1000),
            'session_id': np.random.randint(0, 10000),
            'history': np.random.choice(self.num_products, 5),
            'preferences': self.mock_user_preferences[self.current_user_type]
        }

    def _sample_user_session(self) -> Dict:
        """从KuaiSim数据中采样用户会话"""
        # 这里应该实现从KuaiSim数据中采样的逻辑
        # 简化实现
        return {
            'user_id': np.random.randint(0, 1000),
            'session_id': np.random.randint(0, 10000),
            'history': np.random.choice(self.num_products, 5),
            'preferences': np.random.dirichlet(np.ones(self.num_products))
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        # 验证和修正动作
        action = self._validate_action(action)

        # 计算奖励
        if self.use_kuaisim:
            reward = self._calculate_kuaisim_reward(action)
        else:
            reward = self._calculate_mock_reward(action)

        # 更新环境状态
        self._update_state(action, reward)

        # 检查终止条件
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 获取新状态
        state = self._get_state()
        info = self._get_info()

        self.total_reward += reward

        # 记录历史
        self.episode_history.append({
            'step': self.current_step,
            'action': action.copy(),
            'reward': reward,
            'state': state.copy()
        })

        return state, reward, terminated, truncated, info

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """验证并修正动作"""
        # 确保动作是二进制
        action = (action > 0.5).astype(float)

        # 限制推荐数量
        if np.sum(action) > self.cardinality:
            selected_indices = np.random.choice(
                np.where(action > 0)[0],
                self.cardinality,
                replace=False
            )
            new_action = np.zeros_like(action)
            new_action[selected_indices] = 1
            action = new_action

        return action

    def _calculate_kuaisim_reward(self, action: np.ndarray) -> float:
        """使用KuaiSim计算奖励"""
        try:
            # 选择推荐的物品
            recommended_items = np.where(action > 0)[0]

            if len(recommended_items) == 0:
                return 0.0

            # 使用KuaiSim的用户模型预测交互
            # 这里需要根据实际的KuaiSim接口调整
            reward = self._predict_user_interaction(recommended_items)

            return reward

        except Exception as e:
            logger.warning(f"KuaiSim奖励计算失败: {e}")
            return self._calculate_mock_reward(action)

    def _predict_user_interaction(self, recommended_items: np.ndarray) -> float:
        """预测用户交互（KuaiSim）"""
        # 简化实现 - 实际需要调用KuaiSim的用户模型

        # 模拟用户偏好得分
        preference_scores = np.random.rand(len(recommended_items))

        # 计算期望奖励
        expected_reward = np.sum(preference_scores) / len(recommended_items)

        # 添加随机性
        if np.random.random() < expected_reward:
            return np.random.exponential(2.0)  # 正奖励
        else:
            return 0.1  # 小的正奖励

    def _calculate_mock_reward(self, action: np.ndarray) -> float:
        """计算模拟奖励"""
        if np.sum(action) == 0:
            return 0.0

        # 基于用户偏好计算奖励
        user_prefs = self.current_user_session['preferences']
        displayed_prefs = user_prefs * action

        if np.sum(displayed_prefs) > 0:
            # 模拟用户交互
            interaction_prob = np.sum(displayed_prefs) / self.cardinality

            if np.random.random() < interaction_prob:
                # 用户有交互，计算奖励
                selected_probs = displayed_prefs / np.sum(displayed_prefs)
                interacted_item = np.random.choice(self.num_products, p=selected_probs)

                # 基于物品价值计算奖励
                base_reward = self.mock_prices[interacted_item]

                # 添加用户满意度因子
                satisfaction = user_prefs[interacted_item]
                final_reward = base_reward * (0.5 + satisfaction)

                return final_reward

        return 0.1  # 避免过度惩罚

    def _update_state(self, action: np.ndarray, reward: float):
        """更新环境状态"""
        # 更新"库存"状态
        if reward > 0.5:  # 如果有显著交互
            interacted_items = np.where(action > 0)[0]
            if len(interacted_items) > 0:
                # 随机选择一个被交互的物品
                interacted_item = np.random.choice(interacted_items)
                self.inventory[interacted_item] = max(0, self.inventory[interacted_item] - 1)

        # 更新用户状态
        if self.use_kuaisim:
            self._update_kuaisim_user_state(action, reward)
        else:
            self._update_mock_user_state(action, reward)

    def _update_kuaisim_user_state(self, action: np.ndarray, reward: float):
        """更新KuaiSim用户状态"""
        # 这里应该调用KuaiSim的状态更新逻辑
        pass

    def _update_mock_user_state(self, action: np.ndarray, reward: float):
        """更新模拟用户状态"""
        # 用户偏好会根据交互轻微调整
        if reward > 0.5:
            interacted_items = np.where(action > 0)[0]
            for item in interacted_items:
                # 增强对交互物品的偏好
                self.current_user_session['preferences'][item] *= 1.05

        # 偏好归一化
        prefs = self.current_user_session['preferences']
        self.current_user_session['preferences'] = prefs / np.sum(prefs)

        # 10%概率切换用户类型（模拟新用户）
        if np.random.random() < 0.1:
            self.current_user_type = np.random.randint(0, 4)
            self._init_mock_user_state()

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []

        # 库存/物品可用性信息
        relative_inventory = self.inventory / (self.initial_inventory + 1e-8)
        state.extend(relative_inventory.tolist())

        # 用户类型信息
        user_onehot = np.zeros(4)
        user_onehot[self.current_user_type] = 1
        state.extend(user_onehot.tolist())

        # 时间信息
        time_progress = self.current_step / self.max_steps
        state.append(time_progress)

        # 用户偏好信息（部分）
        user_prefs = self.current_user_session['preferences']
        # 只取前5个物品的偏好作为特征
        state.extend(user_prefs[:5].tolist())

        # 聚合特征
        total_inventory_ratio = np.sum(self.inventory) / np.sum(self.initial_inventory)
        state.append(total_inventory_ratio)

        inventory_std = np.std(relative_inventory)
        state.append(inventory_std)

        preference_entropy = -np.sum(user_prefs * np.log(user_prefs + 1e-8))
        state.append(preference_entropy)

        return np.array(state, dtype=np.float32)

    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            'inventory': self.inventory.copy(),
            'initial_inventory': self.initial_inventory.copy(),
            'current_user_type': self.current_user_type,
            'time_remaining': self.max_steps - self.current_step,
            'total_reward': self.total_reward,
            'prices': getattr(self, 'mock_prices', np.ones(self.num_products)),
            'current_step': self.current_step,
            'user_session': self.current_user_session.copy(),
            'use_kuaisim': self.use_kuaisim
        }


def create_kuaisim_environment(config: Dict) -> KuaiSimAdapter:
    """创建KuaiSim环境的工厂函数"""
    return KuaiSimAdapter(config)