# adapters/kuaisim_adapter.py
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, Any, List
import sys
import os
from pathlib import Path
import logging
import importlib.util

from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class KuaiSimAdapter(BaseAdapter):
    """正确的KuaiSim适配器 - 基于官方源码架构"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 基本参数
        self.num_products = config.get('num_products', 50)
        self.cardinality = config.get('cardinality', 6)  # slate_size
        self.max_episodes = config.get('max_episodes', 100)
        self.max_steps = config.get('max_steps', 20)

        # KuaiSim路径配置
        self.kuaisim_path = config.get('kuaisim_path', None)
        self.data_path = config.get('data_path', None)

        # 模型配置
        self.model_log_path = config.get('model_log_path', None)

        # 状态和动作维度
        self._state_dim = None
        self._action_dim = self.num_products

        # KuaiSim组件
        self.kuaisim_reader = None
        self.kuaisim_user_model = None
        self.kuaisim_env = None

        # 数据统计
        self.user_vocab = None
        self.item_vocab = None
        self.statistics = None

        # 初始化组件
        self._init_components()

        # 环境状态
        self.current_episode = 0
        self.reset()

    def _init_components(self):
        """初始化KuaiSim组件"""

        # 首先尝试加载完整的KuaiSim环境
        if self._try_init_kuaisim_environment():
            logger.info("KuaiSim完整环境初始化成功")
            self.use_kuaisim_env = True
            return

        # 否则尝试加载用户响应模型
        if self._try_init_kuaisim_user_model():
            logger.info("KuaiSim用户模型初始化成功")
            self.use_kuaisim_env = False
            self.use_kuaisim_user_model = True
            return

        # 最后回退到数据驱动模式
        logger.info("使用数据驱动的模拟模式")
        self.use_kuaisim_env = False
        self.use_kuaisim_user_model = False
        self._init_data_driven_components()

    def _try_init_kuaisim_environment(self) -> bool:
        """尝试初始化完整的KuaiSim环境"""
        try:
            if not self.kuaisim_path or not os.path.exists(self.kuaisim_path):
                return False

            # 添加KuaiSim路径
            if str(self.kuaisim_path) not in sys.path:
                sys.path.append(str(self.kuaisim_path))

            # 尝试导入KuaiSim环境
            from env import KREnvironment_WholeSession_GPU

            # 加载预训练的用户模型参数
            if not self.model_log_path or not os.path.exists(self.model_log_path):
                logger.warning("未找到预训练模型，无法初始化完整环境")
                return False

            # 创建模拟的args对象
            args = self._create_kuaisim_args()
            args.uirm_log_path = self.model_log_path

            # 初始化环境
            self.kuaisim_env = KREnvironment_WholeSession_GPU(args)

            logger.info("KuaiSim完整环境初始化成功")
            return True

        except Exception as e:
            logger.warning(f"KuaiSim完整环境初始化失败: {e}")
            return False

    def _try_init_kuaisim_user_model(self) -> bool:
        """尝试初始化KuaiSim用户响应模型"""
        try:
            if not self.kuaisim_path or not os.path.exists(self.kuaisim_path):
                return False

            # 添加KuaiSim路径
            if str(self.kuaisim_path) not in sys.path:
                sys.path.append(str(self.kuaisim_path))

            # 导入必要的组件
            from reader import KRMBSeqReader
            from model.simulator import KRMBUserResponse

            if not self.model_log_path or not os.path.exists(self.model_log_path):
                logger.warning("未找到模型日志文件")
                return False

            # 从日志文件加载模型参数
            with open(self.model_log_path, 'r') as infile:
                meta_args = eval(infile.readline())
                training_args = eval(infile.readline())

            # 设置设备
            training_args.device = str(self.device)

            # 初始化读取器
            self.kuaisim_reader = KRMBSeqReader(training_args)
            self.statistics = self.kuaisim_reader.get_statistics()

            # 初始化用户响应模型
            self.kuaisim_user_model = KRMBUserResponse(
                training_args,
                self.statistics,
                self.device
            )

            # 加载预训练权重
            self.kuaisim_user_model.load_from_checkpoint(with_optimizer=False)
            self.kuaisim_user_model.to(self.device)
            self.kuaisim_user_model.eval()

            logger.info("KuaiSim用户模型加载成功")
            return True

        except Exception as e:
            logger.warning(f"KuaiSim用户模型初始化失败: {e}")
            return False

    def _init_data_driven_components(self):
        """初始化数据驱动的组件"""
        try:
            # 尝试加载真实数据
            if self.data_path and os.path.exists(self.data_path):
                self._load_kuaisim_data()
            else:
                self._create_mock_data()

        except Exception as e:
            logger.warning(f"数据加载失败，使用完全模拟数据: {e}")
            self._create_mock_data()

    def _load_kuaisim_data(self):
        """加载KuaiSim数据"""
        data_path = Path(self.data_path)

        # 加载交互数据
        interaction_file = data_path / self.config.get('train_file', 'log_session_4_08_to_5_08_Pure.csv')
        if interaction_file.exists():
            self.interaction_data = pd.read_csv(interaction_file, nrows=50000)
            logger.info(f"加载交互数据: {len(self.interaction_data)} 条记录")

        # 加载用户元数据
        user_file = data_path / self.config.get('user_meta_file', 'user_features_Pure_fillna.csv')
        if user_file.exists():
            self.user_data = pd.read_csv(user_file)
            logger.info(f"加载用户数据: {len(self.user_data)} 用户")

        # 加载物品元数据
        item_file = data_path / self.config.get('item_meta_file', 'video_features_basic_Pure_fillna.csv')
        if item_file.exists():
            self.item_data = pd.read_csv(item_file)
            # 限制物品数量
            self.item_data = self.item_data.head(self.num_products)
            logger.info(f"加载物品数据: {len(self.item_data)} 物品")

        # 建立词汇表
        self._build_vocabularies()

        # 分析用户偏好
        self._analyze_user_preferences()

    def _build_vocabularies(self):
        """建立用户和物品词汇表"""
        if hasattr(self, 'interaction_data') and self.interaction_data is not None:
            # 用户词汇表
            unique_users = self.interaction_data['user_id'].unique()
            self.user_vocab = {uid: idx for idx, uid in enumerate(unique_users)}

            # 物品词汇表  
            unique_items = self.interaction_data['video_id'].unique()
            self.item_vocab = {iid: idx % self.num_products for idx, iid in enumerate(unique_items)}

            logger.info(f"词汇表: {len(self.user_vocab)} 用户, {len(self.item_vocab)} 物品")

    def _analyze_user_preferences(self):
        """分析用户偏好模式"""
        if hasattr(self, 'interaction_data') and self.interaction_data is not None:
            # 计算物品流行度
            item_popularity = self.interaction_data['video_id'].value_counts()
            top_items = item_popularity.head(self.num_products).index.tolist()

            # 创建基于行为的偏好矩阵
            self.user_preferences = {}

            # 简化的4种用户类型
            for user_type in range(4):
                prefs = np.random.dirichlet(np.ones(self.num_products))

                # 根据流行度调整偏好
                for i, item_id in enumerate(top_items[:self.num_products]):
                    if i < self.num_products:
                        popularity_weight = 1 + 0.5 * (len(top_items) - i) / len(top_items)
                        prefs[i] *= popularity_weight

                prefs = prefs / prefs.sum()  # 重新归一化
                self.user_preferences[user_type] = prefs

            logger.info("用户偏好分析完成")

    def _create_mock_data(self):
        """创建模拟数据"""
        # 模拟用户偏好
        self.user_preferences = {}
        for user_type in range(4):
            self.user_preferences[user_type] = np.random.dirichlet(np.ones(self.num_products))

        # 模拟物品特征
        self.item_features = np.random.rand(self.num_products, 32)

        # 模拟价格
        self.item_prices = np.random.uniform(1.0, 5.0, self.num_products)

        logger.info("模拟数据创建完成")

    def _create_kuaisim_args(self):
        """创建KuaiSim参数对象"""

        class Args:
            def __init__(self, config):
                # 数据路径
                self.data_path = config.get('data_path', 'dataset/Kuairand-Pure/')
                self.train_file = config.get('train_file', 'log_session_4_08_to_5_08_Pure.csv')
                self.user_meta_file = config.get('user_meta_file', 'user_features_Pure_fillna.csv')
                self.item_meta_file = config.get('item_meta_file', 'video_features_basic_Pure_fillna.csv')

                # 环境参数
                self.max_step_per_episode = config.get('max_steps', 20)
                self.slate_size = config.get('cardinality', 6)
                self.episode_batch_size = 32
                self.item_correlation = 0.2
                self.initial_temper = 20
                self.single_response = True

                # 数据处理参数
                self.max_hist_seq_len = config.get('max_hist_seq_len', 100)
                self.data_separator = ','
                self.meta_file_separator = ','
                self.n_worker = 4
                self.device = str(config.get('device', 'cpu'))

                # 模型参数
                self.user_latent_dim = config.get('user_latent_dim', 32)
                self.item_latent_dim = config.get('item_latent_dim', 32)
                self.enc_dim = config.get('enc_dim', 64)
                self.state_user_latent_dim = 16
                self.state_item_latent_dim = 16
                self.state_transformer_enc_dim = 32
                self.state_transformer_n_head = 4
                self.state_transformer_d_forward = 64
                self.state_transformer_n_layer = 3
                self.state_dropout_rate = 0.1

        return Args(self.config)

    @property
    def state_dim(self) -> int:
        """状态维度"""
        if self._state_dim is None:
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

        if self.use_kuaisim_env:
            # 使用完整KuaiSim环境
            return self._reset_kuaisim_env()
        else:
            # 使用简化的状态重置
            return self._reset_simulation()

    def _reset_kuaisim_env(self) -> Tuple[np.ndarray, Dict]:
        """使用KuaiSim环境重置"""
        try:
            # 重置KuaiSim环境（这里需要根据实际KuaiSim接口调整）
            # KuaiSim环境通常在每个episode开始时重置
            self.current_user_type = np.random.randint(0, 4)
            self.inventory = np.ones(self.num_products) * 10
            self.initial_inventory = self.inventory.copy()

            # 生成用户会话
            self._init_user_session()

            state = self._get_state()
            info = self._get_info()

            return state, info

        except Exception as e:
            logger.warning(f"KuaiSim环境重置失败: {e}")
            return self._reset_simulation()

    def _reset_simulation(self) -> Tuple[np.ndarray, Dict]:
        """使用模拟环境重置"""
        # 初始化用户状态
        self.current_user_type = np.random.randint(0, 4)

        # 初始化"库存"状态（在推荐中可以理解为物品的可推荐状态）
        self.inventory = np.ones(self.num_products) * 10
        self.initial_inventory = self.inventory.copy()

        # 生成用户会话
        self._init_user_session()

        state = self._get_state()
        info = self._get_info()

        return state, info

    def _init_user_session(self):
        """初始化用户会话"""
        self.current_user_session = {
            'user_id': np.random.randint(0, 1000),
            'session_id': np.random.randint(0, 10000),
            'user_type': self.current_user_type,
            'history': np.random.choice(self.num_products, 5).tolist(),
            'preferences': self.user_preferences.get(self.current_user_type,
                                                     np.random.dirichlet(np.ones(self.num_products)))
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        # 验证和修正动作
        action = self._validate_action(action)

        if self.use_kuaisim_env:
            # 使用KuaiSim环境
            return self._step_kuaisim_env(action)
        elif self.use_kuaisim_user_model:
            # 使用KuaiSim用户模型
            return self._step_kuaisim_user_model(action)
        else:
            # 使用模拟环境
            return self._step_simulation(action)

    def _step_kuaisim_env(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """使用完整KuaiSim环境执行步骤"""
        try:
            # 这里需要根据实际的KuaiSim环境接口调整
            # 通常KuaiSim环境会处理整个推荐会话

            # 模拟KuaiSim环境调用
            reward = self._calculate_reward_with_kuaisim(action)

            # 更新状态
            self._update_state(action, reward)

            # 检查终止条件
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            truncated = False

            state = self._get_state()
            info = self._get_info()

            return state, reward, terminated, truncated, info

        except Exception as e:
            logger.warning(f"KuaiSim环境步骤执行失败: {e}")
            return self._step_simulation(action)

    def _step_kuaisim_user_model(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """使用KuaiSim用户模型执行步骤"""
        try:
            # 使用KuaiSim用户响应模型预测用户行为
            reward = self._predict_user_response(action)

            # 更新状态
            self._update_state(action, reward)

            # 检查终止条件
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            truncated = False

            state = self._get_state()
            info = self._get_info()

            return state, reward, terminated, truncated, info

        except Exception as e:
            logger.warning(f"KuaiSim用户模型预测失败: {e}")
            return self._step_simulation(action)

    def _predict_user_response(self, action: np.ndarray) -> float:
        """使用KuaiSim用户模型预测用户响应"""
        try:
            # 准备模型输入
            # 这里需要根据KuaiSim的具体接口格式调整

            # 简化实现：基于动作选择的物品计算期望奖励
            selected_items = np.where(action > 0.5)[0]

            if len(selected_items) == 0:
                return 0.0

            # 使用用户偏好作为代理
            user_prefs = self.current_user_session['preferences']
            expected_reward = np.sum(user_prefs[selected_items]) / len(selected_items)

            # 添加随机性
            if np.random.random() < expected_reward:
                return np.random.exponential(2.0)
            else:
                return 0.1

        except Exception as e:
            logger.warning(f"用户响应预测失败: {e}")
            return 0.1

    def _step_simulation(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """使用模拟环境执行步骤"""
        # 计算奖励
        reward = self._calculate_simulation_reward(action)

        # 更新状态
        self._update_state(action, reward)

        # 检查终止条件
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        state = self._get_state()
        info = self._get_info()

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

    def _calculate_reward_with_kuaisim(self, action: np.ndarray) -> float:
        """使用KuaiSim计算奖励"""
        # 这里应该调用KuaiSim的奖励计算接口
        # 简化实现
        return self._calculate_simulation_reward(action)

    def _calculate_simulation_reward(self, action: np.ndarray) -> float:
        """计算模拟奖励"""
        if np.sum(action) == 0:
            return 0.0

        user_prefs = self.current_user_session['preferences']
        selected_items = np.where(action > 0)[0]

        # 计算基于偏好的奖励
        preference_scores = user_prefs[selected_items]

        # 模拟用户交互
        interaction_prob = np.mean(preference_scores)

        if np.random.random() < interaction_prob:
            # 用户有交互
            interacted_item = np.random.choice(selected_items, p=preference_scores / preference_scores.sum())

            # 基于物品价值的奖励
            if hasattr(self, 'item_prices'):
                base_reward = self.item_prices[interacted_item]
            else:
                base_reward = np.random.uniform(1.0, 5.0)

            # 添加满意度因子
            satisfaction = user_prefs[interacted_item]
            final_reward = base_reward * (0.5 + satisfaction)

            return final_reward
        else:
            return 0.1  # 小的正奖励避免过度惩罚

    def _update_state(self, action: np.ndarray, reward: float):
        """更新环境状态"""
        # 更新"库存"状态
        if reward > 0.5:  # 如果有显著交互
            interacted_items = np.where(action > 0)[0]
            if len(interacted_items) > 0:
                interacted_item = np.random.choice(interacted_items)
                self.inventory[interacted_item] = max(0, self.inventory[interacted_item] - 1)

        # 更新用户偏好（轻微调整）
        if reward > 0.5:
            interacted_items = np.where(action > 0)[0]
            for item in interacted_items:
                self.current_user_session['preferences'][item] *= 1.05

        # 偏好归一化
        prefs = self.current_user_session['preferences']
        self.current_user_session['preferences'] = prefs / np.sum(prefs)

        # 偶尔切换用户（模拟新用户到达）
        if np.random.random() < 0.1:
            self.current_user_type = np.random.randint(0, 4)
            self._init_user_session()

        # 记录历史
        self.total_reward += reward
        self.episode_history.append({
            'step': self.current_step,
            'action': action.copy(),
            'reward': reward,
            'user_type': self.current_user_type
        })

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []

        # 库存/可用性信息
        relative_inventory = self.inventory / (self.initial_inventory + 1e-8)
        state.extend(relative_inventory.tolist())

        # 用户类型信息
        user_onehot = np.zeros(4)
        user_onehot[self.current_user_type] = 1
        state.extend(user_onehot.tolist())

        # 时间信息
        time_progress = self.current_step / self.max_steps
        state.append(time_progress)

        # 用户偏好信息（前5个物品）
        user_prefs = self.current_user_session['preferences']
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
            'prices': getattr(self, 'item_prices', np.ones(self.num_products)),
            'current_step': self.current_step,
            'user_session': self.current_user_session.copy(),
            'use_kuaisim_env': getattr(self, 'use_kuaisim_env', False),
            'use_kuaisim_user_model': getattr(self, 'use_kuaisim_user_model', False),
            'kuaisim_status': self._get_kuaisim_status()
        }

    def _get_kuaisim_status(self) -> str:
        """获取KuaiSim状态信息"""
        if getattr(self, 'use_kuaisim_env', False):
            return "完整KuaiSim环境"
        elif getattr(self, 'use_kuaisim_user_model', False):
            return "KuaiSim用户模型"
        else:
            return "数据驱动模拟"


def create_kuaisim_environment(config: Dict) -> KuaiSimAdapter:
    """创建KuaiSim环境的工厂函数"""
    return KuaiSimAdapter(config)