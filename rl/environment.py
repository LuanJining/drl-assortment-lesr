import numpy as np
from typing import Dict, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

class AssortmentEnvironment(gym.Env):
    def __init__(self, 
                 num_products: int = 10,
                 num_customer_types: int = 4,
                 initial_inventory: Optional[np.ndarray] = None,
                 cardinality: int = 4,
                 prices: Optional[np.ndarray] = None,
                 customer_preferences: Optional[np.ndarray] = None):
        
        super().__init__()
        
        self.num_products = num_products
        self.num_customer_types = num_customer_types
        self.cardinality = cardinality
        
        # 初始化库存
        if initial_inventory is None:
            initial_inventory = np.ones(num_products) * 10
        self.initial_inventory = initial_inventory.astype(float)
        
        # 初始化价格
        if prices is None:
            prices = np.random.uniform(1, 5, num_products)
        self.prices = prices
        
        # 初始化客户偏好矩阵
        if customer_preferences is None:
            # 随机生成客户偏好 (客户类型 x 产品)
            customer_preferences = np.random.dirichlet(np.ones(num_products), 
                                                      num_customer_types)
        self.customer_preferences = customer_preferences
        
        # 定义动作和观察空间
        self.action_space = spaces.MultiBinary(num_products)
        
        # 观察空间：库存 + 客户类型 + 时间
        obs_dim = num_products + num_customer_types + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # 客户到达概率
        self.customer_arrival_prob = np.ones(num_customer_types) / num_customer_types
        
        self.max_time = 100
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            
        self.inventory = self.initial_inventory.copy()
        self.time_remaining = self.max_time
        self.total_revenue = 0.0
        self.sales_history = []
        self.current_customer = self._sample_customer()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        
        # 验证动作合法性
        action = self._validate_action(action)
        
        # 客户购买决策
        purchase_prob = self._calculate_purchase_probability(action)
        
        # 模拟购买
        reward = 0.0
        sold_item = -1
        
        if np.sum(action) > 0 and np.random.random() < np.max(purchase_prob):
            # 客户购买了某个产品
            sold_item = np.random.choice(self.num_products, p=purchase_prob)
            reward = self.prices[sold_item]
            self.inventory[sold_item] -= 1
            self.total_revenue += reward
            self.sales_history.append({
                'time': self.max_time - self.time_remaining,
                'product': sold_item,
                'price': reward,
                'customer_type': self.current_customer
            })
        
        # 时间流逝
        self.time_remaining -= 1
        
        # 采样下一个客户
        if self.time_remaining > 0:
            self.current_customer = self._sample_customer()
        
        # 检查终止条件
        terminated = self.time_remaining <= 0 or np.sum(self.inventory) == 0
        truncated = False
        
        # 获取新状态
        obs = self._get_observation()
        info = self._get_info()
        info['sold_item'] = sold_item
        
        return obs, reward, terminated, truncated, info
    
    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """确保动作合法"""
        # 不能展示缺货的产品
        action = action * (self.inventory > 0)
        
        # 限制展示数量
        if np.sum(action) > self.cardinality:
            # 随机选择cardinality个产品
            selected_indices = np.random.choice(
                np.where(action > 0)[0], 
                self.cardinality, 
                replace=False
            )
            new_action = np.zeros_like(action)
            new_action[selected_indices] = 1
            action = new_action
            
        return action
    
    def _calculate_purchase_probability(self, action: np.ndarray) -> np.ndarray:
        """计算购买概率"""
        # 获取客户偏好
        preferences = self.customer_preferences[self.current_customer]
        
        # 只考虑展示的产品
        displayed_prefs = preferences * action
        
        # 归一化为概率分布
        if np.sum(displayed_prefs) > 0:
            purchase_prob = displayed_prefs / np.sum(displayed_prefs)
        else:
            purchase_prob = np.zeros(self.num_products)
            
        return purchase_prob
    
    def _sample_customer(self) -> int:
        """采样客户类型"""
        return np.random.choice(self.num_customer_types, p=self.customer_arrival_prob)
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        obs = []
        
        # 相对库存水平
        relative_inventory = self.inventory / (self.initial_inventory + 1e-8)
        obs.extend(relative_inventory)
        
        # 客户类型 one-hot
        customer_one_hot = np.zeros(self.num_customer_types)
        customer_one_hot[self.current_customer] = 1
        obs.extend(customer_one_hot)
        
        # 剩余时间比例
        obs.append(self.time_remaining / self.max_time)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'inventory': self.inventory.copy(),
            'customer_type': self.current_customer,
            'time_remaining': self.time_remaining,
            'total_revenue': self.total_revenue,
            'prices': self.prices.copy(),
            'initial_inventory': self.initial_inventory.copy()
        }
    
    def render(self):
        """可视化环境状态"""
        print(f"\n=== Time: {self.max_time - self.time_remaining}/{self.max_time} ===")
        print(f"Inventory: {self.inventory}")
        print(f"Customer Type: {self.current_customer}")
        print(f"Total Revenue: {self.total_revenue:.2f}")