import numpy as np
from typing import Dict, Tuple, Optional

class AssortmentEnvironment:
    def __init__(self, 
                 num_products: int = 10,
                 num_customer_types: int = 4,
                 initial_inventory: np.ndarray = None,
                 cardinality: int = 4):
        
        self.num_products = num_products
        self.num_customer_types = num_customer_types
        self.cardinality = cardinality
        
        if initial_inventory is None:
            initial_inventory = np.ones(num_products) * 10
        
        self.initial_inventory = initial_inventory
        self.reset()
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.inventory = self.initial_inventory.copy()
        self.time_remaining = 100
        self.total_revenue = 0
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 确保动作合法（不展示缺货商品）
        action = self._validate_action(action)
        
        # 模拟客户到达和购买
        customer_type = np.random.choice(self.num_customer_types)
        purchase = self._simulate_purchase(action, customer_type)
        
        # 计算奖励
        reward = self._calculate_reward(purchase)
        
        # 更新库存
        if purchase >= 0:
            self.inventory[purchase] -= 1
            
        # 更新时间
        self.time_remaining -= 1
        
        # 检查终止条件
        done = self.time_remaining <= 0 or self.inventory.sum() == 0
        
        # 获取新状态
        next_state = self._get_state()
        
        info = {
            'revenue': reward,
            'inventory': self.inventory.copy(),
            'customer_type': customer_type
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = np.concatenate([
            self.inventory / self.initial_inventory,  # 相对库存水平
            [self.time_remaining / 100.0]            # 剩余时间比例
        ])
        return state