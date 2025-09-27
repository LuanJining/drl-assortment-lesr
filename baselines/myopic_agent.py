import numpy as np
from typing import Optional

class MyopicAgent:
    """贪心算法：只考虑当前收益最大化"""
    
    def __init__(self, num_products: int, cardinality: int):
        self.num_products = num_products
        self.cardinality = cardinality
        
    def select_action(self, inventory: np.ndarray, prices: np.ndarray, 
                     customer_prefs: Optional[np.ndarray] = None) -> np.ndarray:
        """选择当前最优的商品组合"""
        
        # 计算每个产品的期望收益
        if customer_prefs is not None:
            expected_revenue = prices * customer_prefs
        else:
            expected_revenue = prices
        
        # 只考虑有库存的产品
        available = inventory > 0
        expected_revenue = expected_revenue * available
        
        # 选择收益最高的K个产品
        k = min(self.cardinality, np.sum(available))
        if k == 0:
            return np.zeros(self.num_products)
        
        top_k = np.argsort(expected_revenue)[-k:]
        
        action = np.zeros(self.num_products)
        action[top_k] = 1
        
        return action