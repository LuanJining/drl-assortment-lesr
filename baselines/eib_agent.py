import numpy as np


class EIBAgent:
    """Expected Inventory Balance算法：考虑库存平衡"""

    def __init__(self, num_products: int, cardinality: int):
        self.num_products = num_products
        self.cardinality = cardinality

    def select_action(self, inventory: np.ndarray, initial_inventory: np.ndarray,
                      prices: np.ndarray, time_remaining: int) -> np.ndarray:
        """基于库存平衡选择商品"""

        # 计算相对库存水平
        relative_inventory = inventory / (initial_inventory + 1e-8)

        # 计算库存压力（库存越多压力越大）
        inventory_pressure = relative_inventory

        # 结合价格和库存压力
        scores = prices * inventory_pressure

        # 只考虑有库存的产品
        available = inventory > 0
        scores = scores * available

        # 选择得分最高的K个产品
        k = min(self.cardinality, np.sum(available))
        if k == 0:
            return np.zeros(self.num_products)

        top_k = np.argsort(scores)[-k:]

        action = np.zeros(self.num_products)
        action[top_k] = 1

        return action