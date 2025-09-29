import numpy as np


class RandomAgent:
    """随机选择算法：基准对比"""

    def __init__(self, num_products: int, cardinality: int):
        self.num_products = num_products
        self.cardinality = cardinality

    def select_action(self, inventory: np.ndarray) -> np.ndarray:
        """随机选择可用商品"""

        # 找出有库存的产品
        available_products = np.where(inventory > 0)[0]

        if len(available_products) == 0:
            return np.zeros(self.num_products)

        # 随机选择k个产品
        k = min(self.cardinality, len(available_products))
        selected = np.random.choice(available_products, k, replace=False)

        action = np.zeros(self.num_products)
        action[selected] = 1

        return action