import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class DataGenerator:
    """生成训练和测试数据"""

    def __init__(self, num_products: int = 10,
                 num_customer_types: int = 4,
                 seed: Optional[int] = None):

        self.num_products = num_products
        self.num_customer_types = num_customer_types

        if seed is not None:
            np.random.seed(seed)

        # 生成产品属性
        self.product_features = self._generate_product_features()

        # 生成客户偏好
        self.customer_preferences = self._generate_customer_preferences()

        # 生成价格
        self.prices = self._generate_prices()

    def _generate_product_features(self) -> np.ndarray:
        """生成产品特征矩阵"""
        # 5个特征维度：品质、品牌、新鲜度、包装、促销
        features = np.random.rand(self.num_products, 5)

        # 归一化
        features = features / features.sum(axis=0, keepdims=True)

        return features

    def _generate_customer_preferences(self) -> np.ndarray:
        """生成客户偏好矩阵"""
        # 使用Dirichlet分布生成偏好
        preferences = np.zeros((self.num_customer_types, self.num_products))

        for i in range(self.num_customer_types):
            # 每个客户类型有不同的偏好分布
            alpha = np.random.uniform(0.5, 2.0, self.num_products)
            preferences[i] = np.random.dirichlet(alpha)

        return preferences

    def _generate_prices(self) -> np.ndarray:
        """生成产品价格"""
        # 基础价格
        base_prices = np.random.uniform(1.0, 5.0, self.num_products)

        # 根据产品质量调整价格
        quality_factor = self.product_features[:, 0]  # 第一个特征作为质量
        adjusted_prices = base_prices * (0.8 + 0.4 * quality_factor)

        return adjusted_prices

    def generate_transaction_data(self, num_transactions: int = 10000) -> pd.DataFrame:
        """生成历史交易数据"""
        transactions = []

        for _ in range(num_transactions):
            # 随机客户类型
            customer_type = np.random.choice(self.num_customer_types)

            # 随机展示的商品（假设展示4个）
            num_displayed = min(4, self.num_products)
            displayed_products = np.random.choice(
                self.num_products,
                num_displayed,
                replace=False
            )

            # 基于客户偏好计算购买概率
            preferences = self.customer_preferences[customer_type, displayed_products]
            purchase_probs = preferences / preferences.sum()

            # 决定是否购买（80%概率购买）
            if np.random.random() < 0.8:
                purchased_product = np.random.choice(displayed_products, p=purchase_probs)
                revenue = self.prices[purchased_product]
            else:
                purchased_product = -1  # 未购买
                revenue = 0

            transactions.append({
                'customer_type': customer_type,
                'displayed_products': displayed_products.tolist(),
                'purchased_product': purchased_product,
                'revenue': revenue,
                'timestamp': np.random.randint(0, 100)  # 模拟时间戳
            })

        return pd.DataFrame(transactions)

    def generate_customer_sequence(self, length: int = 100,
                                   arrival_rates: Optional[np.ndarray] = None) -> List[int]:
        """生成客户到达序列"""
        if arrival_rates is None:
            # 默认均匀到达率
            arrival_rates = np.ones(self.num_customer_types) / self.num_customer_types

        sequence = np.random.choice(
            self.num_customer_types,
            size=length,
            p=arrival_rates
        )

        return sequence.tolist()

    def generate_initial_inventory(self,
                                   total_inventory: int = 100,
                                   distribution: str = 'uniform') -> np.ndarray:
        """生成初始库存分布"""

        if distribution == 'uniform':
            # 均匀分布
            base_inv = total_inventory // self.num_products
            remainder = total_inventory % self.num_products
            inventory = np.ones(self.num_products) * base_inv
            inventory[:remainder] += 1

        elif distribution == 'skewed':
            # 偏斜分布（某些产品库存更多）
            weights = np.random.dirichlet(np.ones(self.num_products))
            inventory = np.round(weights * total_inventory)

            # 调整以确保总和正确
            diff = total_inventory - inventory.sum()
            if diff > 0:
                inventory[0] += diff
            else:
                inventory[0] = max(0, inventory[0] + diff)

        elif distribution == 'normal':
            # 正态分布
            mean_inv = total_inventory / self.num_products
            std_inv = mean_inv * 0.3
            inventory = np.random.normal(mean_inv, std_inv, self.num_products)
            inventory = np.maximum(inventory, 1)  # 至少1个
            inventory = np.round(inventory)

            # 归一化到总库存
            inventory = inventory * (total_inventory / inventory.sum())
            inventory = np.round(inventory)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return inventory.astype(int)

    def save_config(self, filepath: str):
        """保存配置到文件"""
        config = {
            'num_products': self.num_products,
            'num_customer_types': self.num_customer_types,
            'product_features': self.product_features.tolist(),
            'customer_preferences': self.customer_preferences.tolist(),
            'prices': self.prices.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> 'DataGenerator':
        """从配置文件加载"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        generator = cls(
            num_products=config['num_products'],
            num_customer_types=config['num_customer_types']
        )

        generator.product_features = np.array(config['product_features'])
        generator.customer_preferences = np.array(config['customer_preferences'])
        generator.prices = np.array(config['prices'])

        return generator

    def generate_test_scenarios(self) -> List[Dict]:
        """生成测试场景"""
        scenarios = []

        # 场景1: 高峰期（所有客户类型均匀到达）
        scenarios.append({
            'name': 'High Traffic - Uniform',
            'customer_sequence': self.generate_customer_sequence(200),
            'initial_inventory': self.generate_initial_inventory(100, 'uniform')
        })

        # 场景2: 特定客户群体集中
        concentrated_rates = np.zeros(self.num_customer_types)
        concentrated_rates[0] = 0.7  # 70%是类型0的客户
        concentrated_rates[1:] = 0.3 / (self.num_customer_types - 1)
        scenarios.append({
            'name': 'Concentrated Customer Type',
            'customer_sequence': self.generate_customer_sequence(200, concentrated_rates),
            'initial_inventory': self.generate_initial_inventory(100, 'uniform')
        })

        # 场景3: 库存不平衡
        scenarios.append({
            'name': 'Imbalanced Inventory',
            'customer_sequence': self.generate_customer_sequence(200),
            'initial_inventory': self.generate_initial_inventory(100, 'skewed')
        })

        # 场景4: 库存紧张
        scenarios.append({
            'name': 'Low Inventory',
            'customer_sequence': self.generate_customer_sequence(200),
            'initial_inventory': self.generate_initial_inventory(50, 'uniform')
        })

        return scenarios