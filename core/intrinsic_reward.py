import numpy as np
import importlib.util
import tempfile
import os
from typing import Any


class IntrinsicRewardCalculator:
    def __init__(self):
        self.reward_func = None

    def load_function(self, function_code: str):
        """动态加载奖励函数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(function_code)
            temp_file = f.name

        try:
            spec = importlib.util.spec_from_file_location("reward_module", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'intrinsic_reward'):
                self.reward_func = module.intrinsic_reward
                return True
            else:
                print("错误：未找到intrinsic_reward函数")
                return False

        except Exception as e:
            print(f"加载奖励函数失败: {e}")
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def calculate(self, state: np.ndarray, action: int,
                  next_state: np.ndarray, sold_item: int, price: float) -> float:
        """计算内在奖励"""
        if self.reward_func is None:
            return self._default_reward(state, action, next_state, sold_item, price)

        try:
            reward = self.reward_func(state, action, next_state, sold_item, price)
            return float(reward)
        except Exception as e:
            print(f"奖励计算失败，使用默认方法: {e}")
            return self._default_reward(state, action, next_state, sold_item, price)

    def _default_reward(self, state: np.ndarray, action: int,
                        next_state: np.ndarray, sold_item: int, price: float) -> float:
        """默认奖励函数"""
        reward = 0.0

        # 销售奖励
        if sold_item >= 0:
            reward += price * 0.1

        # 库存平衡奖励（使用状态中的库存信息）
        if len(state) > 10:
            inventory_std = np.std(state[:10])
            reward -= inventory_std * 0.05

        return reward