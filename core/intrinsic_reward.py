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
        """计算内在奖励 - 修复参数类型处理"""
        if self.reward_func is None:
            return self._default_reward(state, action, next_state, sold_item, price)

        try:
            # 确保参数类型正确
            state = self._ensure_array(state)
            next_state = self._ensure_array(next_state)
            action = int(action) if action is not None else 0
            sold_item = int(sold_item) if sold_item is not None else -1
            price = float(price) if price is not None else 0.0

            reward = self.reward_func(state, action, next_state, sold_item, price)
            return float(reward)

        except Exception as e:
            print(f"奖励计算失败，使用默认方法: {e}")
            return self._default_reward(state, action, next_state, sold_item, price)

    def _ensure_array(self, value):
        """确保值是numpy数组"""
        if value is None:
            return np.array([])
        elif isinstance(value, (int, float)):
            return np.array([value])
        elif isinstance(value, list):
            return np.array(value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            try:
                return np.array(value)
            except:
                return np.array([])

    def _default_reward(self, state: np.ndarray, action: int, 
                       next_state: np.ndarray, sold_item: int, price: float) -> float:
        """默认奖励函数 - 加强错误处理"""
        try:
            reward = 0.0

            # 确保参数类型
            state = self._ensure_array(state)
            next_state = self._ensure_array(next_state)
            sold_item = int(sold_item) if sold_item is not None else -1
            price = float(price) if price is not None else 0.0

            # 销售奖励
            if sold_item >= 0 and price > 0:
                reward += price * 0.1

            # 库存平衡奖励（如果状态数组足够长）
            if len(state) > 10:
                try:
                    inventory_features = state[:10]
                    if len(inventory_features) > 1:
                        inventory_std = np.std(inventory_features)
                        reward -= inventory_std * 0.05
                except Exception:
                    pass  # 忽略库存特征计算错误

            # 时间压力奖励（如果状态数组包含时间信息）
            if len(state) > 14:
                try:
                    time_remaining = float(state[14])
                    if 0 <= time_remaining <= 1:
                        reward += (1.0 - time_remaining) * 0.02
                except Exception:
                    pass  # 忽略时间特征计算错误

            # 确保奖励在合理范围内
            reward = np.clip(reward, -10.0, 10.0)

            return float(reward)

        except Exception as e:
            # 如果所有计算都失败，返回小的正奖励
            return 0.01
