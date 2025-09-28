import numpy as np
import importlib.util
import tempfile
import os
from typing import Any, Union


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

    def calculate(self, state: np.ndarray, action: Union[int, np.ndarray],
                  next_state: np.ndarray, sold_item: int, price: float) -> float:
        """计算内在奖励 - 修复参数类型处理"""
        if self.reward_func is None:
            return self._default_reward(state, action, next_state, sold_item, price)

        try:
            # 确保参数类型正确
            state = self._ensure_array(state)
            next_state = self._ensure_array(next_state)
            action = self._process_action(action)
            sold_item = self._safe_convert_to_int(sold_item)
            price = self._safe_convert_to_float(price)

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

    def _process_action(self, action):
        """处理action参数，支持标量和数组"""
        if action is None:
            return 0

        if isinstance(action, (int, float)):
            return int(action)
        elif isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.item())
            else:
                # 如果是多维动作，返回选中的动作索引
                selected_actions = np.where(action > 0.5)[0]
                if len(selected_actions) > 0:
                    return int(selected_actions[0])
                else:
                    return 0
        else:
            try:
                return int(action)
            except:
                return 0

    def _safe_convert_to_int(self, value):
        """安全转换为整数"""
        if value is None:
            return -1

        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return int(value.item())
                else:
                    return int(value[0]) if len(value) > 0 else -1
            else:
                return int(value)
        except:
            return -1

    def _safe_convert_to_float(self, value):
        """安全转换为浮点数"""
        if value is None:
            return 0.0

        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                else:
                    return float(value[0]) if len(value) > 0 else 0.0
            else:
                return float(value)
        except:
            return 0.0

    def _default_reward(self, state: np.ndarray, action: Union[int, np.ndarray],
                        next_state: np.ndarray, sold_item: int, price: float) -> float:
        """默认奖励函数 - 加强错误处理"""
        try:
            reward = 0.0

            # 确保参数类型
            state = self._ensure_array(state)
            next_state = self._ensure_array(next_state)
            action_processed = self._process_action(action)
            sold_item = self._safe_convert_to_int(sold_item)
            price = self._safe_convert_to_float(price)

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

            # 动作多样性奖励
            if isinstance(action, np.ndarray) and len(action) > 1:
                try:
                    # 奖励选择多个不同的动作
                    num_selected = np.sum(action > 0.5)
                    if num_selected > 1:
                        reward += 0.01 * num_selected
                except Exception:
                    pass

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
            print(f"默认奖励计算也失败: {e}")
            return 0.01