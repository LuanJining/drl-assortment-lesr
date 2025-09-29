import numpy as np
from typing import Dict, Any, Callable
import importlib.util
import tempfile
import os


class StateEnhancer:
    def __init__(self):
        self.enhance_func = None
        self.original_dim = None
        self.enhanced_dim = None

    def load_function(self, function_code: str):
        """动态加载状态增强函数"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(function_code)
            temp_file = f.name

        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("enhance_module", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 获取enhance_state函数
            if hasattr(module, 'enhance_state'):
                self.enhance_func = module.enhance_state
                return True
            else:
                print("错误：未找到enhance_state函数")
                return False

        except Exception as e:
            print(f"加载函数失败: {e}")
            return False
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def enhance(self, state: Dict[str, Any]) -> np.ndarray:
        """增强状态"""
        if self.enhance_func is None:
            # 使用默认增强
            return self._default_enhance(state)

        try:
            enhanced = self.enhance_func(
                inventory=state['inventory'],
                customer_type=state['customer_type'],
                prices=state['prices'],
                time_remaining=state['time_remaining'],
                initial_inventory=state['initial_inventory']
            )
            return enhanced
        except Exception as e:
            print(f"状态增强失败，使用默认方法: {e}")
            return self._default_enhance(state)

    def _default_enhance(self, state: Dict[str, Any]) -> np.ndarray:
        """默认状态增强方法"""
        inventory = state['inventory']
        customer_type = state['customer_type']
        time_remaining = state['time_remaining']
        initial_inventory = state['initial_inventory']

        # 基础状态
        base_state = []

        # 相对库存
        relative_inventory = inventory / (initial_inventory + 1e-8)
        base_state.extend(relative_inventory)

        # 客户类型one-hot编码
        customer_encoding = np.zeros(4)
        customer_encoding[customer_type] = 1
        base_state.extend(customer_encoding)

        # 时间
        base_state.append(time_remaining / 100.0)

        # 增强特征
        inventory_sum = inventory.sum()
        initial_sum = initial_inventory.sum()

        # 库存压力
        pressure = 1.0 - (inventory_sum / (initial_sum + 1e-8))
        base_state.append(pressure)

        # 库存不平衡度
        imbalance = np.std(relative_inventory)
        base_state.append(imbalance)

        return np.array(base_state, dtype=np.float32)