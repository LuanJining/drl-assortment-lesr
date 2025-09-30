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
        self._error_count = 0
        self._max_errors = 3  # 只打印前3次错误

    def load_function(self, function_code: str):
        """动态加载状态增强函数"""
        # 确保代码包含 numpy 导入
        if 'import numpy' not in function_code:
            function_code = 'import numpy as np\n\n' + function_code

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(function_code)
            temp_file = f.name

        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("enhance_module", temp_file)
            module = importlib.util.module_from_spec(spec)

            # 在模块的全局命名空间中注入 numpy
            module.__dict__['np'] = np
            module.__dict__['numpy'] = np

            spec.loader.exec_module(module)

            # 获取enhance_state函数
            if hasattr(module, 'enhance_state'):
                self.enhance_func = module.enhance_state
                self._error_count = 0  # 重置错误计数
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
                try:
                    os.remove(temp_file)
                except:
                    pass

    def enhance(self, state: Dict[str, Any]) -> np.ndarray:
        """增强状态 - 带参数类型保护"""
        if self.enhance_func is None:
            return self._default_enhance(state)

        try:
            # 🔧 参数类型保护和转换
            inventory = np.array(state['inventory'], dtype=np.float32)

            # customer_type 必须是标量
            customer_type = state['customer_type']
            if isinstance(customer_type, np.ndarray):
                customer_type = int(customer_type.item())
            else:
                customer_type = int(customer_type)

            # prices 可能是None或数组
            prices = state.get('prices')
            if prices is not None:
                prices = np.array(prices, dtype=np.float32)
            else:
                prices = np.ones(len(inventory), dtype=np.float32)

            # time_remaining 必须是标量
            time_remaining = state['time_remaining']
            if isinstance(time_remaining, np.ndarray):
                time_remaining = float(time_remaining.item())
            else:
                time_remaining = float(time_remaining)

            # initial_inventory
            initial_inventory = np.array(state['initial_inventory'], dtype=np.float32)

            # 调用增强函数
            enhanced = self.enhance_func(
                inventory=inventory,
                customer_type=customer_type,  # 传递标量
                prices=prices,
                time_remaining=time_remaining,  # 传递标量
                initial_inventory=initial_inventory
            )

            # 确保返回正确的numpy数组
            if isinstance(enhanced, (list, tuple)):
                enhanced = np.array(enhanced, dtype=np.float32)
            elif not isinstance(enhanced, np.ndarray):
                enhanced = np.array([enhanced], dtype=np.float32)

            return enhanced.astype(np.float32)

        except Exception as e:
            # 只打印前几次错误，避免日志刷屏
            if self._error_count < self._max_errors:
                print(f"状态增强失败，使用默认方法: {e}")
                self._error_count += 1
            elif self._error_count == self._max_errors:
                print(f"状态增强持续失败，后续错误将被静默...")
                self._error_count += 1

            return self._default_enhance(state)

    def _default_enhance(self, state: Dict[str, Any]) -> np.ndarray:
        """默认状态增强方法"""
        inventory = np.array(state['inventory'], dtype=np.float32)
        customer_type = int(state['customer_type'])
        time_remaining = float(state['time_remaining'])
        initial_inventory = np.array(state['initial_inventory'], dtype=np.float32)

        # 基础状态
        base_state = []

        # 相对库存
        relative_inventory = inventory / (initial_inventory + 1e-8)
        base_state.extend(relative_inventory.tolist())

        # 客户类型one-hot编码
        customer_encoding = np.zeros(4, dtype=np.float32)
        if 0 <= customer_type < 4:
            customer_encoding[customer_type] = 1
        base_state.extend(customer_encoding.tolist())

        # 时间
        base_state.append(time_remaining / 100.0)

        # 增强特征
        inventory_sum = float(inventory.sum())
        initial_sum = float(initial_inventory.sum())

        # 库存压力
        pressure = 1.0 - (inventory_sum / (initial_sum + 1e-8))
        base_state.append(pressure)

        # 库存不平衡度
        imbalance = float(np.std(relative_inventory))
        base_state.append(imbalance)

        return np.array(base_state, dtype=np.float32)