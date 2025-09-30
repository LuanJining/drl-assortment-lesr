import numpy as np
import importlib.util
import tempfile
import os
import ast
from typing import Any


class IntrinsicRewardCalculator:
    def __init__(self):
        self.reward_func = None
        self._error_printed = False

    def _validate_code(self, code: str) -> bool:
        """验证代码的基本安全性和正确性"""
        try:
            # 解析代码为AST
            tree = ast.parse(code)

            # 检查是否有intrinsic_reward函数
            has_function = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'intrinsic_reward':
                    has_function = True

                    # 检查参数数量
                    if len(node.args.args) != 5:
                        print(f"警告：intrinsic_reward 应该有5个参数，实际有 {len(node.args.args)} 个")

                    # 检查是否有未定义的全局变量引用
                    defined_names = {'np', 'numpy', 'state', 'action', 'next_state', 'sold_item', 'price'}

                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Name):
                            if subnode.id not in defined_names and not subnode.id.startswith('_'):
                                # 检查是否是在函数内定义的局部变量
                                if not self._is_local_variable(subnode.id, node):
                                    print(f"警告：代码引用了未定义的变量: {subnode.id}")
                                    return False

            if not has_function:
                print("错误：未找到 intrinsic_reward 函数")
                return False

            return True

        except SyntaxError as e:
            print(f"代码语法错误: {e}")
            return False
        except Exception as e:
            print(f"代码验证失败: {e}")
            return False

    def _is_local_variable(self, var_name: str, func_node: ast.FunctionDef) -> bool:
        """检查变量是否在函数内定义"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return True
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == var_name:
                    return True
        return False

    def load_function(self, function_code: str):
        """动态加载奖励函数"""
        # 确保代码包含 numpy 导入
        if 'import numpy' not in function_code:
            function_code = 'import numpy as np\n\n' + function_code

        # 移除调试 print 语句
        lines = function_code.split('\n')
        filtered_lines = []
        in_function = False
        for line in lines:
            if 'def intrinsic_reward' in line:
                in_function = True
            # 跳过函数内的 print 语句，但保留其他内容
            if in_function and 'print(' in line and not line.strip().startswith('#'):
                # 用注释替换print语句
                filtered_lines.append('    # ' + line.strip() + '  # [print removed]')
                continue
            filtered_lines.append(line)
        function_code = '\n'.join(filtered_lines)

        # 验证代码
        if not self._validate_code(function_code):
            print("代码验证失败，将使用默认奖励函数")
            return False

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(function_code)
            temp_file = f.name

        try:
            spec = importlib.util.spec_from_file_location("reward_module", temp_file)
            module = importlib.util.module_from_spec(spec)

            # 在模块的全局命名空间中注入 numpy
            module.__dict__['np'] = np
            module.__dict__['numpy'] = np

            spec.loader.exec_module(module)

            if hasattr(module, 'intrinsic_reward'):
                self.reward_func = module.intrinsic_reward
                self._error_printed = False  # 重置错误标志
                return True
            else:
                print("错误：未找到intrinsic_reward函数")
                return False

        except Exception as e:
            print(f"加载奖励函数失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def calculate(self, state: np.ndarray, action: Any,
                  next_state: np.ndarray, sold_item: int, price: float) -> float:
        """计算内在奖励"""
        if self.reward_func is None:
            return self._default_reward(state, action, next_state, sold_item, price)

        try:
            # 确保输入参数的类型正确
            state = self._ensure_array(state)
            next_state = self._ensure_array(next_state)

            # action 处理
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action_value = int(action.item())
                elif action.size == 1:
                    action_value = int(action.flatten()[0])
                else:
                    # 多元素数组，取第一个非零索引
                    nonzero = np.where(action > 0)[0]
                    action_value = int(nonzero[0]) if len(nonzero) > 0 else -1
            else:
                action_value = int(action) if action is not None else -1

            # sold_item 处理
            if isinstance(sold_item, np.ndarray):
                sold_item = int(sold_item.item())
            else:
                sold_item = int(sold_item) if sold_item is not None else -1

            # price 处理
            if isinstance(price, np.ndarray):
                price = float(price.item())
            else:
                price = float(price) if price is not None else 0.0

            # 调用奖励函数
            reward = self.reward_func(state, action_value, next_state, sold_item, price)

            # 确保返回值是标量
            if isinstance(reward, np.ndarray):
                reward = float(reward.item())

            return float(reward)

        except Exception as e:
            if not self._error_printed:
                print(f"奖励计算失败，使用默认方法: {e}")
                self._error_printed = True
            return self._default_reward(state, action, next_state, sold_item, price)

    def _ensure_array(self, value: Any) -> np.ndarray:
        """确保值是numpy数组"""
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, (list, tuple)):
            return np.array(value, dtype=np.float32)
        else:
            return np.array([value], dtype=np.float32)

    def _default_reward(self, state: np.ndarray, action: Any,
                        next_state: np.ndarray, sold_item: int, price: float) -> float:
        """默认奖励函数"""
        try:
            reward = 0.0

            # 确保参数是标量
            if isinstance(sold_item, np.ndarray):
                sold_item = int(sold_item.item())
            if isinstance(price, np.ndarray):
                price = float(price.item())

            # 销售奖励
            if sold_item >= 0 and price > 0:
                reward += float(price) * 0.1

            # 库存平衡奖励
            state = self._ensure_array(state)
            if len(state) > 10:
                inventory_features = state[:10]
                inventory_std = float(np.std(inventory_features))
                reward -= inventory_std * 0.05

            return float(reward)

        except Exception:
            return 0.01