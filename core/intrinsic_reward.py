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

    def _extract_function_only(self, code: str, function_name: str) -> str:
        """提取函数定义，移除所有示例/测试代码"""
        import re

        try:
            # 首先清理已弃用的类型
            code = self._clean_code(code)

            # 解析代码为AST
            tree = ast.parse(code)

            # 找到目标函数
            target_function = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    target_function = node
                    break

            if target_function is None:
                raise ValueError(f"未找到函数 {function_name}")

            # 获取函数的源代码行范围
            function_start = target_function.lineno - 1  # 转换为0索引
            function_end = target_function.end_lineno if hasattr(target_function, 'end_lineno') else None

            lines = code.split('\n')

            # 提取函数定义部分
            if function_end is not None:
                function_lines = lines[function_start:function_end]
            else:
                # 如果没有end_lineno，手动查找函数结束位置
                function_lines = []
                in_function = False
                indent_level = None

                for i, line in enumerate(lines[function_start:], start=function_start):
                    stripped = line.lstrip()

                    # 函数定义开始
                    if f'def {function_name}' in line:
                        in_function = True
                        indent_level = len(line) - len(stripped)
                        function_lines.append(line)
                        continue

                    if not in_function:
                        continue

                    # 空行或注释，继续
                    if not stripped or stripped.startswith('#'):
                        function_lines.append(line)
                        continue

                    # 检查缩进级别
                    current_indent = len(line) - len(stripped)

                    # 如果缩进回到函数级别或更小，函数结束
                    if current_indent <= indent_level:
                        break

                    function_lines.append(line)

            # 重新组合代码
            clean_code = '\n'.join(function_lines)

            # 确保有numpy导入
            if 'import numpy' not in clean_code:
                clean_code = 'import numpy as np\n\n' + clean_code

            return clean_code

        except Exception as e:
            print(f"AST提取函数失败: {e}")
            # 降级方案：使用正则表达式
            return self._extract_function_regex(code, function_name)

    def _extract_function_regex(self, code: str, function_name: str) -> str:
        """使用正则表达式提取函数（降级方案）"""
        import re

        # 清理代码
        code = self._clean_code(code)

        lines = code.split('\n')
        function_lines = []
        in_function = False
        function_indent = None

        for line in lines:
            stripped = line.strip()

            # 检测函数定义
            if f'def {function_name}' in line and '(' in line and ':' in line:
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                function_lines.append(line)
                continue

            if not in_function:
                # 保留导入语句
                if stripped.startswith('import ') or stripped.startswith('from '):
                    function_lines.insert(0, line)
                continue

            # 在函数内部
            current_indent = len(line) - len(line.lstrip())

            # 空行或注释
            if not stripped or stripped.startswith('#'):
                function_lines.append(line)
                continue

            # 如果缩进回到函数级别或更小，停止
            if current_indent <= function_indent:
                break

            function_lines.append(line)

        if not function_lines:
            print(f"警告：无法提取函数 {function_name}，使用原始代码")
            return code

        clean_code = '\n'.join(function_lines)

        if 'import numpy' not in clean_code:
            clean_code = 'import numpy as np\n\n' + clean_code

        return clean_code

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

                    # 扩展允许的名称集合，包含Python内置函数和常用numpy函数
                    defined_names = {
                        # 模块和参数
                        'np', 'numpy', 'state', 'action', 'next_state', 'sold_item', 'price',

                        # Python内置函数
                        'sum', 'max', 'min', 'abs', 'len', 'range',
                        'float', 'int', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
                        'enumerate', 'zip', 'map', 'filter', 'sorted',
                        'round', 'pow', 'all', 'any', 'print',

                        # 常见的numpy函数和属性
                        'array', 'zeros', 'ones', 'mean', 'std', 'sqrt', 'exp', 'log',
                        'clip', 'maximum', 'minimum', 'concatenate', 'stack',
                        'reshape', 'flatten', 'squeeze', 'transpose'
                    }

                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Name):
                            if subnode.id not in defined_names and not subnode.id.startswith('_'):
                                # 检查是否是在函数内定义的局部变量
                                if not self._is_local_variable(subnode.id, node):
                                    print(f"警告：代码引用了未定义的变量: {subnode.id}")

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
        # 🔧 新增：提取纯函数代码，移除所有示例代码
        function_code = self._extract_function_only(function_code, 'intrinsic_reward')

        # 验证代码（警告模式）
        self._validate_code(function_code)

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

    def _clean_code(self, code: str) -> str:
        """清理生成的代码，修复常见问题"""
        import re

        # 使用正则表达式精确替换，避免重复替换
        # 只替换 np.float 但不替换 np.float32/np.float64 等
        code = re.sub(r'\bnp\.float\b(?!32|64|16)', 'np.float64', code)
        code = re.sub(r'\bnp\.int\b(?!8|16|32|64)', 'np.int64', code)
        code = re.sub(r'\bnp\.bool\b(?!_)', 'np.bool_', code)

        # 同样处理 numpy.xxx 格式
        code = re.sub(r'\bnumpy\.float\b(?!32|64|16)', 'numpy.float64', code)
        code = re.sub(r'\bnumpy\.int\b(?!8|16|32|64)', 'numpy.int64', code)
        code = re.sub(r'\bnumpy\.bool\b(?!_)', 'numpy.bool_', code)

        return code

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