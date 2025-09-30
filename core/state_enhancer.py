import numpy as np
from typing import Dict, Any, Callable
import importlib.util
import tempfile
import os
import ast


class StateEnhancer:
    def __init__(self):
        self.enhance_func = None
        self.original_dim = None
        self.enhanced_dim = None
        self._error_count = 0
        self._max_errors = 3  # 只打印前3次错误

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

    def load_function(self, function_code: str):
        """动态加载状态增强函数"""
        # 🔧 新增：提取纯函数代码，移除所有示例代码
        function_code = self._extract_function_only(function_code, 'enhance_state')

        # 验证代码（警告模式）
        if not self._validate_code(function_code):
            print("警告：代码验证发现问题，但仍尝试加载")

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
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理临时文件
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

    def _validate_code(self, code: str) -> bool:
        """验证代码的基本安全性和正确性"""
        try:
            tree = ast.parse(code)
            has_function = False

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'enhance_state':
                    has_function = True

                    # 检查参数数量（应该是5个）
                    if len(node.args.args) != 5:
                        print(f"警告：enhance_state 应该有5个参数，实际有 {len(node.args.args)} 个")

                    # 扩展允许的名称集合
                    defined_names = {
                        # 模块和参数
                        'np', 'numpy',
                        'inventory', 'customer_type', 'prices', 'time_remaining', 'initial_inventory',

                        # Python内置函数
                        'sum', 'max', 'min', 'abs', 'len', 'range',
                        'float', 'int', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
                        'enumerate', 'zip', 'map', 'filter', 'sorted',
                        'round', 'pow', 'all', 'any', 'print',

                        # 常见的numpy函数
                        'array', 'zeros', 'ones', 'mean', 'std', 'sqrt', 'exp', 'log',
                        'clip', 'maximum', 'minimum', 'concatenate', 'stack',
                        'reshape', 'flatten', 'squeeze', 'transpose', 'extend', 'append',
                        'tolist', 'astype', 'copy'
                    }

                    # 检查未定义的变量（仅警告）
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Name):
                            if subnode.id not in defined_names and not subnode.id.startswith('_'):
                                if not self._is_local_variable(subnode.id, node):
                                    print(f"警告：代码可能引用了未定义的变量: {subnode.id}")

            if not has_function:
                print("错误：未找到 enhance_state 函数")
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

    def enhance(self, state: Dict[str, Any]) -> np.ndarray:
        """增强状态 - 带参数类型保护"""
        if self.enhance_func is None:
            return self._default_enhance(state)

        try:
            # 参数类型保护和转换
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
                import traceback
                traceback.print_exc()
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