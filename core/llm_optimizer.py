from openai import OpenAI
import json
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


class LLMOptimizer:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o-mini"):
        try:
            # 简化的客户端初始化，只传递必要参数
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=60.0  # 设置超时
            )
        except Exception as e:
            print(f"OpenAI客户端初始化失败: {e}")
            print("尝试使用最小配置...")
            # 备用初始化方式
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e2:
                print(f"备用初始化也失败: {e2}")
                self.client = None

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.iteration_history = []
        self.best_functions = {}

    def _test_connection(self):
        """测试API连接"""
        try:
            if self.client is None:
                return False
            # 发送一个简单的测试请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"API连接测试失败: {e}")
            return False

    def generate_state_representation(self,
                                      task_description: str,
                                      state_info: Dict) -> str:
        """生成状态表示函数"""
        # 如果客户端未初始化，直接返回默认函数
        if self.client is None:
            print("OpenAI客户端未初始化，使用默认状态函数")
            return self._get_default_state_function()

        prompt = self._create_state_prompt(task_description, state_info)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                timeout=30.0
            )

            # 提取响应内容
            content = response.choices[0].message.content
            if not content:
                raise ValueError("No content in the response.")

            code = self._extract_code(content)
            return code if code else self._get_default_state_function()

        except Exception as e:
            print(f"Error generating state representation: {e}")
            print("使用默认状态表示函数")
            return self._get_default_state_function()

    def generate_intrinsic_reward(self,
                                  state_representation: str,
                                  performance_feedback: Dict = None) -> str:
        """生成内在奖励函数"""
        # 如果客户端未初始化，直接返回默认函数
        if self.client is None:
            print("OpenAI客户端未初始化，使用默认奖励函数")
            return self._get_default_reward_function()

        prompt = self._create_reward_prompt(state_representation, performance_feedback)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500,
                timeout=30.0
            )

            # 提取响应内容
            content = response.choices[0].message.content
            if not content:
                raise ValueError("No content in the response.")

            code = self._extract_code(content)
            return code if code else self._get_default_reward_function()

        except Exception as e:
            print(f"Error generating intrinsic reward: {e}")
            print("使用默认奖励函数")
            return self._get_default_reward_function()

    def update_with_feedback(self, feedback: Dict):
        """基于反馈更新优化策略"""
        self.iteration_history.append(feedback)

    def _extract_code(self, response_text: str) -> str:
        """从LLM响应中提取Python代码"""
        if not response_text:
            return ""

        # 查找代码块
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # 如果没有找到标记的代码块，尝试提取函数定义
        func_pattern = r'(def\s+\w+\s*\([^)]*\):.*?)(?=\n\n|\n(?:def|class|import|#)|$)'
        func_matches = re.findall(func_pattern, response_text, re.DOTALL)

        if func_matches:
            return '\n\n'.join(func_matches).strip()

        return response_text.strip()

    def _get_default_state_function(self) -> str:
        """获取默认状态增强函数"""
        return """
import numpy as np

def enhance_state(inventory, customer_type, prices, time_remaining, initial_inventory):
    base_state = []

    # 确保输入是numpy数组
    inventory = np.array(inventory, dtype=np.float32)
    initial_inventory = np.array(initial_inventory, dtype=np.float32)
    prices = np.array(prices, dtype=np.float32) if prices is not None else np.ones(len(inventory), dtype=np.float32)

    # 相对库存
    relative_inventory = inventory / (initial_inventory + 1e-8)
    base_state.extend(relative_inventory.tolist())

    # 客户类型one-hot编码
    customer_encoding = np.zeros(4, dtype=np.float32)
    if 0 <= customer_type < 4:
        customer_encoding[int(customer_type)] = 1.0
    base_state.extend(customer_encoding.tolist())

    # 时间特征
    base_state.append(float(time_remaining) / 100.0)

    # 增强特征
    inventory_sum = float(inventory.sum())
    initial_sum = float(initial_inventory.sum())

    # 库存压力
    pressure = 1.0 - (inventory_sum / (initial_sum + 1e-8))
    base_state.append(pressure)

    # 库存不平衡度
    imbalance = float(np.std(relative_inventory))
    base_state.append(imbalance)

    # 低库存产品比例
    low_stock_ratio = float(np.mean(relative_inventory < 0.3))
    base_state.append(low_stock_ratio)

    # 价格加权库存
    if len(prices) == len(inventory):
        weighted_inventory = float(np.sum(prices * inventory)) / (float(np.sum(prices * initial_inventory)) + 1e-8)
    else:
        weighted_inventory = 0.5
    base_state.append(weighted_inventory)

    return np.array(base_state, dtype=np.float32)
        """.strip()

    def _get_default_reward_function(self) -> str:
        """获取默认奖励函数"""
        return """
import numpy as np

def intrinsic_reward(state, action, next_state, sold_item, price):
    reward = 0.0

    try:
        # 销售奖励
        if sold_item >= 0 and price > 0:
            reward += float(price) * 0.1

        # 库存平衡奖励
        if len(state) > 10:
            inventory_features = state[:10]
            inventory_std = float(np.std(inventory_features))
            reward -= inventory_std * 0.05

        # 时间压力奖励
        if len(state) > 14:
            time_remaining = float(state[14])
            reward += (1.0 - time_remaining) * 0.02

        # 确保奖励在合理范围内
        reward = np.clip(reward, -10.0, 10.0)

    except Exception as e:
        # 如果计算失败，返回小的正奖励
        reward = 0.01

    return float(reward)
        """.strip()

    def _create_state_prompt(self, task_description: str, state_info: Dict) -> str:
        """创建状态表示生成提示"""
        prompt = f"""
任务描述：{task_description}

当前状态信息：
- 库存水平：{state_info['inventory_shape']}维向量，表示每个产品的剩余数量
- 客户类型：{state_info['customer_types']}种不同类型的客户
- 产品数量：{state_info['num_products']}个不同产品
- 展示限制：每次最多展示{state_info.get('cardinality', 4)}个产品

请设计一个Python函数来增强状态表示。函数应该：
1. 计算库存压力指标（如相对库存水平、库存不平衡度）
2. 评估库存的紧急程度
3. 计算预期的未来价值
4. 评估库存分布的均衡性

要求返回一个包含原始状态和新增特征的numpy数组。

请生成一个创新的状态增强函数，确保：
- 函数逻辑正确，避免除零错误
- 包含有意义的特征工程
- 返回numpy数组格式
- 处理边界情况
"""
        return prompt

    def _create_reward_prompt(self, state_representation: str, performance_feedback: Dict) -> str:
        """创建奖励函数生成提示"""
        feedback_str = ""
        if performance_feedback:
            feedback_str = f"\n性能反馈：{json.dumps(performance_feedback, indent=2, ensure_ascii=False)}"

        prompt = f"""
基于以下状态表示函数，设计一个内在奖励函数：

{state_representation}
{feedback_str}

请设计一个内在奖励函数，它应该：
1. 鼓励库存平衡
2. 避免缺货情况
3. 奖励高价值产品的销售
4. 考虑时间压力因素
5. 提供稳定的学习信号

请生成一个平衡且有效的内在奖励函数，确保：
- 数值稳定
- 逻辑清晰
- 避免异常情况
- 提供有意义的学习信号
"""
        return prompt