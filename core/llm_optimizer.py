import openai
import json
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

class LLMOptimizer:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o-mini"):
        openai.api_key = api_key
        openai.base_url = base_url
        self.model = model
        self.iteration_history = []
        self.best_functions = {}
        
    def generate_state_representation(self, 
                                      task_description: str,
                                      state_info: Dict) -> str:
        """生成状态表示函数"""
        prompt = self._create_state_prompt(task_description, state_info)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            # 确保返回值有效
            content = response['choices'][0].get('message', {}).get('content', '')
            if not content:
                raise ValueError("No content in the response.")
            code = self._extract_code(content)
            return code
        except Exception as e:
            print(f"Error generating state representation: {e}")
            return ""
    
    def generate_intrinsic_reward(self, 
                                  state_representation: str,
                                  performance_feedback: Dict = None) -> str:
        """生成内在奖励函数"""
        prompt = self._create_reward_prompt(state_representation, performance_feedback)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            # 确保返回值有效
            content = response['choices'][0].get('message', {}).get('content', '')
            if not content:
                raise ValueError("No content in the response.")
            code = self._extract_code(content)
            return code
        except Exception as e:
            print(f"Error generating intrinsic reward: {e}")
            return ""
    
    def update_with_feedback(self, feedback: Dict):
        """基于反馈更新优化策略"""
        self.iteration_history.append(feedback)
    
    def _extract_code(self, response_text: str) -> str:
        """从LLM响应中提取Python代码"""
        # 查找代码块
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # 如果没有找到标记的代码块，尝试提取函数定义
        func_pattern = r'(def\s+\w+\s*\([^)]*\):[^}]+)'
        func_matches = re.findall(func_pattern, response_text, re.DOTALL)
        
        if func_matches:
            return '\n'.join(func_matches)
        
        return response_text
    
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

示例格式：
```python
import numpy as np

def enhance_state(inventory, customer_type, prices, time_remaining, initial_inventory):
    # 原始状态
    base_state = []
    
    # 相对库存水平
    relative_inventory = inventory / (initial_inventory + 1e-8)
    base_state.extend(relative_inventory)
    
    # 客户类型编码（one-hot）
    customer_encoding = np.zeros(4)
    customer_encoding[customer_type] = 1
    base_state.extend(customer_encoding)
    
    # 时间特征
    base_state.append(time_remaining / 100.0)
    
    # === 新增特征 ===
    
    # 1. 库存压力指标
    inventory_pressure = 1.0 - (inventory.sum() / initial_inventory.sum())
    
    # 2. 库存不平衡度（标准差）
    inventory_imbalance = np.std(relative_inventory)
    
    # 3. 低库存产品比例
    low_stock_ratio = np.mean(relative_inventory < 0.3)
    
    # 4. 价格加权库存
    if len(prices) == len(inventory):
        weighted_inventory = np.sum(prices * inventory) / (np.sum(prices * initial_inventory) + 1e-8)
    else:
        weighted_inventory = 0.5
    
    # 组合所有特征
    enhanced_features = [
        inventory_pressure,
        inventory_imbalance,
        low_stock_ratio,
        weighted_inventory
    ]
    
    base_state.extend(enhanced_features)
    
    return np.array(base_state, dtype=np.float32)
    请生成类似的函数，但要根据任务特点进行优化。
"""
        return prompt

    def _create_reward_prompt(self, state_representation: str, performance_feedback: Dict) -> str:
        """创建奖励函数生成提示"""
        feedback_str = ""
        if performance_feedback:
            feedback_str = f"\n性能反馈：{json.dumps(performance_feedback, indent=2)}"
        
        prompt = f"""
    基于以下状态表示函数，设计一个内在奖励函数：
{state_representation}
{feedback_str}
请设计一个内在奖励函数，它应该：

鼓励库存平衡
避免缺货
奖励高价值产品的销售
考虑时间因素

示例格式：
def intrinsic_reward(enhanced_state, action, next_state, sold_item, price):
    reward = 0.0
    
    # 提取状态特征
    state_dim = len(enhanced_state)
    inventory_pressure = enhanced_state[-4] if state_dim > 4 else 0
    inventory_imbalance = enhanced_state[-3] if state_dim > 4 else 0
    low_stock_ratio = enhanced_state[-2] if state_dim > 4 else 0
    
    # 1. 库存平衡奖励
    balance_reward = -inventory_imbalance * 0.1
    
    # 2. 避免缺货惩罚
    stockout_penalty = -low_stock_ratio * 0.2
    
    # 3. 销售奖励
    if sold_item >= 0:
        sale_reward = price * 0.1
    else:
        sale_reward = 0
    
    # 4. 库存压力惩罚
    pressure_penalty = -inventory_pressure * 0.05
    
    # 组合奖励
    reward = balance_reward + stockout_penalty + sale_reward + pressure_penalty
    
    return reward
    请生成一个优化的内在奖励函数。
"""
        return prompt
