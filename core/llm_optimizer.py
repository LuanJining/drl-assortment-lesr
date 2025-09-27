import openai
import json
import numpy as np
from typing import Dict, List, Tuple

class LLMOptimizer:
    def __init__(self, api_key: str, model: str = "gpt-4-1106-preview"):
        openai.api_key = api_key
        self.model = model
        self.iteration_history = []
    
    def generate_state_representation(self, 
                                     task_description: str,
                                     state_info: Dict) -> str:
        """生成状态表示函数"""
        prompt = self._create_state_prompt(task_description, state_info)
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        code = self._extract_code(response['choices'][0]['message']['content'])
        return code
    
    def generate_intrinsic_reward(self, 
                                 state_representation: str,
                                 performance_feedback: Dict) -> str:
        """生成内在奖励函数"""
        prompt = self._create_reward_prompt(state_representation, performance_feedback)
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        code = self._extract_code(response['choices'][0]['message']['content'])
        return code
    
    def _create_state_prompt(self, task_description: str, state_info: Dict) -> str:
        """创建状态表示生成提示"""
        prompt = f"""
        任务描述：{task_description}
        
        当前状态信息：
        - 库存水平：{state_info['inventory_shape']}维向量
        - 客户类型：{state_info['customer_types']}种
        - 产品价格：{state_info['num_products']}个产品
        
        请设计一个Python函数来增强状态表示，考虑：
        1. 库存压力指标（如相对库存水平）
        2. 客户-产品匹配度
        3. 预期未来收益
        4. 库存平衡度量
        
        返回格式：
```python
        def enhance_state(inventory, customer_type, prices):
            # 你的实现
            return enhanced_state
            """
    return prompt

    