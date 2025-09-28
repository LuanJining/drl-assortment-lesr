import json
import numpy as np
import re
import time
import logging
import requests
from typing import Dict, List, Tuple, Optional

# 尝试导入OpenAI，如果失败则使用requests作为备选
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI库未安装，将使用requests库")


class LLMOptimizer:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # 移除末尾的斜杠
        self.model = model
        self.iteration_history = []
        self.best_functions = {}

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 初始化客户端
        self.client = None
        self.use_requests = False
        self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        # 首先尝试使用requests库（因为测试显示它能正常工作）
        if self._test_requests_connection():
            self.logger.info("使用requests库进行API调用")
            self.use_requests = True
            return True

        # 如果requests失败，尝试OpenAI客户端
        if OPENAI_AVAILABLE:
            try:
                # 简化的OpenAI客户端初始化，只使用必要参数
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )

                # 测试连接
                if self._test_openai_connection():
                    self.logger.info("使用OpenAI客户端进行API调用")
                    return True
                else:
                    self.client = None

            except Exception as e:
                self.logger.error(f"OpenAI客户端初始化失败: {e}")
                self.client = None

        # 如果都失败，记录警告
        self.logger.warning("所有API连接方式都失败，将使用默认函数")
        return False

    def _test_requests_connection(self):
        """测试requests连接"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Requests连接测试失败: {e}")
            return False

    def _test_openai_connection(self):
        """测试OpenAI客户端连接"""
        try:
            if self.client is None:
                return False

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )

            return True

        except Exception as e:
            self.logger.error(f"OpenAI客户端连接测试失败: {e}")
            return False

    def _make_request_with_requests(self, messages: List[Dict], max_retries: int = 3):
        """使用requests库进行API调用"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    if content:
                        return content
                    else:
                        raise ValueError("API返回空内容")
                else:
                    raise Exception(f"API返回错误状态码: {response.status_code}, 内容: {response.text}")

            except Exception as e:
                self.logger.warning(f"Requests API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise

    def _make_request_with_openai(self, messages: List[Dict], max_retries: int = 3):
        """使用OpenAI客户端进行API调用"""
        if self.client is None:
            raise Exception("OpenAI客户端未初始化")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )

                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    raise ValueError("API返回空内容")

            except Exception as e:
                self.logger.warning(f"OpenAI API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise

    def _make_request(self, messages: List[Dict]):
        """智能选择API调用方式"""
        if self.use_requests:
            return self._make_request_with_requests(messages)
        elif self.client is not None:
            return self._make_request_with_openai(messages)
        else:
            raise Exception("没有可用的API连接方式")

    def generate_state_representation(self,
                                      task_description: str,
                                      state_info: Dict) -> str:
        """生成状态表示函数"""
        # 如果没有可用的API连接，直接返回默认函数
        if not self.use_requests and self.client is None:
            self.logger.warning("API不可用，使用默认状态函数")
            return self._get_default_state_function()

        prompt = self._create_state_prompt(task_description, state_info)
        messages = [{"role": "user", "content": prompt}]

        try:
            content = self._make_request(messages)
            code = self._extract_code(content)

            if code:
                self.logger.info("成功生成状态表示函数")
                return code
            else:
                self.logger.warning("未能从响应中提取代码，使用默认函数")
                return self._get_default_state_function()

        except Exception as e:
            self.logger.error(f"生成状态表示函数失败: {e}")
            self.logger.info("使用默认状态表示函数")
            return self._get_default_state_function()

    def generate_intrinsic_reward(self,
                                  state_representation: str,
                                  performance_feedback: Dict = None) -> str:
        """生成内在奖励函数"""
        # 如果没有可用的API连接，直接返回默认函数
        if not self.use_requests and self.client is None:
            self.logger.warning("API不可用，使用默认奖励函数")
            return self._get_default_reward_function()

        prompt = self._create_reward_prompt(state_representation, performance_feedback)
        messages = [{"role": "user", "content": prompt}]

        try:
            content = self._make_request(messages)
            code = self._extract_code(content)

            if code:
                self.logger.info("成功生成内在奖励函数")
                return code
            else:
                self.logger.warning("未能从响应中提取代码，使用默认函数")
                return self._get_default_reward_function()

        except Exception as e:
            self.logger.error(f"生成内在奖励函数失败: {e}")
            self.logger.info("使用默认奖励函数")
            return self._get_default_reward_function()

    def update_with_feedback(self, feedback: Dict):
        """基于反馈更新优化策略"""
        self.iteration_history.append(feedback)

    def _extract_code(self, response_text: str) -> str:
        """从LLM响应中提取Python代码"""
        if not response_text:
            return ""

        # 查找代码块
        code_pattern = r'```python\nfrom core.json_utils import safe_json_dumps\n(.*?)```'
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

要求：
- 函数名必须是 enhance_state
- 输入参数: inventory, customer_type, prices, time_remaining, initial_inventory
- 返回 numpy.ndarray
- 处理边界情况，避免除零错误
- 确保所有特征都是有意义的数值

请生成一个创新的状态增强函数。
"""
        return prompt

    def _create_reward_prompt(self, state_representation: str, performance_feedback: Dict) -> str:
        """创建奖励函数生成提示"""
        feedback_str = ""
        if performance_feedback:
            feedback_str = f"\n性能反馈：{safe_json_dumps(performance_feedback, indent=2, ensure_ascii=False)}"

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

要求：
- 函数名必须是 intrinsic_reward
- 输入参数: state, action, next_state, sold_item, price
- 返回 float 数值
- 数值稳定，避免异常情况
- 提供有意义的学习信号

请生成一个平衡且有效的内在奖励函数。
"""
        return prompt

    def is_api_available(self):
        """检查API是否可用"""
        return self.use_requests or self.client is not None

    def get_connection_status(self):
        """获取连接状态信息"""
        if self.use_requests:
            return "使用requests库连接"
        elif self.client is not None:
            return "使用OpenAI客户端连接"
        else:
            return "无可用连接，使用默认函数"