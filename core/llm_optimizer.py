def _get_default_reward_function(self) -> str:
    """获取默认奖励函数 - 修复action参数处理"""
    return """
import numpy as np

def intrinsic_reward(state, action, next_state, sold_item, price):
    reward = 0.0

    try:
        # 处理action参数，支持标量和数组
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action_processed = int(action.item())
            else:
                # 如果是多维动作，计算选中的动作数量
                num_selected = np.sum(action > 0.5)
                action_processed = int(num_selected)
        else:
            action_processed = int(action) if action is not None else 0

        # 安全转换其他参数
        sold_item = int(sold_item) if sold_item is not None else -1
        price = float(price) if price is not None else 0.0

        # 销售奖励
        if sold_item >= 0 and price > 0:
            reward += price * 0.1

        # 库存平衡奖励
        if len(state) > 10:
            inventory_features = state[:10]
            if len(inventory_features) > 1:
                inventory_std = float(np.std(inventory_features))
                reward -= inventory_std * 0.05

        # 动作多样性奖励
        if isinstance(action, np.ndarray) and len(action) > 1:
            num_selected = np.sum(action > 0.5)
            if num_selected > 1:
                reward += 0.01 * num_selected

        # 时间压力奖励
        if len(state) > 14:
            time_remaining = float(state[14])
            if 0 <= time_remaining <= 1:
                reward += (1.0 - time_remaining) * 0.02

        # 确保奖励在合理范围内
        reward = np.clip(reward, -10.0, 10.0)

    except Exception as e:
        # 如果计算失败，返回小的正奖励
        reward = 0.01

    return float(reward)
        """.strip()