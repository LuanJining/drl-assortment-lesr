import numpy as np
import torch
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Union
import random
import pickle
from dataclasses import dataclass


@dataclass
class Transition:
    """转移数据结构"""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict = None


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state: np.ndarray, action: Union[int, np.ndarray],
             reward: float, next_state: np.ndarray, done: bool,
             info: dict = None):
        """添加经验到缓冲区"""
        transition = Transition(
            state=state.copy() if isinstance(state, np.ndarray) else state,
            action=action.copy() if isinstance(action, np.ndarray) else action,
            reward=reward,
            next_state=next_state.copy() if isinstance(next_state, np.ndarray) else next_state,
            done=done,
            info=info
        )

        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """随机采样一批经验"""
        return random.sample(self.buffer, batch_size)

    def sample_tensors(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """采样并返回张量格式的数据"""
        batch = self.sample(batch_size)

        states = torch.FloatTensor([t.state for t in batch]).to(device)
        actions = torch.LongTensor([t.action for t in batch]).to(device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(device)
        dones = torch.BoolTensor([t.done for t in batch]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够的样本用于训练"""
        return len(self.buffer) >= batch_size

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

    def save(self, filepath: str):
        """保存缓冲区到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filepath: str):
        """从文件加载缓冲区"""
        with open(filepath, 'rb') as f:
            transitions = pickle.load(f)
            self.buffer = deque(transitions, maxlen=self.capacity)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: Union[int, np.ndarray],
             reward: float, next_state: np.ndarray, done: bool,
             priority: Optional[float] = None):
        """添加经验到优先级缓冲区"""
        transition = Transition(
            state=state.copy(),
            action=action.copy() if isinstance(action, np.ndarray) else action,
            reward=reward,
            next_state=next_state.copy(),
            done=done
        )

        if priority is None:
            priority = self.max_priority

        self.buffer.append(transition)
        self.priorities.append(priority ** self.alpha)

        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """基于优先级采样"""
        priorities_array = np.array(self.priorities)
        probabilities = priorities_array / priorities_array.sum()

        # 计算重要性采样权重
        weights = (len(self.buffer) * probabilities) ** (-self.beta)
        weights = weights / weights.max()

        # 按概率采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # 获取对应的转移
        transitions = [self.buffer[idx] for idx in indices]
        sample_weights = weights[indices]

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return transitions, indices, sample_weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新采样经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-8) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class EpisodeBuffer:
    """Episode缓冲区 - 用于存储完整的episode"""

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)

    def add_episode(self, states: List[np.ndarray], actions: List[Union[int, np.ndarray]],
                    rewards: List[float], dones: List[bool], infos: List[dict] = None):
        """添加完整的episode"""
        if infos is None:
            infos = [{}] * len(states)

        episode = {
            'states': [s.copy() for s in states],
            'actions': [a.copy() if isinstance(a, np.ndarray) else a for a in actions],
            'rewards': rewards.copy(),
            'dones': dones.copy(),
            'infos': infos.copy(),
            'length': len(states),
            'total_reward': sum(rewards)
        }

        self.episodes.append(episode)

    def sample_episodes(self, num_episodes: int) -> List[dict]:
        """随机采样episodes"""
        return random.sample(list(self.episodes), min(num_episodes, len(self.episodes)))

    def get_recent_episodes(self, num_episodes: int) -> List[dict]:
        """获取最近的episodes"""
        return list(self.episodes)[-num_episodes:]

    def get_best_episodes(self, num_episodes: int) -> List[dict]:
        """获取奖励最高的episodes"""
        sorted_episodes = sorted(self.episodes, key=lambda x: x['total_reward'], reverse=True)
        return sorted_episodes[:num_episodes]

    def __len__(self):
        return len(self.episodes)


class TrajectoryBuffer:
    """轨迹缓冲区 - 用于策略梯度算法"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def push(self, state: np.ndarray, action: Union[int, np.ndarray],
             reward: float, log_prob: float, value: float, done: bool):
        """添加一步经验"""
        self.states.append(state.copy())
        self.actions.append(action.copy() if isinstance(action, np.ndarray) else action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95,
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """计算回报和优势"""
        returns = []
        advantages = []

        # 计算GAE优势
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[i + 1]

            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)

        # 计算回报
        advantages = np.array(advantages)
        returns = advantages + np.array(self.values)

        # 标准化优势
        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def get_batch(self, device: str = 'cpu') -> dict:
        """获取整个轨迹的批次数据"""
        returns, advantages = self.compute_returns()

        batch = {
            'states': torch.FloatTensor(self.states).to(device),
            'actions': torch.LongTensor(self.actions).to(device),
            'rewards': torch.FloatTensor(self.rewards).to(device),
            'returns': torch.FloatTensor(returns).to(device),
            'advantages': torch.FloatTensor(advantages).to(device),
            'log_probs': torch.FloatTensor(self.log_probs).to(device),
            'values': torch.FloatTensor(self.values).to(device)
        }

        return batch

    def clear(self):
        """清空轨迹"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class MultiStepBuffer:
    """多步回报缓冲区"""

    def __init__(self, capacity: int = 10000, n_steps: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma

        self.buffer = deque(maxlen=capacity)
        self.temp_buffer = deque(maxlen=n_steps)

    def push(self, state: np.ndarray, action: Union[int, np.ndarray],
             reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        transition = Transition(state, action, reward, next_state, done)
        self.temp_buffer.append(transition)

        if len(self.temp_buffer) == self.n_steps or done:
            # 计算多步回报
            n_step_reward = 0
            for i, t in enumerate(self.temp_buffer):
                n_step_reward += (self.gamma ** i) * t.reward

            # 创建n步转移
            first_transition = self.temp_buffer[0]
            last_transition = self.temp_buffer[-1]

            n_step_transition = Transition(
                state=first_transition.state,
                action=first_transition.action,
                reward=n_step_reward,
                next_state=last_transition.next_state,
                done=last_transition.done
            )

            self.buffer.append(n_step_transition)

            if done:
                self.temp_buffer.clear()

    def sample(self, batch_size: int) -> List[Transition]:
        """采样批次"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class CircularBuffer:
    """循环缓冲区 - 固定大小，高效的FIFO"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def push(self, state: np.ndarray, action: Union[int, np.ndarray],
             reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        self.states[self.ptr] = state
        if isinstance(action, np.ndarray):
            self.actions[self.ptr] = action
        else:
            self.actions[self.ptr] = [action]
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """采样批次数据"""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def sample_tensors(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """采样并返回张量"""
        batch = self.sample(batch_size)

        return (
            torch.FloatTensor(batch[0]).to(device),
            torch.FloatTensor(batch[1]).to(device),
            torch.FloatTensor(batch[2]).to(device),
            torch.FloatTensor(batch[3]).to(device),
            torch.BoolTensor(batch[4]).to(device)
        )

    def __len__(self):
        return self.size