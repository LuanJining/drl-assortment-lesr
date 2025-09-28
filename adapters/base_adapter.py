# adapters/base_adapter.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np


class BaseAdapter(ABC):
    """基础适配器接口"""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        pass

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        pass

    @abstractmethod
    def _get_info(self) -> Dict:
        """获取环境信息"""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """状态维度"""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """动作维度"""
        pass