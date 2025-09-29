import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt


class FeedbackAnalyzer:
    def __init__(self):
        self.performance_history = []
        self.lipschitz_history = []

    def analyze_performance(self, performances: List[float],
                            state_functions: List[str],
                            lipschitz_constants: List[np.ndarray] = None) -> Dict[str, Any]:
        """分析性能并生成反馈"""

        feedback = {
            'num_samples': len(performances),
            'best_performance': max(performances),
            'worst_performance': min(performances),
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'best_idx': np.argmax(performances)
        }

        # 分析Lipschitz常数
        if lipschitz_constants:
            feedback['lipschitz_analysis'] = self._analyze_lipschitz(lipschitz_constants)

        # 识别关键特征
        if len(performances) > 1:
            feedback['key_insights'] = self._extract_insights(performances, state_functions)

        self.performance_history.append(feedback)

        return feedback

    def _analyze_lipschitz(self, constants: List[np.ndarray]) -> Dict:
        """分析Lipschitz常数"""
        analysis = {}

        if constants:
            # 找出最稳定的维度
            mean_constants = np.mean([c for c in constants], axis=0)
            analysis['most_stable_dims'] = np.argsort(mean_constants)[:5].tolist()
            analysis['most_unstable_dims'] = np.argsort(mean_constants)[-5:].tolist()
            analysis['mean_stability'] = float(np.mean(mean_constants))

        return analysis

    def _extract_insights(self, performances: List[float],
                          functions: List[str]) -> List[str]:
        """提取关键洞察"""
        insights = []

        # 性能差异分析
        perf_range = max(performances) - min(performances)
        if perf_range > np.mean(performances) * 0.3:
            insights.append("性能差异较大，说明状态表示对性能影响显著")

        # 最佳性能分析
        best_idx = np.argmax(performances)
        if performances[best_idx] > np.mean(performances) * 1.2:
            insights.append(f"样本{best_idx + 1}性能显著优于平均水平")

        return insights

    def plot_history(self):
        """绘制性能历史"""
        if not self.performance_history:
            print("没有历史数据")
            return

        iterations = range(len(self.performance_history))
        best_perfs = [h['best_performance'] for h in self.performance_history]
        mean_perfs = [h['mean_performance'] for h in self.performance_history]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_perfs, 'b-', label='Best Performance')
        plt.plot(iterations, mean_perfs, 'r--', label='Mean Performance')
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.title('Performance History')
        plt.legend()
        plt.grid(True)
        plt.show()

