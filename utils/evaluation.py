import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class Evaluator:
    """评估和分析工具"""
    
    def __init__(self):
        self.metrics_history = []
        self.episode_data = []
        
    def evaluate_policy(self, env, agent, num_episodes: int = 100,
                        verbose: bool = False) -> Dict[str, float]:
        """评估策略性能"""
        
        metrics = {
            'total_revenue': [],
            'avg_revenue_per_step': [],
            'stockout_rate': [],
            'inventory_balance': [],
            'customer_satisfaction': [],
            'inventory_turnover': [],
            'final_inventory_value': []
        }
        
        for episode in range(num_episodes):
            # 重置环境
            state, info = env.reset()
            episode_revenue = 0
            stockouts = 0
            steps = 0
            sales = []
            
            done = False
            while not done:
                # 获取动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 记录指标
                episode_revenue += reward
                if info.get('sold_item', -1) == -1 and np.sum(action) > 0:
                    stockouts += 1
                
                sales.append(info.get('sold_item', -1))
                
                state = next_state
                steps += 1
            
            # 计算episode级别的指标
            metrics['total_revenue'].append(episode_revenue)
            metrics['avg_revenue_per_step'].append(episode_revenue / max(steps, 1))
            metrics['stockout_rate'].append(stockouts / max(steps, 1))
            
            # 库存平衡度（标准差越小越平衡）
            final_inventory = info.get('inventory', np.zeros(10))
            inventory_balance = 1.0 / (1.0 + np.std(final_inventory))
            metrics['inventory_balance'].append(inventory_balance)
            
            # 客户满意度（购买成功率）
            satisfaction = len([s for s in sales if s >= 0]) / max(len(sales), 1)
            metrics['customer_satisfaction'].append(satisfaction)
            
            # 库存周转率
            initial_inv = info.get('initial_inventory', np.ones(10) * 10)
            sold_items = initial_inv - final_inventory
            turnover = np.sum(sold_items) / np.sum(initial_inv)
            metrics['inventory_turnover'].append(turnover)
            
            # 剩余库存价值
            prices = info.get('prices', np.ones(10))
            final_value = np.sum(final_inventory * prices)
            metrics['final_inventory_value'].append(final_value)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - Revenue: {episode_revenue:.2f}")
        
        # 计算统计量
        results = {}
        for key, values in metrics.items():
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
            results[f'{key}_min'] = np.min(values)
            results[f'{key}_max'] = np.max(values)
            results[f'{key}_median'] = np.median(values)
        
        return results
    
    def compare_algorithms(self, env, algorithms: Dict[str, Any], 
                          num_episodes: int = 100) -> pd.DataFrame:
        """比较多个算法"""
        
        comparison_results = []
        
        for name, agent in algorithms.items():
            print(f"评估 {name}...")
            results = self.evaluate_policy(env, agent, num_episodes)
            results['algorithm'] = name
            comparison_results.append(results)
        
        # 转换为DataFrame
        df = pd.DataFrame(comparison_results)
        
        # 计算相对性能
        baseline = df[df['algorithm'] == 'Random']['total_revenue_mean'].values[0]
        df['relative_performance'] = df['total_revenue_mean'] / baseline
        
        return df
    
    def calculate_lipschitz_constant(self, states: np.ndarray, 
                                     rewards: np.ndarray) -> np.ndarray:
        """计算Lipschitz常数"""
        
        num_dims = states.shape[1]
        lipschitz_constants = np.zeros(num_dims)
        
        for dim in range(num_dims):
            # 按维度排序
            sorted_indices = np.argsort(states[:, dim])
            sorted_states = states[sorted_indices, dim]
            sorted_rewards = rewards[sorted_indices]
            
            # 计算相邻点的差分
            state_diffs = np.diff(sorted_states)
            reward_diffs = np.abs(np.diff(sorted_rewards))
            
            # 避免除零
            valid_indices = state_diffs > 1e-8
            if np.any(valid_indices):
                ratios = reward_diffs[valid_indices] / state_diffs[valid_indices]
                lipschitz_constants[dim] = np.max(ratios)
            else:
                lipschitz_constants[dim] = 0
        
        return lipschitz_constants
    
    def analyze_state_importance(self, trajectories: List[Dict]) -> Dict[str, float]:
        """分析状态维度的重要性"""
        
        # 收集所有状态和奖励
        all_states = []
        all_rewards = []
        
        for traj in trajectories:
            all_states.extend(traj['states'])
            all_rewards.extend(traj['rewards'])
        
        states = np.array(all_states)
        rewards = np.array(all_rewards)
        
        # 计算相关性
        correlations = {}
        for i in range(states.shape[1]):
            corr = np.corrcoef(states[:, i], rewards)[0, 1]
            correlations[f'dim_{i}_correlation'] = abs(corr)
        
        # 计算互信息（简化版）
        mutual_info = {}
        for i in range(states.shape[1]):
            # 将状态维度离散化
            bins = np.histogram_bin_edges(states[:, i], bins=10)
            digitized = np.digitize(states[:, i], bins)
            
            # 计算条件熵
            unique_vals = np.unique(digitized)
            conditional_entropy = 0
            
            for val in unique_vals:
                mask = digitized == val
                if np.sum(mask) > 0:
                    subset_rewards = rewards[mask]
                    if len(subset_rewards) > 1:
                        # 计算子集的熵
                        hist, _ = np.histogram(subset_rewards, bins=10)
                        hist = hist / hist.sum()
                        hist = hist[hist > 0]
                        entropy = -np.sum(hist * np.log(hist))
                        conditional_entropy += entropy * (np.sum(mask) / len(rewards))
            
            # 总熵
            total_hist, _ = np.histogram(rewards, bins=10)
            total_hist = total_hist / total_hist.sum()
            total_hist = total_hist[total_hist > 0]
            total_entropy = -np.sum(total_hist * np.log(total_hist))
            
            # 互信息 = 总熵 - 条件熵
            mutual_info[f'dim_{i}_mutual_info'] = total_entropy - conditional_entropy
        
        # 合并结果
        importance = {**correlations, **mutual_info}
        
        return importance
    
    def generate_report(self, results: Dict, output_path: str = "evaluation_report.html"):
        """生成评估报告"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DRL-Assortment-LESR 评估报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 1px solid #ccc; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-card { 
                    background: #f9f9f9; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                }
                .good { color: green; font-weight: bold; }
                .bad { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
        """
        
        html_content += "<h1>📊 DRL-Assortment-LESR 评估报告</h1>"
        
        # 添加摘要
        html_content += "<div class='metric-card'>"
        html_content += "<h2>📈 性能摘要</h2>"
        
        if 'total_revenue_mean' in results:
            revenue = results['total_revenue_mean']
            html_content += f"<p>平均总收益: <span class='good'>{revenue:.2f}</span></p>"
        
        if 'customer_satisfaction_mean' in results:
            satisfaction = results['customer_satisfaction_mean']
            color = 'good' if satisfaction > 0.7 else 'bad'
            html_content += f"<p>客户满意度: <span class='{color}'>{satisfaction:.2%}</span></p>"
        
        if 'stockout_rate_mean' in results:
            stockout = results['stockout_rate_mean']
            color = 'good' if stockout < 0.1 else 'bad'
            html_content += f"<p>缺货率: <span class='{color}'>{stockout:.2%}</span></p>"
        
        html_content += "</div>"
        
        # 添加详细指标表
        html_content += "<h2>📋 详细指标</h2>"
        html_content += "<table>"
        html_content += "<tr><th>指标</th><th>均值</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>"
        
        metric_names = {
            'total_revenue': '总收益',
            'avg_revenue_per_step': '平均步收益',
            'stockout_rate': '缺货率',
            'inventory_balance': '库存平衡度',
            'customer_satisfaction': '客户满意度',
            'inventory_turnover': '库存周转率',
            'final_inventory_value': '剩余库存价值'
        }
        
        for metric_key, metric_name in metric_names.items():
            if f'{metric_key}_mean' in results:
                html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{results[f'{metric_key}_mean']:.4f}</td>
                    <td>{results.get(f'{metric_key}_std', 0):.4f}</td>
                    <td>{results.get(f'{metric_key}_min', 0):.4f}</td>
                    <td>{results.get(f'{metric_key}_max', 0):.4f}</td>
                </tr>
                """
        
        html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"报告已保存到: {output_path}")
    
    def plot_learning_curve(self, episode_rewards: List[float], 
                           window_size: int = 100,
                           save_path: Optional[str] = None):
        """绘制学习曲线"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 原始奖励
        ax1.plot(episode_rewards, alpha=0.3, color='blue')
        
        # 移动平均
        if len(episode_rewards) >= window_size:
            moving_avg = pd.Series(episode_rewards).rolling(window=window_size).mean()
            ax1.plot(moving_avg, color='red', linewidth=2, 
                    label=f'移动平均 (窗口={window_size})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('总收益')
        ax1.set_title('学习曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 收益分布
        ax2.hist(episode_rewards, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(episode_rewards), color='red', 
                   linestyle='--', label=f'均值: {np.mean(episode_rewards):.2f}')
        ax2.axvline(np.median(episode_rewards), color='green', 
                   linestyle='--', label=f'中位数: {np.median(episode_rewards):.2f}')
        
        ax2.set_xlabel('总收益')
        ax2.set_ylabel('频次')
        ax2.set_title('收益分布')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_comparison(self, comparison_df: pd.DataFrame, 
                       save_path: Optional[str] = None):
        """绘制算法比较图"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics_to_plot = [
            ('total_revenue_mean', '平均总收益'),
            ('customer_satisfaction_mean', '客户满意度'),
            ('stockout_rate_mean', '缺货率'),
            ('inventory_balance_mean', '库存平衡度'),
            ('inventory_turnover_mean', '库存周转率'),
            ('relative_performance', '相对性能')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            if metric in comparison_df.columns:
                data = comparison_df[['algorithm', metric]].set_index('algorithm')
                data.plot(kind='bar', ax=ax, legend=False)
                ax.set_title(title)
                ax.set_xlabel('算法')
                ax.set_ylabel(title)
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for i, v in enumerate(data[metric]):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图已保存到: {save_path}")
        
        plt.show()