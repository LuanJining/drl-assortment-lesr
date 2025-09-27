import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import pandas as pd
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import HTML

class Visualizer:
    """可视化工具"""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_inventory_dynamics(self, inventory_history: List[np.ndarray],
                               product_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
        """绘制库存动态变化"""
        
        if not inventory_history:
            print("没有库存历史数据")
            return
        
        num_products = len(inventory_history[0])
        time_steps = len(inventory_history)
        
        if product_names is None:
            product_names = [f'产品{i+1}' for i in range(num_products)]
        
        # 创建数据矩阵
        inventory_matrix = np.array(inventory_history).T
        
        # 绘制热力图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 热力图
        sns.heatmap(inventory_matrix, ax=ax1, cmap='YlOrRd_r', 
                   cbar_kws={'label': '库存量'},
                   xticklabels=range(0, time_steps, max(1, time_steps//20)),
                   yticklabels=product_names)
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('产品')
        ax1.set_title('库存热力图')
        
        # 折线图
        for i in range(num_products):
            ax2.plot(inventory_matrix[i], label=product_names[i], linewidth=2)
        
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('库存量')
        ax2.set_title('库存变化曲线')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"库存动态图已保存到: {save_path}")
        
        plt.show()
    
    def plot_sales_pattern(self, sales_data: List[Dict],
                          save_path: Optional[str] = None):
        """绘制销售模式"""
        
        if not sales_data:
            print("没有销售数据")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(sales_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 销售时间分布
        if 'time' in df.columns:
            df['time'].hist(bins=30, ax=axes[0, 0], edgecolor='black')
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('销售次数')
            axes[0, 0].set_title('销售时间分布')
        
        # 2. 产品销售分布
        if 'product' in df.columns:
            product_sales = df['product'].value_counts()
            product_sales.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_xlabel('产品ID')
            axes[0, 1].set_ylabel('销售次数')
            axes[0, 1].set_title('产品销售分布')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 客户类型分布
        if 'customer_type' in df.columns:
            customer_dist = df['customer_type'].value_counts()
            colors = plt.cm.Set3(range(len(customer_dist)))
            customer_dist.plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%',
                             colors=colors)
            axes[1, 0].set_title('客户类型分布')
            axes[1, 0].set_ylabel('')
        
        # 4. 收益分布
        if 'price' in df.columns:
            df['price'].hist(bins=20, ax=axes[1, 1], edgecolor='black')
            axes[1, 1].axvline(df['price'].mean(), color='red', 
                              linestyle='--', label=f"均值: {df['price'].mean():.2f}")
            axes[1, 1].set_xlabel('收益')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].set_title('收益分布')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"销售模式图已保存到: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, data: Dict) -> go.Figure:
        """创建交互式仪表板"""
        
        from plotly.subplots import make_subplots
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('收益趋势', '库存水平', '客户满意度', '性能指标'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # 1. 收益趋势
        if 'episode_rewards' in data:
            episodes = list(range(len(data['episode_rewards'])))
            fig.add_trace(
                go.Scatter(x=episodes, y=data['episode_rewards'],
                          mode='lines', name='Episode收益',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            # 添加移动平均
            window = min(100, len(episodes) // 10)
            if len(episodes) > window:
                moving_avg = pd.Series(data['episode_rewards']).rolling(window).mean()
                fig.add_trace(
                    go.Scatter(x=episodes, y=moving_avg,
                              mode='lines', name=f'移动平均({window})',
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )
        
        # 2. 库存水平
        if 'final_inventory' in data:
            products = [f'P{i+1}' for i in range(len(data['final_inventory']))]
            fig.add_trace(
                go.Bar(x=products, y=data['final_inventory'],
                      name='最终库存',
                      marker_color='lightgreen'),
                row=1, col=2
            )
            
            if 'initial_inventory' in data:
                fig.add_trace(
                    go.Bar(x=products, y=data['initial_inventory'],
                          name='初始库存',
                          marker_color='lightblue'),
                    row=1, col=2
                )
        
        # 3. 客户满意度趋势
        if 'satisfaction_history' in data:
            episodes = list(range(len(data['satisfaction_history'])))
            fig.add_trace(
                go.Scatter(x=episodes, y=data['satisfaction_history'],
                          mode='lines+markers', name='客户满意度',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # 4. 性能指标
        if 'performance_score' in data:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=data['performance_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "综合性能"},
                    delta={'reference': data.get('baseline_score', 50)},
                    gauge={
                        'axis': {'range': [None, 200]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"},
                            {'range': [100, 150], 'color': "lightgreen"},
                            {'range': [150, 200], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': data.get('target_score', 150)
                        }
                    }
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="DRL-Assortment-LESR 实时监控面板",
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def animate_episode(self, episode_data: Dict, 
                       save_path: Optional[str] = None):
        """动画展示一个episode的执行过程"""
        
        inventory_history = episode_data.get('inventory_history', [])
        actions_history = episode_data.get('actions_history', [])
        rewards_history = episode_data.get('rewards_history', [])
        
        if not inventory_history:
            print("没有episode数据")
            return
        
        num_products = len(inventory_history[0])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 初始化图表
        inventory_bars = axes[0, 0].bar(range(num_products), inventory_history[0])
        axes[0, 0].set_ylim(0, max([max(inv) for inv in inventory_history]) * 1.1)
        axes[0, 0].set_xlabel('产品')
        axes[0, 0].set_ylabel('库存量')
        axes[0, 0].set_title('当前库存')
        
        # 动作展示
        action_bars = axes[0, 1].bar(range(num_products), 
                                    actions_history[0] if actions_history else np.zeros(num_products))
        axes[0, 1].set_ylim(0, 1.2)
        axes[0, 1].set_xlabel('产品')
        axes[0, 1].set_ylabel('是否展示')
        axes[0, 1].set_title('当前动作')
        
        # 累积奖励
        cumulative_rewards = np.cumsum(rewards_history) if rewards_history else [0]
        reward_line, = axes[1, 0].plot([], [], 'b-', linewidth=2)
        axes[1, 0].set_xlim(0, len(inventory_history))
        axes[1, 0].set_ylim(0, max(cumulative_rewards) * 1.1 if cumulative_rewards else 1)
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('累积奖励')
        axes[1, 0].set_title('累积奖励')
        
        # 文本信息
        info_text = axes[1, 1].text(0.5, 0.5, '', ha='center', va='center', 
                                   fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        def update(frame):
            # 更新库存
            for bar, height in zip(inventory_bars, inventory_history[frame]):
                bar.set_height(height)
            
            # 更新动作
            if frame < len(actions_history):
                for bar, height in zip(action_bars, actions_history[frame]):
                    bar.set_height(height)
            
            # 更新累积奖励
            if frame > 0 and rewards_history:
                reward_line.set_data(range(frame), cumulative_rewards[:frame])
            
            # 更新文本信息
            info_text.set_text(f'时间步: {frame}\n'
                             f'当前奖励: {rewards_history[frame] if frame < len(rewards_history) else 0:.2f}\n'
                             f'总库存: {sum(inventory_history[frame]):.0f}')
            
            return list(inventory_bars) + list(action_bars) + [reward_line, info_text]
        
        anim = FuncAnimation(fig, update, frames=len(inventory_history),
                           interval=100, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"动画已保存到: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def plot_state_space_analysis(self, states: np.ndarray,
                                 rewards: np.ndarray,
                                 save_path: Optional[str] = None):
        """状态空间分析可视化"""
        
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. PCA降维可视化
        if states.shape[1] > 2:
            pca = PCA(n_components=2)
            states_pca = pca.fit_transform(states)
            
            scatter = axes[0, 0].scatter(states_pca[:, 0], states_pca[:, 1],
                                        c=rewards, cmap='viridis', alpha=0.6)
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            axes[0, 0].set_title('PCA状态空间可视化')
            plt.colorbar(scatter, ax=axes[0, 0], label='奖励')
        
        # 2. t-SNE降维可视化（如果样本不太多）
        if states.shape[0] < 5000 and states.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            states_tsne = tsne.fit_transform(states[:1000])  # 限制样本数
            
            scatter = axes[0, 1].scatter(states_tsne[:, 0], states_tsne[:, 1],
                                        c=rewards[:1000], cmap='plasma', alpha=0.6)
            axes[0, 1].set_xlabel('t-SNE 1')
            axes[0, 1].set_ylabel('t-SNE 2')
            axes[0, 1].set_title('t-SNE状态空间可视化')
            plt.colorbar(scatter, ax=axes[0, 1], label='奖励')
        
        # 3. 状态分布直方图
        axes[1, 0].hist(states.mean(axis=1), bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('平均状态值')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('状态分布')
        axes[1, 0].axvline(states.mean(), color='red', linestyle='--', 
                          label=f'均值: {states.mean():.2f}')
        axes[1, 0].legend()
        
        # 4. 奖励与状态相关性
        state_means = states.mean(axis=1)
        axes[1, 1].hexbin(state_means, rewards, gridsize=30, cmap='YlOrRd')
        axes[1, 1].set_xlabel('平均状态值')
        axes[1, 1].set_ylabel('奖励')
        axes[1, 1].set_title('状态-奖励相关性')
        
        # 添加趋势线
        z = np.polyfit(state_means, rewards, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(state_means, p(state_means), "r--", alpha=0.8, 
                       label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"状态空间分析图已保存到: {save_path}")
        
        plt.show()