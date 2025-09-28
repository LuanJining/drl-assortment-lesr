import argparse
import yaml
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import logging

# 导入项目核心组件
from core.llm_optimizer import LLMOptimizer  # LLM优化器，用于生成状态表示和奖励函数
from core.state_enhancer import StateEnhancer  # 状态增强器，用于处理环境状态
from core.intrinsic_reward import IntrinsicRewardCalculator  # 内在奖励计算器
from core.feedback_analyzer import FeedbackAnalyzer  # 反馈分析器，用于分析性能并提供反馈

# 导入强化学习相关组件
from rl.a2c_agent import A2CAgent  # A2C（Advantage Actor-Critic）智能体
from rl.environment import AssortmentEnvironment  # 产品组合优化环境

# 导入基准算法智能体
from baselines.myopic_agent import MyopicAgent  # 短视算法智能体
from baselines.eib_agent import EIBAgent  # EIB（Expected Improvement Bound）算法智能体
from baselines.random_agent import RandomAgent  # 随机算法智能体

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # 创建日志记录器


class EnhancedEnvironmentWrapper:
    """增强环境包装器，用于包装基础环境并添加状态增强和内在奖励功能"""

    def __init__(self, base_env, state_enhancer, reward_calculator):
        """初始化增强环境包装器
        
        Args:
            base_env: 基础环境对象（AssortmentEnvironment实例）
            state_enhancer: 状态增强器对象（StateEnhancer实例）
            reward_calculator: 内在奖励计算器对象（IntrinsicRewardCalculator实例）
        """
        self.env = base_env
        self.state_enhancer = state_enhancer
        self.reward_calculator = reward_calculator

    def reset(self, seed=None):
        """重置环境到初始状态并返回增强后的观察
        
        Args:
            seed: 随机种子（可选）
            
        Returns:
            tuple: (增强后的观察, 环境信息)
        """
        obs, info = self.env.reset(seed)
        enhanced_obs = self.state_enhancer.enhance(info)  # 增强原始观察
        return enhanced_obs, info

    def step(self, action):
        """执行动作并返回增强后的环境反馈
        
        Args:
            action: 智能体选择的动作
            
        Returns:
            tuple: (增强后的观察, 总奖励, 是否终止, 是否截断, 环境信息)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 增强状态
        enhanced_obs = self.state_enhancer.enhance(info)

        # 计算内在奖励
        intrinsic = self.reward_calculator.calculate(
            enhanced_obs,
            action,
            enhanced_obs,  # 这里简化了，实际应该是前一个状态
            info.get('sold_item', -1),  # 获取卖出的商品，默认为-1（没卖出）
            reward
        )

        # 组合奖励：外在奖励 + 内在奖励（乘以权重）
        total_reward = reward + intrinsic * 0.1  # 权重可调

        return enhanced_obs, total_reward, terminated, truncated, info


def train_agent(agent, env, num_episodes=1000, log_freq=100):
    """训练智能体 - 修复梯度问题的版本
    
    Args:
        agent: 要训练的智能体对象
        env: 训练环境
        num_episodes: 训练的回合数
        log_freq: 日志记录频率
        
    Returns:
        float: 最后100个回合的平均奖励
    """
    episode_rewards = []  # 存储每个回合的奖励

    for episode in range(num_episodes):
        state, info = env.reset()  # 重置环境
        episode_reward = 0  # 当前回合的奖励
        done = False  # 回合结束标志

        # 存储轨迹数据
        trajectory_states = []  # 存储状态序列
        trajectory_actions = []  # 存储动作索引序列
        trajectory_rewards = []  # 存储奖励序列

        while not done:
            # 确保状态是tensor格式
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加批次维度

            # 获取有效动作掩码（基于库存）
            inventory = info.get('inventory', np.ones(agent.action_dim))
            mask = (inventory <= 0).astype(np.float32)  # 库存为0的商品不可选
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)  # 添加批次维度

            # 选择动作（不使用no_grad，保持梯度）
            action_logits, value = agent(state_tensor)

            # 应用掩码：将不可选动作的概率设为负无穷
            action_logits = action_logits.masked_fill(mask_tensor.bool(), -float('inf'))

            # 采样动作
            action_probs = torch.softmax(action_logits, dim=-1)  # 计算动作概率
            dist = torch.distributions.Categorical(action_probs)  # 创建分类分布
            action_idx = dist.sample()  # 从分布中采样动作索引

            # 转换为二进制动作向量
            action = np.zeros(agent.action_dim, dtype=np.float32)
            if action_idx.item() < agent.action_dim:
                action[action_idx.item()] = 1

            # 执行动作
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated  # 回合结束条件

            # 存储轨迹（仅存储必要数据）
            trajectory_states.append(state.copy())
            trajectory_actions.append(action_idx.item())
            trajectory_rewards.append(reward)

            # 更新状态和信息
            state = next_state
            info = next_info
            episode_reward += reward

        episode_rewards.append(episode_reward)  # 记录本回合总奖励

        # 使用轨迹数据更新智能体
        if len(trajectory_states) > 0:
            update_agent_fixed(agent, trajectory_states, trajectory_actions, trajectory_rewards)

        # 日志记录
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])  # 计算最近log_freq个回合的平均奖励
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                        f"Avg Reward: {avg_reward:.2f}")

    return np.mean(episode_rewards[-100:])  # 返回最后100个回合的平均奖励


def update_agent_fixed(agent, states, actions, rewards):
    """修复的智能体更新函数，实现A2C算法的更新步骤
    
    Args:
        agent: 要更新的智能体
        states: 状态序列
        actions: 动作序列
        rewards: 奖励序列
    """
    try:
        # 转换为tensor
        states_tensor = torch.FloatTensor(states)  # [episode_length, state_dim]
        actions_tensor = torch.LongTensor(actions)  # [episode_length]

        # 计算折扣回报
        gamma = 0.99  # 折扣因子
        returns = []
        G = 0  # 累积回报
        for r in reversed(rewards):  # 从后向前计算
            G = r + gamma * G
            returns.insert(0, G)  # 插入到列表前端，保持原始顺序
        returns_tensor = torch.FloatTensor(returns)

        # 前向传播获取logits和values（保持梯度）
        action_logits, values = agent(states_tensor)  # 不使用no_grad
        values = values.squeeze()  # 去除多余的维度

        # 计算log概率
        action_probs = torch.softmax(action_logits, dim=-1)  # 计算动作概率
        dist = torch.distributions.Categorical(action_probs)  # 创建分类分布
        log_probs = dist.log_prob(actions_tensor)  # 计算所选动作的对数概率

        # 计算优势函数
        advantages = returns_tensor - values.detach()  # 优势 = 实际回报 - 预测值

        # 计算各部分损失
        actor_loss = -(log_probs * advantages).mean()  # 策略损失
        critic_loss = ((values - returns_tensor) ** 2).mean()  # 价值函数损失
        entropy_loss = -dist.entropy().mean()  # 熵损失（鼓励探索）

        # 总损失：策略损失 + 价值函数损失（带权重） + 熵损失（带权重）
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        # 反向传播更新参数
        agent.optimizer.zero_grad()  # 清零梯度
        total_loss.backward()  # 反向传播计算梯度

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

        agent.optimizer.step()  # 更新参数

    except Exception as e:
        logger.warning(f"更新智能体失败: {e}")
        import traceback
        traceback.print_exc()


def evaluate_baseline(agent_class, env, num_episodes=100):
    """评估基准算法的性能
    
    Args:
        agent_class: 基准算法智能体类
        env: 评估环境
        num_episodes: 评估的回合数
        
    Returns:
        tuple: (平均奖励, 奖励标准差)
    """
    agent = agent_class(env.env.num_products, env.env.cardinality)  # 创建基准智能体
    total_rewards = []  # 存储每个回合的奖励

    for _ in range(num_episodes):
        state, info = env.env.reset()  # 使用原始环境
        episode_reward = 0  # 当前回合的奖励
        done = False  # 回合结束标志

        while not done:
            # 基准算法决策
            try:
                if isinstance(agent, MyopicAgent):
                    # 短视算法需要库存和价格信息
                    action = agent.select_action(
                        info['inventory'],
                        info['prices']
                    )
                elif isinstance(agent, EIBAgent):
                    # EIB算法需要更多信息
                    action = agent.select_action(
                        info['inventory'],
                        info['initial_inventory'],
                        info['prices'],
                        info['time_remaining']
                    )
                else:  # RandomAgent
                    # 随机算法只需要库存信息
                    action = agent.select_action(info['inventory'])

                _, reward, terminated, truncated, info = env.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            except Exception as e:
                logger.warning(f"基准算法执行失败: {e}")
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)  # 返回平均奖励和标准差


def save_results(results, output_dir):
    """保存实验结果
    
    Args:
        results: 结果字典
        output_dir: 输出目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳

    # 保存性能数据（numpy格式）
    np.save(output_dir / f"results_{timestamp}.npy", results)

    # 保存摘要（文本格式）
    with open(output_dir / f"summary_{timestamp}.txt", 'w') as f:
        f.write("实验结果摘要\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


def main(args):
    """主函数，程序入口
    
    Args:
        args: 命令行参数
    """
    # 加载配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)  # 使用yaml加载配置

    # 创建输出目录
    output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，支持多级目录和已存在目录

    # 获取LLM相关配置
    api_key = config['llm'].get('api_key')  # API密钥
    base_url = config['llm'].get('base_url')  # API基础URL

    logger.info("=== 开始DRL-Assortment-LESR训练 ===")

    # 初始化核心组件
    llm_optimizer = LLMOptimizer(
        api_key=api_key,
        base_url=base_url,
        model=config['llm']['model']  # LLM模型名称
    )

    state_enhancer = StateEnhancer()  # 状态增强器
    reward_calculator = IntrinsicRewardCalculator()  # 内在奖励计算器
    feedback_analyzer = FeedbackAnalyzer()  # 反馈分析器

    # 创建基础环境
    base_env = AssortmentEnvironment(
        num_products=config['env']['num_products'],  # 产品数量
        num_customer_types=config['env']['num_customer_types'],  # 客户类型数量
        initial_inventory=np.ones(config['env']['num_products']) * config['env']['initial_inventory'],  # 初始库存
        cardinality=config['env']['cardinality']  # 产品组合大小
    )

    # 初始化最佳性能和最佳函数
    best_performance = -float('inf')  # 最佳性能
    best_state_func = None  # 最佳状态表示函数
    best_reward_func = None  # 最佳奖励函数

    # 主训练循环
    for iteration in range(config['training']['num_iterations']):
        logger.info(f"\n=== 迭代 {iteration + 1}/{config['training']['num_iterations']} ===")

        # Step 1: 生成状态表示函数
        logger.info("生成状态表示函数...")
        state_functions = []  # 存储生成的状态表示函数
        reward_functions = []  # 存储生成的奖励函数

        for sample_idx in range(config['llm']['samples_per_iteration']):
            logger.info(f"生成样本 {sample_idx + 1}/{config['llm']['samples_per_iteration']}")

            try:
                # 生成状态增强函数
                state_func = llm_optimizer.generate_state_representation(
                    task_description=config['task']['description'],  # 任务描述
                    state_info={
                        'inventory_shape': config['env']['num_products'],  # 库存形状
                        'customer_types': config['env']['num_customer_types'],  # 客户类型数量
                        'num_products': config['env']['num_products'],  # 产品数量
                        'cardinality': config['env']['cardinality']  # 产品组合大小
                    }
                )

                # 生成奖励函数
                reward_func = llm_optimizer.generate_intrinsic_reward(
                    state_representation=state_func,  # 状态表示函数
                    performance_feedback=None if iteration == 0 else feedback_analyzer.performance_history[-1]  # 性能反馈
                )

                if state_func and reward_func:
                    state_functions.append(state_func)
                    reward_functions.append(reward_func)
                else:
                    logger.warning(f"样本 {sample_idx + 1} 生成失败")

            except Exception as e:
                logger.error(f"生成样本 {sample_idx + 1} 时出错: {e}")

        # 如果没有成功生成函数，使用默认函数
        if not state_functions:
            logger.warning("使用默认状态增强函数")
            state_functions = [llm_optimizer._get_default_state_function()]
            reward_functions = [llm_optimizer._get_default_reward_function()]

        # Step 2: 训练和评估
        performances = []  # 存储每个函数的性能

        for idx in range(len(state_functions)):
            logger.info(f"\n训练样本 {idx + 1}/{len(state_functions)}")

            try:
                # 加载函数
                state_loaded = state_enhancer.load_function(state_functions[idx])
                reward_loaded = reward_calculator.load_function(reward_functions[idx])

                if not (state_loaded and reward_loaded):
                    logger.warning(f"样本 {idx + 1} 函数加载失败")
                    performances.append(0.0)
                    continue

                # 创建增强环境
                enhanced_env = EnhancedEnvironmentWrapper(
                    base_env, state_enhancer, reward_calculator
                )

                # 获取状态维度
                test_state, _ = enhanced_env.reset()
                state_dim = len(test_state)

                logger.info(f"状态维度: {state_dim}")
                # 注释掉详细的状态打印，减少日志
                # print(f"测试状态: {test_state}")

                # 创建智能体
                agent = A2CAgent(
                    state_dim=state_dim,  # 状态维度
                    action_dim=config['env']['num_products'],  # 动作维度（产品数量）
                    hidden_dim=config['agent']['hidden_dim'],  # 隐藏层维度
                    learning_rate=config['agent']['learning_rate']  # 学习率
                )

                # 训练（减少episode数量进行测试）
                performance = train_agent(
                    agent,
                    enhanced_env,
                    num_episodes=min(200, config['training']['episodes_per_sample']),  # 减少到200进行测试
                    log_freq=50  # 增加日志频率
                )

                performances.append(performance)
                logger.info(f"样本 {idx + 1} 性能: {performance:.2f}")

            except Exception as e:
                logger.error(f"训练样本 {idx + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                performances.append(0.0)

        # Step 3: 分析反馈
        if performances:
            # 分析性能，生成反馈
            feedback = feedback_analyzer.analyze_performance(
                performances,
                state_functions
            )

            # 选择最佳
            best_idx = np.argmax(performances)  # 找到性能最好的函数索引
            if performances[best_idx] > best_performance:
                best_performance = performances[best_idx]
                best_state_func = state_functions[best_idx]
                best_reward_func = reward_functions[best_idx]

                # 保存最佳函数
                with open(output_dir / "best_state_func.py", 'w') as f:
                    f.write(best_state_func)
                with open(output_dir / "best_reward_func.py", 'w') as f:
                    f.write(best_reward_func)

                logger.info(f"新最佳性能: {best_performance:.2f}")

            # 更新LLM优化器，提供反馈
            llm_optimizer.update_with_feedback(feedback)

    logger.info("\n=== 评估基准算法 ===")

    # 创建最终环境
    if best_state_func and best_reward_func:
        state_enhancer.load_function(best_state_func)
        reward_calculator.load_function(best_reward_func)
    final_env = EnhancedEnvironmentWrapper(base_env, state_enhancer, reward_calculator)

    # 评估基准算法
    results = {'best_rl_performance': best_performance}  # 存储结果

    # 评估各种基准算法
    for name, agent_class in [
        ('Random', RandomAgent),  # 随机算法
        ('Myopic', MyopicAgent),  # 短视算法
        ('EIB', EIBAgent)  # EIB算法
    ]:
        try:
            mean_reward, std_reward = evaluate_baseline(agent_class, final_env)
            results[f'{name}_mean'] = mean_reward  # 平均奖励
            results[f'{name}_std'] = std_reward  # 奖励标准差
            logger.info(f"{name} Agent: {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            logger.error(f"评估{name}算法失败: {e}")

    # 保存结果
    save_results(results, output_dir)

    # 显示改进
    if 'Random_mean' in results and results['Random_mean'] > 0:
        # 计算相对于随机策略的性能提升百分比
        improvement = (best_performance - results['Random_mean']) / results['Random_mean'] * 100
        logger.info(f"\n最终性能提升: {improvement:.1f}% (相对于随机策略)")

    logger.info(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DRL-Assortment-LESR Training')
    parser.add_argument('--config', default='config/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    # 设置随机种子，保证实验可复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 运行主程序
    try:
        main(args)
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        import traceback

        traceback.print_exc()
        raise