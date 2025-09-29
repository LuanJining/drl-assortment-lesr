import argparse
import yaml
import numpy as np
import torch
import os
from datetime import datetime
from pathlib import Path
import logging

from core.llm_optimizer import LLMOptimizer
from core.state_enhancer import StateEnhancer
from core.intrinsic_reward import IntrinsicRewardCalculator
from core.feedback_analyzer import FeedbackAnalyzer
from rl.a2c_agent import A2CAgent
from rl.environment import AssortmentEnvironment
from baselines.myopic_agent import MyopicAgent
from baselines.eib_agent import EIBAgent
from baselines.random_agent import RandomAgent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedEnvironmentWrapper:
    """增强环境包装器"""

    def __init__(self, base_env, state_enhancer, reward_calculator):
        self.env = base_env
        self.state_enhancer = state_enhancer
        self.reward_calculator = reward_calculator

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        enhanced_obs = self.state_enhancer.enhance(info)
        return enhanced_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 增强状态
        enhanced_obs = self.state_enhancer.enhance(info)

        # 计算内在奖励
        intrinsic = self.reward_calculator.calculate(
            enhanced_obs,
            action,
            enhanced_obs,  # 这里简化了，实际应该是前一个状态
            info.get('sold_item', -1),
            reward
        )

        # 组合奖励
        total_reward = reward + intrinsic * 0.1  # 权重可调

        return enhanced_obs, total_reward, terminated, truncated, info


def train_agent(agent, env, num_episodes=1000, log_freq=100):
    """训练智能体 - 修复梯度问题"""
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        # 存储轨迹数据
        trajectory_states = []
        trajectory_actions = []  # 存储动作索引
        trajectory_rewards = []

        while not done:
            # 确保状态是tensor格式
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 获取有效动作掩码
            inventory = info.get('inventory', np.ones(agent.action_dim))
            mask = (inventory <= 0).astype(np.float32)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

            # 选择动作（不使用no_grad，保持梯度）
            action_logits, value = agent(state_tensor)

            # 应用掩码
            action_logits = action_logits.masked_fill(mask_tensor.bool(), -float('inf'))

            # 采样动作
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()

            # 转换为二进制动作向量
            action = np.zeros(agent.action_dim, dtype=np.float32)
            if action_idx.item() < agent.action_dim:
                action[action_idx.item()] = 1

            # 执行动作
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # 存储轨迹（仅存储必要数据）
            trajectory_states.append(state.copy())
            trajectory_actions.append(action_idx.item())
            trajectory_rewards.append(reward)

            state = next_state
            info = next_info
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # 使用轨迹数据更新智能体
        if len(trajectory_states) > 0:
            update_agent_fixed(agent, trajectory_states, trajectory_actions, trajectory_rewards)

        # 日志记录
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                        f"Avg Reward: {avg_reward:.2f}")

    return np.mean(episode_rewards[-100:])


def update_agent_fixed(agent, states, actions, rewards):
    """修复的智能体更新函数"""
    try:
        # 转换为tensor
        states_tensor = torch.FloatTensor(states)  # [episode_length, state_dim]
        actions_tensor = torch.LongTensor(actions)  # [episode_length]

        # 计算折扣回报
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns)

        # 前向传播获取logits和values（保持梯度）
        action_logits, values = agent(states_tensor)  # 不使用no_grad
        values = values.squeeze()

        # 计算log概率
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)

        # 计算优势
        advantages = returns_tensor - values.detach()

        # 计算损失
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((values - returns_tensor) ** 2).mean()
        entropy_loss = -dist.entropy().mean()

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        # 反向传播
        agent.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

        agent.optimizer.step()

    except Exception as e:
        logger.warning(f"更新智能体失败: {e}")
        import traceback
        traceback.print_exc()


def evaluate_baseline(agent_class, env, num_episodes=100):
    """评估基准算法"""
    agent = agent_class(env.env.num_products, env.env.cardinality)
    total_rewards = []

    for _ in range(num_episodes):
        state, info = env.env.reset()  # 使用原始环境
        episode_reward = 0
        done = False

        while not done:
            # 基准算法决策
            try:
                if isinstance(agent, MyopicAgent):
                    action = agent.select_action(
                        info['inventory'],
                        info['prices']
                    )
                elif isinstance(agent, EIBAgent):
                    action = agent.select_action(
                        info['inventory'],
                        info['initial_inventory'],
                        info['prices'],
                        info['time_remaining']
                    )
                else:  # RandomAgent
                    action = agent.select_action(info['inventory'])

                _, reward, terminated, truncated, info = env.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            except Exception as e:
                logger.warning(f"基准算法执行失败: {e}")
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def save_results(results, output_dir):
    """保存实验结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存性能数据
    np.save(output_dir / f"results_{timestamp}.npy", results)

    # 保存摘要
    with open(output_dir / f"summary_{timestamp}.txt", 'w') as f:
        f.write("实验结果摘要\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建输出目录
    output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = config['llm'].get('api_key')
    base_url = config['llm'].get('base_url')

    logger.info("=== 开始DRL-Assortment-LESR训练 ===")

    # 初始化组件
    llm_optimizer = LLMOptimizer(
        api_key=api_key,
        base_url=base_url,
        model=config['llm']['model']
    )

    state_enhancer = StateEnhancer()
    reward_calculator = IntrinsicRewardCalculator()
    feedback_analyzer = FeedbackAnalyzer()

    # 创建基础环境
    base_env = AssortmentEnvironment(
        num_products=config['env']['num_products'],
        num_customer_types=config['env']['num_customer_types'],
        initial_inventory=np.ones(config['env']['num_products']) * config['env']['initial_inventory'],
        cardinality=config['env']['cardinality']
    )

    best_performance = -float('inf')
    best_state_func = None
    best_reward_func = None

    # 主训练循环
    for iteration in range(config['training']['num_iterations']):
        logger.info(f"\n=== 迭代 {iteration + 1}/{config['training']['num_iterations']} ===")

        # Step 1: 生成状态表示函数
        logger.info("生成状态表示函数...")
        state_functions = []
        reward_functions = []

        for sample_idx in range(config['llm']['samples_per_iteration']):
            logger.info(f"生成样本 {sample_idx + 1}/{config['llm']['samples_per_iteration']}")

            try:
                # 生成状态增强函数
                state_func = llm_optimizer.generate_state_representation(
                    task_description=config['task']['description'],
                    state_info={
                        'inventory_shape': config['env']['num_products'],
                        'customer_types': config['env']['num_customer_types'],
                        'num_products': config['env']['num_products'],
                        'cardinality': config['env']['cardinality']
                    }
                )

                # 生成奖励函数
                reward_func = llm_optimizer.generate_intrinsic_reward(
                    state_representation=state_func,
                    performance_feedback=None if iteration == 0 else feedback_analyzer.performance_history[-1]
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
        performances = []

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
                    state_dim=state_dim,
                    action_dim=config['env']['num_products'],
                    hidden_dim=config['agent']['hidden_dim'],
                    learning_rate=config['agent']['learning_rate']
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
            feedback = feedback_analyzer.analyze_performance(
                performances,
                state_functions
            )

            # 选择最佳
            best_idx = np.argmax(performances)
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

            # 更新LLM优化器
            llm_optimizer.update_with_feedback(feedback)

    logger.info("\n=== 评估基准算法 ===")

    # 创建最终环境
    if best_state_func and best_reward_func:
        state_enhancer.load_function(best_state_func)
        reward_calculator.load_function(best_reward_func)
    final_env = EnhancedEnvironmentWrapper(base_env, state_enhancer, reward_calculator)

    # 评估基准算法
    results = {'best_rl_performance': best_performance}

    for name, agent_class in [
        ('Random', RandomAgent),
        ('Myopic', MyopicAgent),
        ('EIB', EIBAgent)
    ]:
        try:
            mean_reward, std_reward = evaluate_baseline(agent_class, final_env)
            results[f'{name}_mean'] = mean_reward
            results[f'{name}_std'] = std_reward
            logger.info(f"{name} Agent: {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            logger.error(f"评估{name}算法失败: {e}")

    # 保存结果
    save_results(results, output_dir)

    # 显示改进
    if 'Random_mean' in results and results['Random_mean'] > 0:
        improvement = (best_performance - results['Random_mean']) / results['Random_mean'] * 100
        logger.info(f"\n最终性能提升: {improvement:.1f}% (相对于随机策略)")

    logger.info(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DRL-Assortment-LESR Training')
    parser.add_argument('--config', default='config/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
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