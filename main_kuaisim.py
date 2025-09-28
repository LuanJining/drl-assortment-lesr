# main_kuaisim.py - 更新版本支持真实KuaiSim集成
import argparse
import yaml
import numpy as np
import torch
import os
from datetime import datetime
from pathlib import Path
import logging

# 原有组件
from core.llm_optimizer import LLMOptimizer
from core.state_enhancer import StateEnhancer
from core.intrinsic_reward import IntrinsicRewardCalculator
from core.feedback_analyzer import FeedbackAnalyzer
from rl.a2c_agent import A2CAgent

# 新的KuaiSim适配器
from adapters.kuaisim_adapter import create_kuaisim_environment

# 基准算法
from baselines.myopic_agent import MyopicAgent
from baselines.eib_agent import EIBAgent
from baselines.random_agent import RandomAgent

# 现有工具
from utils.evaluation import Evaluator
from utils.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KuaiSimEnvironmentWrapper:
    """KuaiSim增强环境包装器 - 修复版本"""

    def __init__(self, base_env, state_enhancer, reward_calculator):
        self.env = base_env
        self.state_enhancer = state_enhancer
        self.reward_calculator = reward_calculator

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        enhanced_obs = self.state_enhancer.enhance(self._format_info_for_enhancer(info, obs))
        return enhanced_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        enhanced_obs = self.state_enhancer.enhance(self._format_info_for_enhancer(info, obs))

        # 计算内在奖励 - 修复action参数类型问题
        try:
            intrinsic = self.reward_calculator.calculate(
                enhanced_obs,
                action,  # 直接传递action数组
                enhanced_obs,
                info.get('sold_item', -1),
                reward
            )
        except Exception as e:
            logger.warning(f"内在奖励计算失败: {e}")
            intrinsic = 0.0

        total_reward = reward + intrinsic * 0.1
        return enhanced_obs, total_reward, terminated, truncated, info

    def _format_info_for_enhancer(self, info, obs):
        """格式化信息给状态增强器"""
        return {
            'inventory': info.get('inventory', np.ones(10)),
            'customer_type': info.get('current_user_type', 0),
            'prices': info.get('prices', np.ones(10)),
            'time_remaining': info.get('time_remaining', 10),
            'initial_inventory': info.get('initial_inventory', np.ones(10))
        }


def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建输出目录
    output_dir = Path("results") / f"kuaisim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 开始KuaiSim集成训练 ===")

    # 创建KuaiSim环境
    try:
        kuaisim_env = create_kuaisim_environment(config['kuaisim'])
        logger.info(f"KuaiSim环境创建成功")

        # 获取环境状态信息
        test_state, test_info = kuaisim_env.reset()
        kuaisim_status = test_info.get('kuaisim_status', '未知')
        logger.info(f"KuaiSim状态: {kuaisim_status}")
        logger.info(f"状态维度: {kuaisim_env.state_dim}, 动作维度: {kuaisim_env.action_dim}")

    except Exception as e:
        logger.error(f"KuaiSim环境创建失败: {e}")
        return

    # 初始化LLM组件
    try:
        llm_optimizer = LLMOptimizer(
            api_key=config['llm']['api_key'],
            base_url=config['llm']['base_url'],
            model=config['llm']['model']
        )

        state_enhancer = StateEnhancer()
        reward_calculator = IntrinsicRewardCalculator()
        feedback_analyzer = FeedbackAnalyzer()

        logger.info(f"LLM组件初始化成功")
        logger.info(f"API连接状态: {llm_optimizer.get_connection_status()}")

    except Exception as e:
        logger.error(f"LLM组件初始化失败: {e}")
        return

    best_performance = -float('inf')
    best_state_func = None
    best_reward_func = None

    # 主训练循环
    for iteration in range(config['training']['num_iterations']):
        logger.info(f"\n=== 迭代 {iteration + 1}/{config['training']['num_iterations']} ===")

        # 生成状态和奖励函数
        state_functions = []
        reward_functions = []

        for sample_idx in range(config['llm']['samples_per_iteration']):
            logger.info(f"生成样本 {sample_idx + 1}/{config['llm']['samples_per_iteration']}")

            try:
                # 生成状态增强函数 - 适配KuaiSim
                state_func = llm_optimizer.generate_state_representation(
                    task_description=config['task']['description'],
                    state_info={
                        'inventory_shape': config['kuaisim']['num_products'],
                        'customer_types': 4,
                        'num_products': config['kuaisim']['num_products'],
                        'cardinality': config['kuaisim']['cardinality'],
                        'is_kuaisim': True,
                        'slate_size': config['kuaisim']['cardinality']
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
                    logger.info(f"样本 {sample_idx + 1} 生成成功")

            except Exception as e:
                logger.error(f"生成样本 {sample_idx + 1} 时出错: {e}")

        # 如果没有成功生成，使用默认函数
        if not state_functions:
            logger.warning("使用默认函数")
            state_functions = [llm_optimizer._get_default_state_function()]
            reward_functions = [llm_optimizer._get_default_reward_function()]

        # 训练和评估
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
                enhanced_env = KuaiSimEnvironmentWrapper(
                    kuaisim_env, state_enhancer, reward_calculator
                )

                # 获取状态维度
                test_state, _ = enhanced_env.reset()
                state_dim = len(test_state)

                logger.info(f"样本 {idx + 1} 状态维度: {state_dim}")

                # 创建智能体
                agent = A2CAgent(
                    state_dim=state_dim,
                    action_dim=config['kuaisim']['num_products'],
                    hidden_dim=config['agent']['hidden_dim'],
                    learning_rate=config['agent']['learning_rate']
                )

                # 训练
                performance = train_kuaisim_agent(
                    agent, enhanced_env,
                    num_episodes=config['training']['episodes_per_sample'],
                    log_freq=20
                )

                performances.append(performance)
                logger.info(f"样本 {idx + 1} 性能: {performance:.2f}")

            except Exception as e:
                logger.error(f"训练样本 {idx + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                performances.append(0.0)

        # 分析反馈
        if performances:
            feedback = feedback_analyzer.analyze_performance(performances, state_functions)

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

            llm_optimizer.update_with_feedback(feedback)

    # 评估基准算法
    logger.info("\n=== 评估基准算法 ===")

    # 使用最佳函数创建最终环境
    if best_state_func and best_reward_func:
        state_enhancer.load_function(best_state_func)
        reward_calculator.load_function(best_reward_func)

    final_env = KuaiSimEnvironmentWrapper(kuaisim_env, state_enhancer, reward_calculator)

    results = {'best_rl_performance': best_performance}

    # 评估基准算法
    baseline_algorithms = {
        'Random': RandomAgent(config['kuaisim']['num_products'], config['kuaisim']['cardinality']),
        'Myopic': MyopicAgent(config['kuaisim']['num_products'], config['kuaisim']['cardinality']),
        'EIB': EIBAgent(config['kuaisim']['num_products'], config['kuaisim']['cardinality'])
    }

    for name, agent in baseline_algorithms.items():
        try:
            mean_reward, std_reward = evaluate_baseline_kuaisim(agent, kuaisim_env)
            results[f'{name}_mean'] = mean_reward
            results[f'{name}_std'] = std_reward
            logger.info(f"{name}: {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            logger.error(f"评估{name}失败: {e}")

    # 保存结果
    save_results(results, output_dir)

    # 显示改进
    if 'Random_mean' in results and results['Random_mean'] > 0:
        improvement = (best_performance - results['Random_mean']) / results['Random_mean'] * 100
        logger.info(f"\n最终性能提升: {improvement:.1f}% (相对于随机策略)")

    # 显示KuaiSim状态
    logger.info(f"\nKuaiSim集成状态: {kuaisim_status}")
    logger.info(f"结果已保存到: {output_dir}")
    logger.info("KuaiSim集成训练完成!")


def train_kuaisim_agent(agent, env, num_episodes=50, log_freq=10):
    """训练智能体 - 修复版本"""
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []

        step_count = 0
        max_steps_per_episode = 20  # 防止无限循环

        while not done and step_count < max_steps_per_episode:
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # 获取有效动作掩码
                inventory = info.get('inventory', np.ones(agent.action_dim))
                mask = (inventory <= 0).astype(np.float32)

                # 选择动作 - 使用智能体的select_action方法
                action, log_prob = agent.select_action(state, mask=mask)

                # 执行动作
                next_state, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                # 存储轨迹
                trajectory_states.append(state.copy())
                # 找到选中的动作索引
                action_idx = np.where(action > 0.5)[0]
                if len(action_idx) > 0:
                    trajectory_actions.append(action_idx[0])
                else:
                    trajectory_actions.append(0)
                trajectory_rewards.append(reward)

                state = next_state
                info = next_info
                episode_reward += reward
                step_count += 1

            except Exception as e:
                logger.warning(f"训练步骤失败: {e}")
                break

        episode_rewards.append(episode_reward)

        # 更新智能体
        if len(trajectory_states) > 0:
            update_agent_fixed(agent, trajectory_states, trajectory_actions, trajectory_rewards)

        # 日志记录
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return np.mean(episode_rewards[-min(10, len(episode_rewards)):])


def update_agent_fixed(agent, states, actions, rewards):
    """更新智能体 - 从原版复制"""
    try:
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)

        # 计算折扣回报
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns)

        # 前向传播
        action_logits, values = agent(states_tensor)
        values = values.squeeze()

        # 计算损失
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)

        advantages = returns_tensor - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((values - returns_tensor) ** 2).mean()
        entropy_loss = -dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        # 反向传播
        agent.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        agent.optimizer.step()

    except Exception as e:
        logger.warning(f"更新智能体失败: {e}")


def evaluate_baseline_kuaisim(agent, env, num_episodes=20):
    """评估基准算法"""
    total_rewards = []

    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 20:
            try:
                if isinstance(agent, MyopicAgent):
                    action = agent.select_action(info['inventory'], info['prices'])
                elif isinstance(agent, EIBAgent):
                    action = agent.select_action(
                        info['inventory'], info['initial_inventory'],
                        info['prices'], info['time_remaining']
                    )
                else:  # RandomAgent
                    action = agent.select_action(info['inventory'])

                _, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1

            except Exception as e:
                logger.warning(f"基准算法执行失败: {e}")
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def save_results(results, output_dir):
    """保存结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存性能数据
    np.save(output_dir / f"kuaisim_results_{timestamp}.npy", results)

    # 保存摘要
    with open(output_dir / f"kuaisim_summary_{timestamp}.txt", 'w') as f:
        f.write("KuaiSim集成实验结果摘要\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DRL-Assortment-LESR with KuaiSim')
    parser.add_argument('--config', default='config/kuaisim_config.yaml',
                        help='KuaiSim配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        main(args)
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        import traceback

        traceback.print_exc()