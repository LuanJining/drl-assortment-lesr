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
        self.previous_state = None

    def reset(self, seed=None):
        obs, info = self.env.reset(seed)
        enhanced_obs = self.state_enhancer.enhance(info)
        self.previous_state = enhanced_obs.copy()
        return enhanced_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 增强状态
        enhanced_obs = self.state_enhancer.enhance(info)

        # 计算内在奖励
        try:
            if isinstance(action, np.ndarray) and len(action) > 1:
                selected_products = np.where(action > 0)[0]
                action_idx = selected_products[0] if len(selected_products) > 0 else -1
            else:
                action_idx = action

            intrinsic = self.reward_calculator.calculate(
                state=self.previous_state if self.previous_state is not None else enhanced_obs,
                action=action_idx,
                next_state=enhanced_obs,
                sold_item=info.get('sold_item', -1),
                price=reward
            )

            # 🔧 裁剪内在奖励
            intrinsic = np.clip(intrinsic, -10.0, 10.0)

        except Exception as e:
            intrinsic = 0.0
            if not hasattr(self, '_intrinsic_error_logged'):
                print(f"内在奖励计算出错: {e}")
                self._intrinsic_error_logged = True

        # 组合奖励
        total_reward = reward + intrinsic * 0.1

        # 🔧 裁剪总奖励
        total_reward = np.clip(total_reward, -100.0, 100.0)

        self.previous_state = enhanced_obs.copy()

        return enhanced_obs, total_reward, terminated, truncated, info


def train_agent(agent, env, num_episodes=1000, log_freq=100):
    """训练智能体"""
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            inventory = info.get('inventory', np.ones(agent.action_dim))
            mask = (inventory <= 0).astype(np.float32)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)

            action_logits, value = agent(state_tensor)
            action_logits = action_logits.masked_fill(mask_tensor.bool(), -float('inf'))

            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()

            action = np.zeros(agent.action_dim, dtype=np.float32)
            if action_idx.item() < agent.action_dim:
                action[action_idx.item()] = 1

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            trajectory_states.append(state.copy())
            trajectory_actions.append(action_idx.item())
            trajectory_rewards.append(reward)

            state = next_state
            info = next_info
            episode_reward += reward

        episode_rewards.append(episode_reward)

        if len(trajectory_states) > 0:
            update_agent_fixed(agent, trajectory_states, trajectory_actions, trajectory_rewards)

        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                        f"Avg Reward: {avg_reward:.2f}")

    return np.mean(episode_rewards[-100:])


def update_agent_fixed(agent, states, actions, rewards):
    """修复的智能体更新函数"""
    try:
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)

        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns)

        action_logits, values = agent(states_tensor)
        values = values.squeeze()

        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)

        advantages = returns_tensor - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = ((values - returns_tensor) ** 2).mean()
        entropy_loss = -dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        agent.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

        agent.optimizer.step()

    except Exception as e:
        logger.warning(f"更新智能体失败: {e}")


def evaluate_baseline(agent_class, env, num_episodes=100):
    """评估基准算法"""
    agent = agent_class(env.env.num_products, env.env.cardinality)
    total_rewards = []

    for _ in range(num_episodes):
        state, info = env.env.reset()
        episode_reward = 0
        done = False

        while not done:
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
                else:
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

    np.save(output_dir / f"results_{timestamp}.npy", results)

    with open(output_dir / f"summary_{timestamp}.txt", 'w') as f:
        f.write("实验结果摘要\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


def validate_generated_functions(state_func: str, reward_func: str) -> bool:
    """验证生成的函数质量"""
    issues = []
    warnings = []

    # === 检查状态函数 ===
    if 'def enhance_state' not in state_func:
        issues.append("状态函数缺少函数定义")

    if 'return' not in state_func:
        issues.append("状态函数没有return语句")

    # 检查是否有示例代码
    state_lines = state_func.split('\n')
    after_function = False
    function_indent = None

    for line in state_lines:
        if 'def enhance_state' in line:
            after_function = True
            function_indent = len(line) - len(line.lstrip())
            continue

        if after_function:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # 检查缩进
            current_indent = len(line) - len(line.lstrip())

            # 如果回到函数级别或更外层，且不是空行/注释
            if current_indent <= function_indent and stripped:
                if not stripped.startswith('def ') and not stripped.startswith('class '):
                    issues.append(f"状态函数后有模块级代码: {stripped[:50]}")
                    break

    # 检查特征数量（粗略估计）
    feature_count = state_func.count('features.append') + state_func.count('features.extend')
    if feature_count < 3:
        warnings.append("状态函数可能生成特征过少")

    # === 检查奖励函数 ===
    if 'def intrinsic_reward' not in reward_func:
        issues.append("奖励函数缺少函数定义")

    if 'return' not in reward_func:
        issues.append("奖励函数没有return语句")

    # 检查是否有示例代码
    reward_lines = reward_func.split('\n')
    after_function = False
    function_indent = None

    for line in reward_lines:
        if 'def intrinsic_reward' in line:
            after_function = True
            function_indent = len(line) - len(line.lstrip())
            continue

        if after_function:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            current_indent = len(line) - len(line.lstrip())

            if current_indent <= function_indent and stripped:
                if not stripped.startswith('def ') and not stripped.startswith('class '):
                    issues.append(f"奖励函数后有模块级代码: {stripped[:50]}")
                    break

    # 检查常见错误
    if 'sold_item * price' in reward_func:
        warnings.append("奖励函数可能错误使用 sold_item（它是索引不是数量）")

    if 'enhance_state' in reward_func:
        issues.append("奖励函数不应引用 enhance_state")

    # 输出结果
    if issues:
        logger.error(f"生成的函数有严重问题: {'; '.join(issues)}")
        return False

    if warnings:
        logger.warning(f"生成的函数有潜在问题: {'; '.join(warnings)}")

    return True


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = config['llm'].get('api_key')
    base_url = config['llm'].get('base_url')

    logger.info("=== 开始DRL-Assortment-LESR训练 ===")

    llm_optimizer = LLMOptimizer(
        api_key=api_key,
        base_url=base_url,
        model=config['llm']['model']
    )

    state_enhancer = StateEnhancer()
    reward_calculator = IntrinsicRewardCalculator()
    feedback_analyzer = FeedbackAnalyzer()

    base_env = AssortmentEnvironment(
        num_products=config['env']['num_products'],
        num_customer_types=config['env']['num_customer_types'],
        initial_inventory=np.ones(config['env']['num_products']) * config['env']['initial_inventory'],
        cardinality=config['env']['cardinality']
    )

    best_performance = -float('inf')
    best_state_func = None
    best_reward_func = None

    for iteration in range(config['training']['num_iterations']):
        logger.info(f"\n=== 迭代 {iteration + 1}/{config['training']['num_iterations']} ===")

        logger.info("生成状态表示函数...")
        state_functions = []
        reward_functions = []

        for sample_idx in range(config['llm']['samples_per_iteration']):
            logger.info(f"生成样本 {sample_idx + 1}/{config['llm']['samples_per_iteration']}")

            try:
                state_func = llm_optimizer.generate_state_representation(
                    task_description=config['task']['description'],
                    state_info={
                        'inventory_shape': config['env']['num_products'],
                        'customer_types': config['env']['num_customer_types'],
                        'num_products': config['env']['num_products'],
                        'cardinality': config['env']['cardinality']
                    }
                )

                reward_func = llm_optimizer.generate_intrinsic_reward(
                    state_representation=state_func,
                    performance_feedback=feedback_analyzer.get_serializable_feedback()
                )

                # 🔧 新增：立即验证
                if not validate_generated_functions(state_func, reward_func):
                    logger.warning(f"样本 {sample_idx + 1} 验证失败，跳过")
                    continue

                # 🔧 新增：尝试预加载测试
                test_state_enhancer = StateEnhancer()
                test_reward_calc = IntrinsicRewardCalculator()

                if not test_state_enhancer.load_function(state_func):
                    logger.warning(f"样本 {sample_idx + 1} 状态函数加载失败")
                    continue

                if not test_reward_calc.load_function(reward_func):
                    logger.warning(f"样本 {sample_idx + 1} 奖励函数加载失败")
                    continue

                # 验证通过，添加到列表
                state_functions.append(state_func)
                reward_functions.append(reward_func)

                # 🔧 新增：保存生成的函数（用于调试）
                if config.get('debug', {}).get('save_generated_functions', True):
                    debug_dir = output_dir / "generated_functions" / f"iter_{iteration + 1}"
                    debug_dir.mkdir(parents=True, exist_ok=True)

                    with open(debug_dir / f"sample_{sample_idx + 1}_state.py", 'w') as f:
                        f.write(state_func)
                    with open(debug_dir / f"sample_{sample_idx + 1}_reward.py", 'w') as f:
                        f.write(reward_func)

            except Exception as e:
                logger.error(f"生成样本 {sample_idx + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()

        if not state_functions:
            logger.warning("使用默认状态增强函数")
            state_functions = [llm_optimizer._get_default_state_function()]
            reward_functions = [llm_optimizer._get_default_reward_function()]

        performances = []

        for idx in range(len(state_functions)):
            logger.info(f"\n训练样本 {idx + 1}/{len(state_functions)}")

            try:
                state_loaded = state_enhancer.load_function(state_functions[idx])
                reward_loaded = reward_calculator.load_function(reward_functions[idx])

                if not (state_loaded and reward_loaded):
                    logger.warning(f"样本 {idx + 1} 函数加载失败")
                    performances.append(0.0)
                    continue

                enhanced_env = EnhancedEnvironmentWrapper(
                    base_env, state_enhancer, reward_calculator
                )

                test_state, _ = enhanced_env.reset()
                state_dim = len(test_state)

                logger.info(f"状态维度: {state_dim}")

                # 🔧 检查状态维度
                if state_dim < 10:
                    logger.warning(f"状态维度太小({state_dim})，可能影响性能")

                agent = A2CAgent(
                    state_dim=state_dim,
                    action_dim=config['env']['num_products'],
                    hidden_dim=config['agent']['hidden_dim'],
                    learning_rate=config['agent']['learning_rate']
                )

                performance = train_agent(
                    agent,
                    enhanced_env,
                    num_episodes=min(200, config['training']['episodes_per_sample']),
                    log_freq=50
                )

                # 🔧 检测异常性能
                if performance > 10000:
                    logger.warning(f"样本 {idx + 1} 性能异常高({performance:.2f})，可能奖励爆炸")
                    performance = 0.0

                performances.append(performance)
                logger.info(f"样本 {idx + 1} 性能: {performance:.2f}")

            except Exception as e:
                logger.error(f"训练样本 {idx + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                performances.append(0.0)

        if performances:
            feedback = feedback_analyzer.analyze_performance(
                performances,
                state_functions
            )

            best_idx = np.argmax(performances)
            if performances[best_idx] > best_performance and performances[best_idx] < 10000:
                best_performance = performances[best_idx]
                best_state_func = state_functions[best_idx]
                best_reward_func = reward_functions[best_idx]

                with open(output_dir / "best_state_func.py", 'w') as f:
                    f.write(best_state_func)
                with open(output_dir / "best_reward_func.py", 'w') as f:
                    f.write(best_reward_func)

                logger.info(f"新最佳性能: {best_performance:.2f}")

            llm_optimizer.update_with_feedback(feedback)

    logger.info("\n=== 评估基准算法 ===")

    if best_state_func and best_reward_func:
        state_enhancer.load_function(best_state_func)
        reward_calculator.load_function(best_reward_func)
    final_env = EnhancedEnvironmentWrapper(base_env, state_enhancer, reward_calculator)

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

    save_results(results, output_dir)

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        main(args)
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        import traceback

        traceback.print_exc()
        raise