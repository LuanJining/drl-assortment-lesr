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
    """训练智能体"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        while not done:
            # 获取动作和价值
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_logits, value = agent(state_tensor)
                
                # 获取有效动作掩码
                mask = (info['inventory'] == 0).astype(np.float32)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
                
                # 应用掩码
                action_logits = action_logits.masked_fill(mask_tensor.bool(), -float('inf'))
                
                # 采样动作
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
                
                # 转换为二进制动作
                action = np.zeros(agent.action_dim)
                if action_idx.item() < agent.action_dim:
                    action[action_idx.item()] = 1
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录轨迹
            states.append(state)
            actions.append(action_idx.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # 更新智能体
        if len(states) > 0:
            update_agent(agent, states, actions, rewards, values, log_probs)
        
        # 日志记录
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                       f"Avg Reward: {avg_reward:.2f}")
    
    return np.mean(episode_rewards[-100:])

def update_agent(agent, states, actions, rewards, values, log_probs):
    """更新A2C智能体"""
    # 计算回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    values = torch.FloatTensor(values)
    log_probs = torch.stack(log_probs)
    
    # 计算优势
    advantages = returns - values
    
    # 计算损失
    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    
    # 总损失
    loss = actor_loss + 0.5 * critic_loss
    
    # 反向传播
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

def evaluate_baseline(agent_class, env, num_episodes=100):
    """评估基准算法"""
    agent = agent_class(env.num_products, env.cardinality)
    total_rewards = []
    
    for _ in range(num_episodes):
        state, info = env.env.reset()  # 使用原始环境
        episode_reward = 0
        done = False
        
        while not done:
            # 基准算法决策
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
            state_functions.append(state_func)
            
            # 生成奖励函数
            reward_func = llm_optimizer.generate_intrinsic_reward(
                state_representation=state_func,
                performance_feedback=None if iteration == 0 else feedback_analyzer.performance_history[-1]
            )
            reward_functions.append(reward_func)
        
        # Step 2: 训练和评估
        performances = []
        lipschitz_constants = []
        
        for idx in range(len(state_functions)):
            logger.info(f"\n训练样本 {idx + 1}/{len(state_functions)}")
            
            # 加载函数
            state_enhancer.load_function(state_functions[idx])
            reward_calculator.load_function(reward_functions[idx])
            
            # 创建增强环境
            enhanced_env = EnhancedEnvironmentWrapper(
                base_env, state_enhancer, reward_calculator
            )
            
            # 获取状态维度
            test_state, _ = enhanced_env.reset()
            state_dim = len(test_state)
            
            # 创建智能体
            agent = A2CAgent(
                state_dim=state_dim,
                action_dim=config['env']['num_products'],
                hidden_dim=config['agent']['hidden_dim'],
                learning_rate=config['agent']['learning_rate']
            )
            
            # 训练
            performance = train_agent(
                agent, 
                enhanced_env,
                num_episodes=config['training']['episodes_per_sample']
            )
            
            performances.append(performance)
            logger.info(f"样本 {idx + 1} 性能: {performance:.2f}")
        
        # Step 3: 分析反馈
        feedback = feedback_analyzer.analyze_performance(
            performances, 
            state_functions,
            lipschitz_constants
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
        mean_reward, std_reward = evaluate_baseline(agent_class, final_env)
        results[f'{name}_mean'] = mean_reward
        results[f'{name}_std'] = std_reward
        logger.info(f"{name} Agent: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 保存结果
    save_results(results, output_dir)
    
    # 显示改进
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
    main(args)