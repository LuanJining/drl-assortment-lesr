import argparse
import yaml
import numpy as np
from pathlib import Path
from core.llm_optimizer import LLMOptimizer
from rl.a2c_agent import A2CAgent
from rl.environment import AssortmentEnvironment
from utils.evaluation import evaluate_policy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化LLM优化器
    llm_optimizer = LLMOptimizer(
        api_key=args.api_key,
        model=config['llm']['model']
    )
    
    # 创建环境
    env = AssortmentEnvironment(
        num_products=config['env']['num_products'],
        num_customer_types=config['env']['num_customer_types'],
        cardinality=config['env']['cardinality']
    )
    
    best_performance = -float('inf')
    
    # 迭代优化循环
    for iteration in range(config['training']['num_iterations']):
        logger.info(f"=== 迭代 {iteration + 1}/{config['training']['num_iterations']} ===")
        
        # 步骤1：生成状态表示函数
        logger.info("生成状态表示函数...")
        state_functions = []
        for i in range(config['llm']['samples_per_iteration']):
            state_func = llm_optimizer.generate_state_representation(
                task_description=config['task']['description'],
                state_info={
                    'inventory_shape': env.num_products,
                    'customer_types': env.num_customer_types,
                    'num_products': env.num_products
                }
            )
            state_functions.append(state_func)
        
        # 步骤2：训练和评估每个状态表示
        performances = []
        for i, state_func in enumerate(state_functions):
            logger.info(f"训练样本 {i + 1}/{len(state_functions)}")
            
            # 创建增强环境
            enhanced_env = create_enhanced_environment(env, state_func)
            
            # 创建智能体
            agent = A2CAgent(
                state_dim=enhanced_env.observation_space.shape[0],
                action_dim=env.num_products,
                hidden_dim=config['agent']['hidden_dim']
            )
            
            # 训练
            performance = train_agent(
                agent, 
                enhanced_env,
                num_episodes=config['training']['episodes_per_sample']
            )
            
            performances.append(performance)
            logger.info(f"样本 {i + 1} 性能: {performance:.2f}")
        
        # 步骤3：选择最佳性能并生成反馈
        best_idx = np.argmax(performances)
        current_best = performances[best_idx]
        
        if current_best > best_performance:
            best_performance = current_best
            best_state_func = state_functions[best_idx]
            save_best_model(best_state_func, iteration)
            logger.info(f"新最佳性能: {best_performance:.2f}")
        
        # 步骤4：生成改进建议
        logger.info("生成改进建议...")
        feedback = analyze_performance(performances, state_functions)
        llm_optimizer.update_with_feedback(feedback)
    
    logger.info(f"训练完成！最佳性能: {best_performance:.2f}")

def train_agent(agent, env, num_episodes):
    """训练智能体"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新智能体
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            logger.info(f"Episode {episode + 1}, 平均奖励: {avg_reward:.2f}")
    
    return np.mean(total_rewards[-100:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_config.yaml')
    parser.add_argument('--api-key', required=True, help='OpenAI API密钥')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)