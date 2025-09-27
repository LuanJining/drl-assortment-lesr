# DRL-Assortment-LESR

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **D**eep **R**einforcement **L**earning for **Assortment** optimization with **L**LM-**E**nhanced **S**tate **R**epresentation

一个创新的强化学习项目，结合大语言模型(LLM)自动生成状态表示和奖励函数，用于优化在线商品推荐策略。

## 🎯 项目概述

DRL-Assortment-LESR 是一个端到端的强化学习系统，专门解决电商场景中的商品推荐优化问题。项目的核心创新在于使用大语言模型自动生成和优化状态表示函数与内在奖励函数，从而提升强化学习算法的性能。

### ✨ 主要特性

- 🤖 **LLM增强的状态表示**：自动生成领域特定的状态特征
- 🎁 **智能奖励设计**：LLM生成平衡多目标的内在奖励函数
- 🏆 **多算法对比**：内置Random、Myopic、EIB等基准算法
- 📊 **完整评估体系**：支持多维度性能分析和可视化
- ⚙️ **灵活配置**：YAML配置文件，支持快速实验调整
- 🔌 **API集成**：支持ChatAnywhere等第三方LLM API服务
- 📈 **实时监控**：训练过程可视化和性能追踪

### 🎪 应用场景

- 电商平台商品推荐优化
- 库存管理与销售策略
- 个性化营销策略制定
- 供应链优化决策

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib
- OpenAI API Key (推荐使用ChatAnywhere)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/drl-assortment-lesr.git
   cd drl-assortment-lesr
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置API密钥**
   
   编辑 `config/train_config.yaml`：
   ```yaml
   llm:
     api_key: "your-api-key-here"
     base_url: "https://api.chatanywhere.tech/v1"
     model: "gpt-4o-mini"
   ```

### 运行示例

1. **测试API连接**
   ```bash
   python api_test_script.py
   ```

2. **开始训练**
   ```bash
   python main.py --config config/train_config.yaml
   ```

3. **查看结果**
   ```bash
   # 训练完成后，结果保存在 results/ 目录
   ls results/
   ```

## 📁 项目结构

```
drl-assortment-lesr/
├── 📁 core/                    # 核心模块
│   ├── llm_optimizer.py        # LLM优化器
│   ├── state_enhancer.py       # 状态增强器
│   ├── intrinsic_reward.py     # 内在奖励计算器
│   └── feedback_analyzer.py    # 反馈分析器
├── 📁 rl/                      # 强化学习模块
│   ├── a2c_agent.py           # A2C智能体
│   ├── environment.py         # 环境定义
│   ├── networks.py            # 神经网络架构
│   └── replay_buffer.py       # 经验回放缓冲区
├── 📁 baselines/               # 基准算法
│   ├── random_agent.py        # 随机算法
│   ├── myopic_agent.py        # 贪心算法
│   └── eib_agent.py           # 库存平衡算法
├── 📁 utils/                   # 工具模块
│   ├── evaluation.py          # 评估工具
│   ├── visualization.py       # 可视化工具
│   └── data_generator.py      # 数据生成器
├── 📁 config/                  # 配置文件
│   ├── train_config.yaml      # 训练配置
│   ├── env_config.yaml        # 环境配置
│   └── llm_prompts.yaml       # LLM提示模板
├── 📄 main.py                 # 主程序入口
├── 📄 requirements.txt        # 依赖列表
└── 📄 README.md              # 项目文档
```

## ⚙️ 配置说明

### 训练配置 (`config/train_config.yaml`)

```yaml
# LLM配置
llm:
  model: "gpt-4o-mini"           # 推荐使用价格便宜的模型
  temperature: 0.7               # 创造性参数
  samples_per_iteration: 3       # 每轮迭代的采样数
  max_iterations: 3              # 最大迭代次数

# 环境配置
env:
  num_products: 10               # 产品数量
  num_customer_types: 4          # 客户类型数
  cardinality: 4                 # 最大展示商品数
  
# 训练配置
training:
  episodes_per_sample: 200       # 每个样本的训练轮数
  num_iterations: 3              # 总迭代次数
```

### 环境配置 (`config/env_config.yaml`)

```yaml
environment:
  products:
    num_products: 10
    price_range: [1.0, 5.0]      # 价格范围
    
  customers:
    num_types: 4
    preference_type: "dirichlet"  # 偏好分布类型
    
  constraints:
    cardinality: 4                # 展示限制
    max_time: 100                 # 最大时间步
```

## 🎮 使用示例

### 基础训练

```python
import yaml
from core.llm_optimizer import LLMOptimizer
from rl.a2c_agent import A2CAgent
from rl.environment import AssortmentEnvironment

# 加载配置
with open('config/train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化组件
llm_optimizer = LLMOptimizer(
    api_key=config['llm']['api_key'],
    base_url=config['llm']['base_url'],
    model=config['llm']['model']
)

# 生成状态表示函数
state_func = llm_optimizer.generate_state_representation(
    task_description="优化商品推荐策略",
    state_info={'num_products': 10, 'customer_types': 4}
)

# 开始训练...
```

### 自定义状态特征

```python
# 手动定义状态增强函数
def custom_enhance_state(inventory, customer_type, prices, time_remaining, initial_inventory):
    # 添加自定义特征
    features = []
    
    # 基础特征
    relative_inventory = inventory / (initial_inventory + 1e-8)
    features.extend(relative_inventory)
    
    # 自定义特征
    urgency_score = calculate_urgency(inventory, time_remaining)
    features.append(urgency_score)
    
    return np.array(features)
```

### 评估和对比

```python
from utils.evaluation import Evaluator

evaluator = Evaluator()

# 对比多个算法
algorithms = {
    'DRL': trained_agent,
    'Random': RandomAgent(num_products=10, cardinality=4),
    'Myopic': MyopicAgent(num_products=10, cardinality=4)
}

results = evaluator.compare_algorithms(env, algorithms, num_episodes=100)
print(results)
```

## 📊 结果分析

训练完成后，系统会生成以下输出：

### 性能指标

- **总收益 (Total Revenue)**：销售总收入
- **客户满意度 (Customer Satisfaction)**：成功购买率
- **库存平衡度 (Inventory Balance)**：库存分布均衡性
- **缺货率 (Stockout Rate)**：缺货发生频率
- **库存周转率 (Inventory Turnover)**：库存使用效率

### 可视化报告

```python
from utils.visualization import Visualizer

vis = Visualizer()

# 绘制学习曲线
vis.plot_learning_curve(episode_rewards, save_path='results/learning_curve.png')

# 绘制算法对比
vis.plot_comparison(comparison_df, save_path='results/algorithm_comparison.png')

# 生成交互式面板
dashboard = vis.create_interactive_dashboard(results_data)
dashboard.show()
```

### 示例结果

```
📊 实验结果摘要
==================================================
DRL-LESR性能: 245.67 ± 12.34
Random基准: 180.23 ± 15.67
Myopic基准: 210.45 ± 11.89
EIB基准: 225.12 ± 13.45

相对于Random提升: +36.3%
相对于最佳基准提升: +9.1%
```

## 🛠️ 故障排除

### 常见问题

1. **API连接失败**
   ```bash
   # 测试API连接
   python api_test_script.py
   ```
   - 检查API Key是否正确
   - 确认网络连接正常
   - 验证base_url格式

2. **内存不足**
   ```yaml
   # 减少训练参数
   training:
     episodes_per_sample: 100  # 从200减少到100
     batch_size: 16           # 从32减少到16
   ```

3. **训练太慢**
   ```yaml
   # 减少迭代次数
   llm:
     samples_per_iteration: 2  # 从3减少到2
     max_iterations: 2         # 从3减少到2
   ```

4. **CUDA相关错误**
   ```bash
   # 强制使用CPU
   export CUDA_VISIBLE_DEVICES=""
   python main.py --config config/train_config.yaml
   ```

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black .
flake8 .
```

## 📝 更新日志

### v1.0.0 (2024-01-XX)
- 🎉 首次发布
- ✨ LLM增强的状态表示
- 🚀 A2C算法实现
- 📊 完整评估系统

### 计划功能
- [ ] 支持更多RL算法 (PPO, SAC)
- [ ] 多智能体协作
- [ ] 分布式训练支持
- [ ] Web界面管理

## 📄 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- 作者：栾吉宁
- 邮箱：luanjining@163.com
- 项目链接：[https://github.com/your-username/drl-assortment-lesr](https://github.com/LuanJining/drl-assortment-lesr)

## 🙏 致谢

感谢以下项目和技术的支持：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenAI](https://openai.com/) - GPT API服务
- [ChatAnywhere](https://api.chatanywhere.tech/) - API服务提供商
- [Gymnasium](https://gymnasium.farama.org/) - 强化学习环境

---

⭐ 如果这个项目对你有帮助，请给我们一个 Star！

💡 有问题或建议？欢迎提交 [Issue](https://github.com/your-username/drl-assortment-lesr/issues)！
