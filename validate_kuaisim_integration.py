# validate_kuaisim_integration.py
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import yaml
import torch
import importlib.util


def validate_kuaisim_integration(config_path="config/kuaisim_config.yaml"):
    """验证KuaiSim集成是否正确配置"""

    print("=== KuaiSim集成验证工具 ===\n")

    # 加载配置
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        kuaisim_config = config['kuaisim']
        print(f"✓ 配置文件加载成功: {config_path}")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

    validation_results = {}

    # 1. 验证KuaiSim源码
    print(f"\n1. 验证KuaiSim源码")
    kuaisim_path = kuaisim_config.get('kuaisim_path')

    if kuaisim_path and kuaisim_path.strip():
        kuaisim_path = Path(kuaisim_path)
        if kuaisim_path.exists():
            print(f"   ✓ KuaiSim路径存在: {kuaisim_path}")

            # 检查关键文件
            key_files = {
                'reader.py': kuaisim_path / 'reader.py',
                'simulator': kuaisim_path / 'model' / 'simulator.py',
                'environment': kuaisim_path / 'env' / '__init__.py',
                'utils': kuaisim_path / 'utils.py'
            }

            missing_files = []
            for name, file_path in key_files.items():
                if file_path.exists():
                    print(f"   ✓ {name} 存在")
                else:
                    print(f"   ✗ {name} 缺失: {file_path}")
                    missing_files.append(name)

            if not missing_files:
                validation_results['kuaisim_source'] = True
                print(f"   ✓ KuaiSim源码文件完整")

                # 测试导入
                if _test_kuaisim_import(kuaisim_path):
                    print(f"   ✓ KuaiSim模块导入成功")
                    validation_results['kuaisim_import'] = True
                else:
                    print(f"   ✗ KuaiSim模块导入失败")
                    validation_results['kuaisim_import'] = False
            else:
                validation_results['kuaisim_source'] = False
                print(f"   ✗ KuaiSim源码不完整，缺失: {missing_files}")
        else:
            print(f"   ✗ KuaiSim路径不存在: {kuaisim_path}")
            validation_results['kuaisim_source'] = False
    else:
        print(f"   - 未配置KuaiSim源码路径（将使用数据驱动模式）")
        validation_results['kuaisim_source'] = None

    # 2. 验证数据集
    print(f"\n2. 验证KuaiSim数据集")
    data_path = Path(kuaisim_config.get('data_path', 'dataset/Kuairand-Pure/'))

    if data_path.exists():
        print(f"   ✓ 数据路径存在: {data_path}")

        # 验证数据文件
        data_files = {
            'train_file': kuaisim_config.get('train_file', 'log_session_4_08_to_5_08_Pure.csv'),
            'user_meta_file': kuaisim_config.get('user_meta_file', 'user_features_Pure_fillna.csv'),
            'item_meta_file': kuaisim_config.get('item_meta_file', 'video_features_basic_Pure_fillna.csv')
        }

        data_status = {}
        for file_type, filename in data_files.items():
            file_path = data_path / filename
            if file_path.exists():
                print(f"   ✓ {file_type}: {filename}")

                # 验证文件内容
                try:
                    if file_type == 'train_file':
                        df = pd.read_csv(file_path, nrows=100)
                        expected_cols = ['user_id', 'video_id', 'session_id']
                        missing_cols = [col for col in expected_cols if col not in df.columns]
                        if missing_cols:
                            print(f"     ⚠ 缺少必要列: {missing_cols}")
                        else:
                            print(f"     ✓ 数据格式正确，包含 {len(df)} 行样本")
                            data_status[file_type] = True
                    else:
                        df = pd.read_csv(file_path, nrows=10)
                        print(f"     ✓ 文件可读，包含 {len(df.columns)} 列")
                        data_status[file_type] = True

                except Exception as e:
                    print(f"     ✗ 文件读取失败: {e}")
                    data_status[file_type] = False
            else:
                print(f"   ✗ {file_type}: {filename} 不存在")
                data_status[file_type] = False

        validation_results['data_files'] = all(data_status.values())
    else:
        print(f"   ✗ 数据路径不存在: {data_path}")
        validation_results['data_files'] = False

    # 3. 验证预训练模型
    print(f"\n3. 验证预训练模型")
    model_log_path = kuaisim_config.get('model_log_path')

    if model_log_path and os.path.exists(model_log_path):
        print(f"   ✓ 模型日志文件存在: {model_log_path}")

        try:
            # 验证日志文件格式
            with open(model_log_path, 'r') as f:
                first_line = f.readline()
                second_line = f.readline()

            # 检查是否是KuaiSim模型日志格式
            if 'reader' in first_line and 'model' in first_line:
                print(f"   ✓ 模型日志格式正确")

                # 检查对应的模型文件
                model_file_path = model_log_path.replace('.model.log', '.model')
                if os.path.exists(model_file_path):
                    print(f"   ✓ 模型权重文件存在")
                    validation_results['pretrained_model'] = True
                else:
                    print(f"   ⚠ 模型权重文件不存在: {model_file_path}")
                    validation_results['pretrained_model'] = False
            else:
                print(f"   ✗ 模型日志格式不正确")
                validation_results['pretrained_model'] = False

        except Exception as e:
            print(f"   ✗ 模型日志文件读取失败: {e}")
            validation_results['pretrained_model'] = False
    else:
        print(f"   - 未配置预训练模型路径（将使用数据驱动模式）")
        validation_results['pretrained_model'] = None

    # 4. 验证PyTorch环境
    print(f"\n4. 验证PyTorch环境")
    try:
        import torch
        print(f"   ✓ PyTorch版本: {torch.__version__}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✓ 可用设备: {device}")

        if torch.cuda.is_available():
            print(f"   ✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")

        validation_results['pytorch'] = True

    except Exception as e:
        print(f"   ✗ PyTorch环境验证失败: {e}")
        validation_results['pytorch'] = False

    # 5. 测试适配器创建
    print(f"\n5. 测试适配器创建")
    try:
        from adapters.kuaisim_adapter import create_kuaisim_environment

        # 创建测试配置
        test_config = {
            'num_products': 10,
            'cardinality': 3,
            'max_episodes': 5,
            'max_steps': 10,
            'kuaisim_path': kuaisim_config.get('kuaisim_path'),
            'data_path': kuaisim_config.get('data_path'),
            'model_log_path': kuaisim_config.get('model_log_path'),
            'device': 'cpu'  # 强制使用CPU进行测试
        }

        # 创建环境
        env = create_kuaisim_environment(test_config)
        print("   ✓ 适配器创建成功")

        # 测试重置
        state, info = env.reset()
        print(f"   ✓ 环境重置成功，状态维度: {len(state)}")

        # 显示环境状态
        kuaisim_status = info.get('kuaisim_status', '未知')
        print(f"   ✓ KuaiSim状态: {kuaisim_status}")

        # 测试动作
        action = np.zeros(test_config['num_products'])
        action[0] = 1  # 选择第一个物品

        next_state, reward, terminated, truncated, next_info = env.step(action)
        print(f"   ✓ 动作执行成功，奖励: {reward:.3f}")

        validation_results['adapter'] = True

    except Exception as e:
        print(f"   ✗ 适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        validation_results['adapter'] = False

    # 6. 总结验证结果
    print(f"\n=== 验证总结 ===")

    # 统计结果
    total_checks = len([v for v in validation_results.values() if v is not None])
    passed_checks = len([v for v in validation_results.values() if v is True])

    print(f"总检查项: {total_checks}")
    print(f"通过检查: {passed_checks}")

    # 详细状态
    if validation_results.get('kuaisim_source', False) and validation_results.get('kuaisim_import', False):
        print("✓ KuaiSim源码可用且可导入")
        mode = "完整KuaiSim模式"
    elif validation_results.get('pretrained_model', False):
        print("✓ 预训练模型可用")
        mode = "用户模型模式"
    elif validation_results.get('data_files', False):
        print("✓ 数据文件可用")
        mode = "数据驱动模式"
    else:
        print("⚠ 将使用完全模拟模式")
        mode = "完全模拟模式"

    print(f"推荐运行模式: {mode}")

    # 运行建议
    print(f"\n=== 运行建议 ===")

    if validation_results.get('adapter', False):
        print("✓ 可以运行KuaiSim集成")
        print("  命令: python main_kuaisim.py --config config/kuaisim_config.yaml")

        if not validation_results.get('kuaisim_source', False):
            print("💡 提示: 配置KuaiSim源码路径可启用完整功能")

        if not validation_results.get('pretrained_model', False):
            print("💡 提示: 训练用户响应模型可提升真实性")

    else:
        print("✗ 无法运行，请检查以下问题:")

        if not validation_results.get('pytorch', False):
            print("  - 安装PyTorch: pip install torch")

        if not validation_results.get('adapter', False):
            print("  - 检查适配器代码是否正确")

    return validation_results.get('adapter', False)


def _test_kuaisim_import(kuaisim_path):
    """测试KuaiSim模块导入"""
    try:
        # 临时添加路径
        sys.path.insert(0, str(kuaisim_path))

        # 尝试导入关键模块
        import reader
        import utils

        # 尝试导入模型模块
        from model import simulator

        # 移除临时路径
        sys.path.remove(str(kuaisim_path))

        return True

    except Exception as e:
        # 移除临时路径
        if str(kuaisim_path) in sys.path:
            sys.path.remove(str(kuaisim_path))
        return False


def suggest_kuaisim_setup():
    """提供KuaiSim设置建议"""
    print(f"\n=== KuaiSim设置建议 ===")
    print("1. 克隆KuaiSim源码:")
    print("   git clone https://github.com/CharlieMat/KRLBenchmark.git")
    print()
    print("2. 安装依赖:")
    print("   conda install pytorch pandas scikit-learn tqdm")
    print()
    print("3. 下载数据集:")
    print("   # 从KuaiSim官方获取Kuairand-Pure数据集")
    print()
    print("4. 训练用户模型:")
    print("   cd KRLBenchmark/code")
    print("   bash run_multibehavior.sh")
    print()
    print("5. 更新配置文件:")
    print("   # 设置kuaisim_path指向KRLBenchmark/code")
    print("   # 设置model_log_path指向训练好的模型")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='验证KuaiSim集成')
    parser.add_argument('--config', default='config/kuaisim_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--setup-guide', action='store_true',
                        help='显示设置指南')
    args = parser.parse_args()

    if args.setup_guide:
        suggest_kuaisim_setup()
        sys.exit(0)

    # 验证集成
    success = validate_kuaisim_integration(args.config)

    if not success:
        print(f"\n运行设置指南: python {__file__} --setup-guide")

    sys.exit(0 if success else 1)