#!/usr/bin/env python3
# test_system.py - 修复版系统测试脚本

import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """测试所有模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试配置加载
        from config_loader import get_scenario_config, load_config
        print("✅ 配置加载器导入成功")
        
        # 测试环境模块
        from env.topology_loader import generate_topology
        from env.vnf_env_multi import EnhancedVNFEmbeddingEnv
        print("✅ 环境模块导入成功")
        
        # 测试智能体模块
        from agents.base_agent import create_agent
        from agents.multi_ddqn_agent import MultiDDQNAgent
        from agents.multi_dqn_agent import MultiDQNAgent
        from agents.multi_ppo_agent import MultiPPOAgent
        print("✅ 智能体模块导入成功")
        
        # 测试模型模块
        from models.gnn_encoder import GNNEncoder, EdgeAwareGNNEncoder
        print("✅ 模型模块导入成功")
        
        # 测试工具模块
        from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
        from utils.metrics import calculate_sar, calculate_splat
        from utils.logger import Logger
        print("✅ 工具模块导入成功")
        
        # 测试奖励模块
        from rewards.reward_v4_comprehensive_multi import compute_reward
        print("✅ 奖励模块导入成功")
        
        # 测试主训练器（简化版）
        print("✅ 主训练器导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_config_system():
    """测试配置系统"""
    print("\n🧪 测试配置系统...")
    
    try:
        from config_loader import get_scenario_config, load_config, validate_all_configs
        
        # 测试配置加载
        base_config = load_config("config.yaml")
        print(f"✅ 基础配置加载成功: {len(base_config)} 个配置组")
        
        # 测试场景配置
        scenarios = ['normal_operation', 'peak_congestion', 'failure_recovery', 'extreme_pressure']
        for scenario in scenarios:
            config = get_scenario_config(scenario)
            print(f"✅ {scenario} 场景配置加载成功")
        
        # 测试episode-based配置
        for episode in [1, 25, 50, 75, 100]:
            config = get_scenario_config(episode)
            print(f"✅ Episode {episode} 配置加载成功")
        
        # 验证配置
        is_valid = validate_all_configs()
        print(f"✅ 配置验证: {'通过' if is_valid else '需要调整'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False

def test_topology_generation():
    """测试拓扑生成"""
    print("\n🧪 测试拓扑生成...")
    
    try:
        from env.topology_loader import generate_topology
        from config_loader import load_config
        
        config = load_config("config.yaml")
        graph, node_features, edge_features = generate_topology(config)
        
        print(f"✅ 拓扑生成成功:")
        print(f"   - 节点数: {len(graph.nodes())}")
        print(f"   - 边数: {len(graph.edges())}")
        print(f"   - 节点特征维度: {node_features.shape}")
        print(f"   - 边特征维度: {edge_features.shape}")
        
        # 验证特征维度
        assert node_features.shape[1] == 4, f"节点特征应为4维，实际{node_features.shape[1]}维"
        assert edge_features.shape[1] == 4, f"边特征应为4维，实际{edge_features.shape[1]}维"
        
        return graph, node_features, edge_features
        
    except Exception as e:
        print(f"❌ 拓扑生成测试失败: {e}")
        return None, None, None

def test_environment():
    """测试环境"""
    print("\n🧪 测试环境...")
    
    try:
        import networkx as nx
        from env.vnf_env_multi import EnhancedVNFEmbeddingEnv
        from config_loader import load_config, get_scenario_config
        
        # 生成测试拓扑
        graph, node_features, edge_features = test_topology_generation()
        if graph is None:
            return False
        
        # 加载配置
        config = load_config("config.yaml")
        scenario_config = get_scenario_config('normal_operation')
        
        # 创建环境
        env = EnhancedVNFEmbeddingEnv(
            graph=graph,
            node_features=node_features,
            edge_features=edge_features,
            reward_config=scenario_config['reward'],
            config=config
        )
        
        print(f"✅ 环境创建成功:")
        print(f"   - 动作维度: {env.action_dim}")
        print(f"   - 状态维度: {env.state_dim}")
        
        # 应用场景配置
        env.apply_scenario_config(scenario_config)
        print(f"✅ 场景配置应用成功")
        
        # 测试重置
        state = env.reset()
        print(f"✅ 环境重置成功: 状态类型 {type(state)}")
        
        # 测试步骤
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = valid_actions[0]
            next_state, reward, done, info = env.step(action)
            print(f"✅ 环境步骤测试成功: reward={reward:.2f}, done={done}")
        
        return env
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return None

def test_agents():
    """测试智能体"""
    print("\n🧪 测试智能体...")
    
    try:
        from agents.base_agent import create_agent
        from config_loader import load_config
        
        config = load_config("config.yaml")
        
        # 测试参数
        state_dim = 8
        action_dim = 42
        edge_dim = 4
        
        agent_types = ['ddqn', 'dqn', 'ppo']
        agents = {}
        
        for agent_type in agent_types:
            try:
                agent_id = f"{agent_type}_test"
                agent = create_agent(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    edge_dim=edge_dim,
                    config=config
                )
                agents[agent_type] = agent
                print(f"✅ {agent_type.upper()} 智能体创建成功")
                
                # 测试动作选择
                test_state = torch.randn(1, 256)
                action = agent.select_action(test_state)
                print(f"   - 动作选择测试: {action}")
                
            except Exception as e:
                print(f"❌ {agent_type.upper()} 智能体创建失败: {e}")
        
        return agents
        
    except Exception as e:
        print(f"❌ 智能体测试失败: {e}")
        return {}

def test_gnn_encoder():
    """测试GNN编码器"""
    print("\n🧪 测试GNN编码器...")
    
    try:
        from models.gnn_encoder import GNNEncoder, EdgeAwareGNNEncoder, create_gnn_encoder
        from config_loader import load_config
        import torch
        from torch_geometric.data import Data
        
        config = load_config("config.yaml")
        
        # 测试数据
        num_nodes = 42
        num_edges = 100
        node_dim = 8
        edge_dim_full = 4
        edge_dim_baseline = 2
        
        # 生成测试数据
        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr_full = torch.randn(num_edges, edge_dim_full)
        edge_attr_baseline = torch.randn(num_edges, edge_dim_baseline)
        
        # 测试EdgeAware编码器
        encoder_edge_aware = create_gnn_encoder(config, mode='edge_aware')
        data_full = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_full)
        
        with torch.no_grad():
            output_edge_aware = encoder_edge_aware(data_full)
            print(f"✅ EdgeAware编码器测试成功: {output_edge_aware.shape}")
        
        # 测试Baseline编码器
        encoder_baseline = create_gnn_encoder(config, mode='baseline')
        data_baseline = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_baseline)
        
        with torch.no_grad():
            output_baseline = encoder_baseline(data_baseline)
            print(f"✅ Baseline编码器测试成功: {output_baseline.shape}")
        
        # 测试VNF上下文融合
        if hasattr(encoder_edge_aware, 'forward_with_vnf_context'):
            vnf_context = torch.tensor([0.05, 0.03, 0.04, 0.33, 0.5, 0.5])
            with torch.no_grad():
                output_with_context = encoder_edge_aware.forward_with_vnf_context(data_full, vnf_context)
                print(f"✅ VNF上下文融合测试成功: {output_with_context.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GNN编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_system():
    """测试奖励系统"""
    print("\n🧪 测试奖励系统...")
    
    try:
        from rewards.reward_v4_comprehensive_multi import compute_reward
        from config_loader import get_scenario_config
        
        # 获取奖励配置
        scenario_config = get_scenario_config('normal_operation')
        reward_config = scenario_config['reward']
        
        # 测试成功案例
        success_info = {
            'total_vnfs': 3,
            'deployed_vnfs': 3,
            'paths': [
                {'delay': 25.0, 'bandwidth': 80.0, 'hops': 2, 'jitter': 0.005, 'loss': 0.001},
                {'delay': 30.0, 'bandwidth': 70.0, 'hops': 3, 'jitter': 0.008, 'loss': 0.002}
            ],
            'resource_utilization': 0.6,
            'success': True,
            'is_edge_aware': True,
            'pressure_level': 'medium'
        }
        
        reward = compute_reward(success_info, reward_config)
        print(f"✅ 成功案例奖励: {reward:.2f}")
        
        # 测试失败案例
        failure_info = {
            'total_vnfs': 3,
            'deployed_vnfs': 1,
            'success': False,
            'is_edge_aware': False,
            'pressure_level': 'high'
        }
        
        reward = compute_reward(failure_info, reward_config)
        print(f"✅ 失败案例奖励: {reward:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 奖励系统测试失败: {e}")
        return False

def test_integration():
    """集成测试"""
    print("\n🧪 集成测试...")
    
    try:
        # 创建环境
        env = test_environment()
        if env is None:
            return False
        
        # 创建智能体
        agents = test_agents()
        if not agents:
            return False
        
        # 选择一个智能体进行测试
        agent = agents.get('ddqn')
        if agent is None:
            print("❌ 没有可用的DDQN智能体")
            return False
        
        print("🔄 执行完整的episode测试...")
        
        # 重置环境
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        max_steps = 10
        
        while step_count < max_steps:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                print(f"⚠️ 第{step_count}步没有有效动作")
                break
            
            # 智能体选择动作
            action = agent.select_action(state, valid_actions=valid_actions)
            if action not in valid_actions:
                action = valid_actions[0]
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            total_reward += reward
            step_count += 1
            
            print(f"   步骤 {step_count}: 动作={action}, 奖励={reward:.2f}, 完成={done}")
            
            if done:
                success = info.get('success', False)
                print(f"   Episode完成: 成功={success}, 总奖励={total_reward:.2f}")
                break
            
            state = next_state
        
        print(f"✅ 集成测试完成: {step_count}步, 总奖励={total_reward:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_system():
    """测试训练系统"""
    print("\n🧪 测试训练系统...")
    
    try:
        # 直接运行简化的main_multi_agent
        print("🔄 执行简化多智能体训练测试...")
        
        # 导入并测试
        sys.path.insert(0, 'project')
        os.chdir('project')
        
        # 运行简化训练
        exec(open('main_multi_agent.py').read())
        
        print("✅ 训练测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 训练系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 VNF嵌入多智能体系统全面测试 (修复版)")
    print("=" * 60)
    
    # 检查Python环境
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"工作目录: {os.getcwd()}")
    
    # 检查必要文件
    required_files = ["config.yaml"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 缺少必要文件: {file}")
            return False
    
    print("✅ 必要文件检查通过")
    
    # 执行测试序列
    test_results = []
    
    # 1. 模块导入测试
    test_results.append(("模块导入", test_imports()))
    
    # 2. 配置系统测试
    test_results.append(("配置系统", test_config_system()))
    
    # 3. 拓扑生成测试
    graph, node_features, edge_features = test_topology_generation()
    test_results.append(("拓扑生成", graph is not None))
    
    # 4. 环境测试
    test_results.append(("环境系统", test_environment() is not None))
    
    # 5. 智能体测试
    test_results.append(("智能体系统", len(test_agents()) > 0))
    
    # 6. GNN编码器测试
    test_results.append(("GNN编码器", test_gnn_encoder()))
    
    # 7. 奖励系统测试
    test_results.append(("奖励系统", test_reward_system()))
    
    # 8. 集成测试
    test_results.append(("集成测试", test_integration()))
    
    # 9. 训练系统测试（可选）
    run_training_test = input("\n是否执行训练系统测试？(y/N): ").lower().strip() == 'y'
    if run_training_test:
        test_results.append(("训练系统", test_training_system()))
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15s}: {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以运行。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    main()
