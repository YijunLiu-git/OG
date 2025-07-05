# main_multi_agent.py - 修复版

import os
import yaml
import torch
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
import argparse
import random

from env.vnf_env_multi import MultiVNFEmbeddingEnv
from env.topology_loader import generate_topology
from agents.base_agent import create_agent
from utils.logger import Logger
from utils.metrics import calculate_sar, calculate_splat
from config_loader import get_scenario_config, print_scenario_plan, validate_all_configs, load_config

class MultiAgentTrainer:
    """
    多智能体VNF嵌入训练器 - 修复版本
    
    主要修复：
    1. ✅ 确保维度一致性
    2. ✅ 简化场景配置逻辑
    3. ✅ 改进错误处理
    4. ✅ 添加更详细的日志记录
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        print("🚀 初始化多智能体训练器...")
        
        # 加载配置
        self.config = load_config(config_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")
        
        # 训练参数
        self.episodes = self.config['train']['episodes']
        self.save_interval = 50
        self.eval_interval = 25
        self.agent_types = ['ddqn', 'dqn', 'ppo']
        
        # 创建结果目录
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 场景相关
        self.current_scenario = "normal_operation"
        self.scenario_start_episode = 1
        self.last_applied_scenario = None
        
        # 初始化组件
        self._setup_network_topology()
        self._setup_environments()
        self._setup_agents()
        self._setup_logging()
        
        print(f"✅ 多智能体训练器初始化完成")
        print(f"   - 智能体类型: {self.agent_types}")
        print(f"   - 训练轮数: {self.episodes}")
        print(f"   - 网络节点: {len(self.graph.nodes())}")
    
    def _setup_network_topology(self):
        """设置网络拓扑"""
        print("🌐 设置网络拓扑...")
        
        try:
            # 使用完整的配置字典生成拓扑
            full_config = {
                'topology': self.config['topology'],
                'vnf_requirements': self.config['vnf_requirements'],
                'dimensions': self.config['dimensions']
            }
            
            self.graph, self.node_features, self.edge_features = generate_topology(config=full_config)
            
            # ✅ 验证拓扑生成结果
            num_nodes = len(self.graph.nodes())
            num_edges = len(self.graph.edges())
            
            # 确保连通性
            if not nx.is_connected(self.graph):
                print("⚠️ 图不连通，尝试添加边...")
                # 简单处理：连接所有孤立的组件
                components = list(nx.connected_components(self.graph))
                for i in range(len(components) - 1):
                    node1 = list(components[i])[0]
                    node2 = list(components[i + 1])[0]
                    self.graph.add_edge(node1, node2)
                    # 为新边添加特征
                    new_edge_features = np.array([[50.0, 5.0, 0.1, 0.01]])  # 默认边特征
                    self.edge_features = np.vstack([self.edge_features, new_edge_features])
            
            # 验证特征维度
            expected_node_dim = 4  # 基础特征维度
            expected_edge_dim = 4  # 边特征维度
            
            assert self.node_features.shape[1] == expected_node_dim, \
                f"节点特征应为{expected_node_dim}维，实际{self.node_features.shape[1]}维"
            assert self.edge_features.shape[1] == expected_edge_dim, \
                f"边特征应为{expected_edge_dim}维，实际{self.edge_features.shape[1]}维"
            
            print(f"✅ 网络拓扑生成完成:")
            print(f"   - 节点数: {num_nodes}")
            print(f"   - 边数: {len(self.graph.edges())}")
            print(f"   - 连通性: {nx.is_connected(self.graph)}")
            print(f"   - 节点特征: {self.node_features.shape}")
            print(f"   - 边特征: {self.edge_features.shape}")
            
        except Exception as e:
            print(f"❌ 网络拓扑设置失败: {e}")
            raise
    
    def _setup_environments(self):
        """设置训练和测试环境"""
        print("🌍 设置训练环境...")
        
        try:
            reward_config = self.config['reward']
            chain_length_range = tuple(self.config['vnf_requirements']['chain_length_range'])
            
            # 环境配置
            env_config = {
                'topology': self.config['topology'],
                'vnf_requirements': self.config['vnf_requirements'],
                'reward': self.config['reward'],
                'train': self.config['train'],
                'dimensions': self.config['dimensions']
            }
            
            # Edge-aware环境（使用完整的4维边特征）
            self.env_edge_aware = MultiVNFEmbeddingEnv(
                graph=self.graph.copy(),
                node_features=self.node_features.copy(),
                edge_features=self.edge_features.copy(),
                reward_config=reward_config,
                chain_length_range=chain_length_range,
                config=env_config.copy()
            )
            
            # Baseline环境
            self.env_baseline = MultiVNFEmbeddingEnv(
                graph=self.graph.copy(),
                node_features=self.node_features.copy(),
                edge_features=self.edge_features.copy(),
                reward_config=reward_config,
                chain_length_range=chain_length_range,
                config=env_config.copy()
            )
            # 标记Baseline环境
            self.env_baseline.is_baseline_mode = True
            
            print(f"✅ 环境设置完成:")
            print(f"   - Edge-aware环境: 4维边特征")
            print(f"   - Baseline环境: 4维环境特征，智能体感知2维")
            
        except Exception as e:
            print(f"❌ 环境设置失败: {e}")
            raise
    
    def _setup_agents(self):
        """设置智能体"""
        print("🤖 设置智能体...")
        
        try:
            # ✅ 确保维度配置正确
            state_dim = 8  # 环境输出的固定维度
            action_dim = len(self.graph.nodes())
            
            print(f"📊 智能体参数:")
            print(f"   - 状态维度: {state_dim} (固定)")
            print(f"   - 动作维度: {action_dim}")
            
            # Edge-aware智能体
            self.agents_edge_aware = {}
            for agent_type in self.agent_types:
                agent_id = f"{agent_type}_edge_aware"
                edge_dim = self.config['gnn']['edge_aware']['edge_dim']  # 4维
                
                self.agents_edge_aware[agent_type] = create_agent(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    edge_dim=edge_dim,
                    config=self.config
                )
                print(f"✅ {agent_id} 创建成功 (边特征: {edge_dim}维)")
            
            # Baseline智能体
            self.agents_baseline = {}
            for agent_type in self.agent_types:
                agent_id = f"{agent_type}_baseline"
                edge_dim = self.config['gnn']['baseline']['edge_dim']  # 2维
                
                self.agents_baseline[agent_type] = create_agent(
                    agent_type=agent_type,
                    agent_id=agent_id,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    edge_dim=edge_dim,
                    config=self.config
                )
                print(f"✅ {agent_id} 创建成功 (边特征: {edge_dim}维)")
            
            print(f"✅ 智能体设置完成")
            
        except Exception as e:
            print(f"❌ 智能体设置失败: {e}")
            raise
    
    def _setup_logging(self):
        """设置日志记录"""
        print("📊 设置日志记录...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.loggers = {}
            
            for agent_type in self.agent_types:
                # Edge-aware日志器
                logger_id = f"{agent_type}_edge_aware"
                self.loggers[logger_id] = Logger(
                    log_dir=os.path.join(self.results_dir, f"{logger_id}_{timestamp}")
                )
                # Baseline日志器
                logger_id = f"{agent_type}_baseline"
                self.loggers[logger_id] = Logger(
                    log_dir=os.path.join(self.results_dir, f"{logger_id}_{timestamp}")
                )
            
            print(f"✅ 日志记录设置完成")
            
        except Exception as e:
            print(f"❌ 日志设置失败: {e}")
            raise

    def _update_scenario(self, episode: int):
        """更新当前场景"""
        try:
            # 确定当前应该是哪个场景
            new_scenario = None
            if episode <= 25:
                new_scenario = "normal_operation"
            elif episode <= 50:
                new_scenario = "peak_congestion"
            elif episode <= 75:
                new_scenario = "failure_recovery"
            else:
                new_scenario = "extreme_pressure"
            
            # 只在场景真正改变时才应用配置
            if new_scenario and new_scenario != self.current_scenario:
                print(f"\n🎯 场景切换: {self.current_scenario} → {new_scenario} (Episode {episode})")
                
                self.current_scenario = new_scenario
                self.scenario_start_episode = episode
                
                # 获取场景配置
                scenario_config = get_scenario_config(episode)
                
                # 应用场景配置到环境
                print(f"🔧 应用场景配置...")
                self.env_edge_aware.apply_scenario_config(scenario_config)
                self.env_baseline.apply_scenario_config(scenario_config)
                
                self.last_applied_scenario = new_scenario
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ 场景更新失败: {e}")
            return False
    
    def train_single_episode(self, agent, env, agent_id: str) -> Dict[str, Any]:
        """训练单个episode"""
        try:
            state = env.reset()
            total_reward = 0.0
            step_count = 0
            success = False
            info = {}
            
            # 重置智能体episode统计
            if hasattr(agent, 'reset_episode_stats'):
                agent.reset_episode_stats()
            
            max_steps = getattr(env, 'max_episode_steps', 20)
            
            while step_count < max_steps:
                # 获取有效动作
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    info = {'success': False, 'reason': 'no_valid_actions'}
                    break
                
                # 选择动作
                try:
                    action = agent.select_action(state, valid_actions=valid_actions)
                    # ✅ 确保动作是int类型
                    if isinstance(action, (list, np.ndarray)):
                        action = int(action[0]) if len(action) > 0 else valid_actions[0]
                    else:
                        action = int(action)
                    
                    # 验证动作有效性
                    if action not in valid_actions:
                        action = random.choice(valid_actions)
                        
                except Exception as e:
                    print(f"⚠️ 动作选择失败: {e}")
                    action = random.choice(valid_actions)
                
                # 执行动作
                try:
                    next_state, reward, done, info = env.step(action)
                    
                    # 存储经验
                    agent.store_transition(state, action, reward, next_state, done)
                    
                    # 更新状态和统计
                    state = next_state
                    total_reward += reward
                    step_count += 1
                    
                    # 学习更新
                    if hasattr(agent, 'should_update') and agent.should_update():
                        learning_info = agent.learn()
                    elif (hasattr(agent, 'replay_buffer') and 
                          len(getattr(agent, 'replay_buffer', [])) >= getattr(agent, 'batch_size', 32)):
                        learning_info = agent.learn()
                    
                    if done:
                        success = info.get('success', False)
                        break
                        
                except Exception as e:
                    print(f"⚠️ Step执行失败: {e}")
                    break
            
            # 最后一次学习更新
            try:
                if hasattr(agent, 'experiences') and len(getattr(agent, 'experiences', [])) > 0:
                    if hasattr(agent, 'should_update') and agent.should_update():
                        learning_info = agent.learn()
            except Exception as e:
                pass  # 忽略学习错误
            
            # 计算episode统计
            current_scenario_name = getattr(env, 'current_scenario_name', self.current_scenario)
            
            sar = 1.0 if success else 0.0
            splat = info.get('splat', info.get('avg_delay', float('inf'))) if success else float('inf')
            jitter = info.get('avg_jitter', 0.0) if success else 0.0
            loss = info.get('avg_loss', 0.0) if success else 0.0
            
            episode_stats = {
                'total_reward': total_reward,
                'steps': step_count,
                'success': success,
                'sar': sar,
                'splat': splat,
                'jitter': jitter,
                'loss': loss,
                'info': info,
                'scenario': current_scenario_name
            }
            
            return episode_stats
            
        except Exception as e:
            print(f"❌ Episode训练失败: {e}")
            return {
                'total_reward': -10.0,
                'steps': 0,
                'success': False,
                'sar': 0.0,
                'splat': float('inf'),
                'jitter': 0.0,
                'loss': 0.0,
                'info': {'error': str(e)},
                'scenario': self.current_scenario
            }
    
    def train(self):
        """主训练循环"""
        print(f"\n🚀 开始多智能体渐进式场景训练...")
        print(f"目标episodes: {self.episodes}")
        
        # 显示训练计划
        print_scenario_plan()
        
        all_results = {
            'edge_aware': {agent_type: {
                'rewards': [], 'sar': [], 'splat': [], 'success': [], 
                'jitter': [], 'loss': [], 'scenarios': []
            } for agent_type in self.agent_types},
            'baseline': {agent_type: {
                'rewards': [], 'sar': [], 'splat': [], 'success': [], 
                'jitter': [], 'loss': [], 'scenarios': []
            } for agent_type in self.agent_types}
        }
        
        # 初始化第一个场景
        print(f"🔧 初始化第一个场景...")
        try:
            initial_scenario_config = get_scenario_config(1)
            self.env_edge_aware.apply_scenario_config(initial_scenario_config)
            self.env_baseline.apply_scenario_config(initial_scenario_config)
            print(f"✅ 初始场景设置完成")
        except Exception as e:
            print(f"⚠️ 初始场景设置失败: {e}")
        
        # 主训练循环
        for episode in range(1, self.episodes + 1):
            try:
                # 检查并更新场景
                scenario_changed = self._update_scenario(episode)
                
                if episode % 25 == 0 or scenario_changed:
                    print(f"\n📍 Episode {episode}/{self.episodes} - 场景: {self.current_scenario}")
                
                # 训练Edge-aware智能体
                for agent_type in self.agent_types:
                    try:
                        agent = self.agents_edge_aware[agent_type]
                        env = self.env_edge_aware
                        episode_stats = self.train_single_episode(agent, env, f"{agent_type}_edge_aware")
                        
                        # 记录结果
                        all_results['edge_aware'][agent_type]['rewards'].append(episode_stats['total_reward'])
                        all_results['edge_aware'][agent_type]['sar'].append(episode_stats['sar'])
                        all_results['edge_aware'][agent_type]['splat'].append(episode_stats['splat'])
                        all_results['edge_aware'][agent_type]['success'].append(episode_stats['success'])
                        all_results['edge_aware'][agent_type]['jitter'].append(episode_stats['jitter'])
                        all_results['edge_aware'][agent_type]['loss'].append(episode_stats['loss'])
                        all_results['edge_aware'][agent_type]['scenarios'].append(episode_stats['scenario'])
                        
                        # 记录日志
                        logger_id = f"{agent_type}_edge_aware"
                        if logger_id in self.loggers:
                            self.loggers[logger_id].log_episode(episode, episode_stats)
                    
                    except Exception as e:
                        print(f"⚠️ Edge-aware {agent_type} 训练失败: {e}")
                
                # 训练Baseline智能体
                for agent_type in self.agent_types:
                    try:
                        agent = self.agents_baseline[agent_type]
                        env = self.env_baseline
                        episode_stats = self.train_single_episode(agent, env, f"{agent_type}_baseline")
                        
                        # 记录结果
                        all_results['baseline'][agent_type]['rewards'].append(episode_stats['total_reward'])
                        all_results['baseline'][agent_type]['sar'].append(episode_stats['sar'])
                        all_results['baseline'][agent_type]['splat'].append(episode_stats['splat'])
                        all_results['baseline'][agent_type]['success'].append(episode_stats['success'])
                        all_results['baseline'][agent_type]['jitter'].append(episode_stats['jitter'])
                        all_results['baseline'][agent_type]['loss'].append(episode_stats['loss'])
                        all_results['baseline'][agent_type]['scenarios'].append(episode_stats['scenario'])
                        
                        # 记录日志
                        logger_id = f"{agent_type}_baseline"
                        if logger_id in self.loggers:
                            self.loggers[logger_id].log_episode(episode, episode_stats)
                    
                    except Exception as e:
                        print(f"⚠️ Baseline {agent_type} 训练失败: {e}")
                
                # 定期打印进度
                if episode % 25 == 0:
                    self._print_progress(episode, all_results)
                
                # 定期保存模型
                if episode % self.save_interval == 0:
                    self._save_models(episode)
                
                # 内存清理
                if episode % 10 == 0:
                    self._cleanup_memory()
                    
            except Exception as e:
                print(f"❌ Episode {episode} 训练失败: {e}")
                continue
        
        # 最终分析
        try:
            self._final_analysis(all_results)
        except Exception as e:
            print(f"⚠️ 最终分析失败: {e}")
        
        print(f"\n🎉 渐进式场景训练完成!")
        return all_results
    
    def _print_progress(self, episode: int, results: Dict):
        """打印训练进度"""
        try:
            print(f"\n📊 Episode {episode} 性能统计 (场景: {self.current_scenario}):")
            window = 25
            start_idx = max(0, episode - window)
            
            for variant in ['edge_aware', 'baseline']:
                print(f"\n{variant.upper()}:")
                for agent_type in self.agent_types:
                    if agent_type in results[variant]:
                        recent_sar = np.mean(results[variant][agent_type]['sar'][start_idx:])
                        recent_splat_values = [s for s in results[variant][agent_type]['splat'][start_idx:] 
                                             if s != float('inf')]
                        recent_splat = np.mean(recent_splat_values) if recent_splat_values else float('inf')
                        recent_reward = np.mean(results[variant][agent_type]['rewards'][start_idx:])
                        
                        print(f"  {agent_type.upper()}:")
                        print(f"    SAR: {recent_sar:.3f}")
                        print(f"    SPLat: {recent_splat:.2f}")
                        print(f"    Reward: {recent_reward:.1f}")
                        
        except Exception as e:
            print(f"⚠️ 进度打印失败: {e}")
    
    def _save_models(self, episode: int):
        """保存模型检查点"""
        try:
            checkpoint_dir = os.path.join(self.results_dir, "checkpoints", f"episode_{episode}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存Edge-aware智能体
            for agent_type, agent in self.agents_edge_aware.items():
                filepath = os.path.join(checkpoint_dir, f"{agent_type}_edge_aware.pth")
                if hasattr(agent, 'save_checkpoint'):
                    agent.save_checkpoint(filepath)
            
            # 保存Baseline智能体
            for agent_type, agent in self.agents_baseline.items():
                filepath = os.path.join(checkpoint_dir, f"{agent_type}_baseline.pth")
                if hasattr(agent, 'save_checkpoint'):
                    agent.save_checkpoint(filepath)
                    
            print(f"💾 模型检查点已保存: episode_{episode}")
            
        except Exception as e:
            print(f"⚠️ 模型保存失败: {e}")
    
    def _cleanup_memory(self):
        """清理内存"""
        try:
            # 清理智能体内存
            for agents in [self.agents_edge_aware, self.agents_baseline]:
                for agent in agents.values():
                    if hasattr(agent, 'cleanup_memory'):
                        agent.cleanup_memory()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ 内存清理失败: {e}")
    
    def _final_analysis(self, results: Dict):
        """最终性能分析"""
        print(f"\n🎯 最终性能分析:")
        print(f"{'='*70}")
        
        try:
            # 按场景分组分析
            scenario_data = []
            
            for scenario_name in ['normal_operation', 'peak_congestion', 'failure_recovery', 'extreme_pressure']:
                print(f"\n📋 {scenario_name}:")
                
                for variant in ['edge_aware', 'baseline']:
                    print(f"\n  {variant.upper()}:")
                    for agent_type in self.agent_types:
                        if agent_type in results[variant]:
                            # 获取该场景的数据
                            scenario_episodes = []
                            for i, ep_scenario in enumerate(results[variant][agent_type]['scenarios']):
                                if ep_scenario == scenario_name:
                                    scenario_episodes.append(i)
                            
                            if scenario_episodes:
                                # 计算平均性能
                                recent_episodes = scenario_episodes[-10:] if len(scenario_episodes) >= 10 else scenario_episodes
                                
                                scenario_sar = np.mean([results[variant][agent_type]['sar'][i] for i in recent_episodes])
                                scenario_splat_values = [results[variant][agent_type]['splat'][i] for i in recent_episodes 
                                                       if results[variant][agent_type]['splat'][i] != float('inf')]
                                scenario_splat = np.mean(scenario_splat_values) if scenario_splat_values else float('inf')
                                scenario_reward = np.mean([results[variant][agent_type]['rewards'][i] for i in recent_episodes])
                                
                                print(f"    {agent_type.upper()}:")
                                print(f"      SAR: {scenario_sar:.3f}")
                                print(f"      SPLat: {scenario_splat:.2f}")
                                print(f"      Reward: {scenario_reward:.1f}")
                                
                                scenario_data.append({
                                    'Scenario': scenario_name,
                                    'Variant': variant,
                                    'Algorithm': agent_type.upper(),
                                    'SAR': scenario_sar,
                                    'SPLat': scenario_splat,
                                    'Reward': scenario_reward
                                })
            
            # 保存结果
            df_scenarios = pd.DataFrame(scenario_data)
            results_csv_path = os.path.join(self.results_dir, 'scenario_results.csv')
            df_scenarios.to_csv(results_csv_path, index=False)
            
            print(f"\n💾 结果已保存: {results_csv_path}")
            
        except Exception as e:
            print(f"⚠️ 最终分析失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VNF嵌入多智能体训练 (修复版)')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    args = parser.parse_args()
    
    try:
        # 验证配置文件
        if not os.path.exists(args.config):
            print(f"❌ 配置文件不存在: {args.config}")
            return
        
        # 运行测试模式
        if args.test:
            print("🧪 运行测试模式...")
            # 这里可以添加测试代码
            return
        
        # 创建训练器
        trainer = MultiAgentTrainer(config_path=args.config)
        
        # 设置训练轮数
        if args.episodes:
            trainer.episodes = args.episodes
            trainer.config['train']['episodes'] = args.episodes
        
        # 开始训练
        results = trainer.train()
        
        print(f"\n✅ 训练任务完成!")
        print(f"📁 结果保存在: {trainer.results_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()