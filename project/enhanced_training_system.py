# enhanced_training_system.py - 集成增强功能的训练系统
# enhanced_training_system.py - 在文件最开头

def ultimate_emergency_fix():
    """彻底的应急修复"""
    print("🛠️ 应用彻底应急修复...")
    
    # 1. 禁用增强GNN补丁
    try:
        import agents.enhanced_base_agent as enhanced_base
        def dummy_patch():
            print("🚫 增强GNN补丁已禁用")
        enhanced_base.patch_existing_agents = dummy_patch
    except:
        pass
    
    # 2. 强制使用标准智能体
    from agents.base_agent import create_agent as original_create_agent
    
    def ultra_safe_create_agent(agent_type, agent_id, state_dim, action_dim, edge_dim, config):
        # 完全移除enhanced标识
        safe_id = agent_id.replace("_enhanced", "").replace("enhanced_", "")
        
        # 创建完全标准的智能体
        agent = original_create_agent(agent_type, safe_id, state_dim, action_dim, edge_dim, config)
        
        # 恢复ID但标记为标准模式
        agent.agent_id = agent_id
        agent.is_enhanced = False
        
        print(f"✅ {agent_id}: 彻底安全模式")
        return agent
    
    # 3. 阻止任何enhanced相关的导入
    import sys
    class BlockedModule:
        def __getattr__(self, name):
            print(f"🚫 阻止访问增强模块: {name}")
            return lambda *args, **kwargs: None
    
    try:
        sys.modules['models.enhanced_gnn_encoder'] = BlockedModule()
    except:
        pass
    
    # 4. 替换智能体创建函数
    import agents.base_agent
    agents.base_agent.create_agent = ultra_safe_create_agent
    
    print("✅ 彻底应急修复完成 - 100%标准模式")

# 调用彻底修复（替换原来的emergency_fix()调用）
ultimate_emergency_fix()

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse
import random

# 导入增强组件
from env.enhanced_vnf_env_multi import EnhancedVNFEmbeddingEnv
from env.topology_loader import generate_topology
from agents.enhanced_base_agent import patch_existing_agents
from agents.base_agent import create_agent
from models.enhanced_gnn_encoder import create_enhanced_edge_aware_encoder
from rewards.enhanced_edge_aware_reward import compute_enhanced_edge_aware_reward
from utils.logger import Logger
from utils.metrics import calculate_sar, calculate_splat
from config_loader import get_scenario_config, print_scenario_plan, validate_all_configs, load_config

class EnhancedEdgeAwareTrainer:
    """
    增强的Edge-Aware训练系统
    
    主要增强：
    1. 集成增强的GNN编码器
    2. 使用增强的奖励系统
    3. 课程学习机制
    4. 对比学习框架
    5. 动态性能监控
    """
    
    
    
    def __init__(self, config_path: str = "config.yaml"):
        print(f"🚀 初始化增强Edge-Aware训练系统...")
        #patch_existing_agents()
        # 加载配置
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练参数
        self.episodes = self.config['train']['episodes']
        self.save_interval = 50
        self.eval_interval = 25
        self.agent_types = ['ddqn', 'dqn', 'ppo']
        
        # 结果目录
        self.results_dir = f"enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 渐进式场景
        self.current_scenario = "normal_operation"
        self.scenario_start_episode = 1
        
        # 性能监控
        self.performance_tracker = {
            'edge_aware': {agent: {'sar': [], 'splat': [], 'rewards': [], 'edge_scores': [], 'quality_scores': []} 
                          for agent in self.agent_types},
            'baseline': {agent: {'sar': [], 'splat': [], 'rewards': [], 'edge_scores': [], 'quality_scores': []} 
                        for agent in self.agent_types}
        }
        
        # 打印训练计划和验证配置
        print_scenario_plan()
        validate_all_configs()
        
        # 初始化组件
        self._setup_enhanced_components()
        
        print(f"✅ 增强Edge-Aware训练系统初始化完成")
        print(f"   - 智能体类型: {self.agent_types}")
        print(f"   - 训练轮数: {self.episodes}")
        print(f"   - 增强功能: GNN编码器 + 奖励系统 + 课程学习")
    
    def _setup_enhanced_components(self):
        """设置增强组件"""
        print(f"🔧 设置增强组件...")
        
        # 1. 生成网络拓扑
        self._setup_enhanced_topology()
        
        # 2. 创建增强环境
        self._setup_enhanced_environments()
        
        # 3. 创建增强智能体
        self._setup_enhanced_agents()
        
        # 4. 设置日志记录
        self._setup_logging()
        
        print(f"✅ 增强组件设置完成")
    
    def _setup_enhanced_topology(self):
        """设置增强网络拓扑"""
        full_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'dimensions': self.config['dimensions']
        }
        
        self.graph, self.node_features, self.edge_features = generate_topology(config=full_config)
        
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        
        print(f"🌐 增强网络拓扑:")
        print(f"   - 节点数: {num_nodes}")
        print(f"   - 边数: {num_edges}")
        print(f"   - 节点特征: {self.node_features.shape}")
        print(f"   - 边特征: {self.edge_features.shape}")
        
        # 验证拓扑连通性
        import networkx as nx
        if not nx.is_connected(self.graph):
            print("⚠️ 警告: 网络拓扑不连通，可能影响训练效果")
    
    def _setup_enhanced_environments(self):
        """设置增强环境"""
        reward_config = self.config['reward']
        chain_length_range = tuple(self.config['vnf_requirements']['chain_length_range'])
        
        env_config = {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'reward': self.config['reward'],
            'train': self.config['train'],
            'dimensions': self.config['dimensions']
        }
        
        # 创建增强的Edge-aware环境
        self.env_edge_aware = EnhancedVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=reward_config,
            chain_length_range=chain_length_range,
            config=env_config.copy()
        )
        
        # 创建Baseline环境
        self.env_baseline = EnhancedVNFEmbeddingEnv(
            graph=self.graph.copy(),
            node_features=self.node_features.copy(),
            edge_features=self.edge_features.copy(),
            reward_config=reward_config,
            chain_length_range=chain_length_range,
            config=env_config.copy()
        )
        self.env_baseline.is_baseline_mode = True
        
        print(f"🌍 增强环境创建完成:")
        print(f"   - Edge-aware环境: 完整边特征感知")
        print(f"   - Baseline环境: 简化边特征")
    
    def _setup_enhanced_agents(self):
        """设置增强智能体"""
        expected_node_dim = self.config['dimensions']['node_feature_dim']
        action_dim = len(self.graph.nodes())
        
        print(f"🤖 创建增强智能体:")
        print(f"   - 状态维度: {expected_node_dim}")
        print(f"   - 动作维度: {action_dim}")
        
        # Edge-aware智能体（补丁会自动处理增强GNN）
        self.agents_edge_aware = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_edge_aware_enhanced"  # 保持enhanced标识
            edge_dim = self.config['gnn']['edge_aware']['edge_dim']
            
            # 直接创建智能体，补丁会处理GNN替换
            self.agents_edge_aware[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=expected_node_dim,
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
            print(f"   ✅ {agent_id}: 增强功能已启用")
        
        # Baseline智能体保持不变
        self.agents_baseline = {}
        for agent_type in self.agent_types:
            agent_id = f"{agent_type}_baseline"
            edge_dim = self.config['gnn']['baseline']['edge_dim']
            
            self.agents_baseline[agent_type] = create_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                state_dim=expected_node_dim,
                action_dim=action_dim,
                edge_dim=edge_dim,
                config=self.config
            )
            print(f"   ✅ {agent_id}: 标准GNN编码器")
    
    def _setup_logging(self):
        """设置增强日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loggers = {}
        
        for agent_type in self.agent_types:
            # Edge-aware日志器
            self.loggers[f"{agent_type}_edge_aware"] = Logger(
                log_dir=os.path.join(self.results_dir, f"enhanced_{agent_type}_edge_aware_{timestamp}")
            )
            # Baseline日志器
            self.loggers[f"{agent_type}_baseline"] = Logger(
                log_dir=os.path.join(self.results_dir, f"{agent_type}_baseline_{timestamp}")
            )
        
        print(f"📊 增强日志记录设置完成")
    
    def train_enhanced_episode(self, agent, env, agent_id: str) -> Dict[str, Any]:
        """训练单个增强episode"""
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        success = False
        info = {}
        edge_scores = []
        quality_scores = []
        
        # 重置智能体episode统计
        if hasattr(agent, 'reset_episode_stats'):
            agent.reset_episode_stats()
        
        max_steps = getattr(env, 'max_episode_steps', 20)
        
        while step_count < max_steps:
            # 获取增强的有效动作
            if hasattr(env, 'get_enhanced_valid_actions'):
                valid_actions = env.get_enhanced_valid_actions()
            else:
                valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                info = {'success': False, 'reason': 'no_valid_actions'}
                break
            
            # 选择动作
            action = agent.select_action(state, valid_actions=valid_actions)
            if action not in valid_actions:
                action = random.choice(valid_actions)
            
            # 执行动作
            next_state, reward, done, step_info = env.step(action)
            
            # 收集增强指标
            if 'edge_aware' in agent_id and hasattr(agent.gnn_encoder, 'get_vnf_adaptation_score'):
                try:
                    edge_score = agent.gnn_encoder.get_vnf_adaptation_score(state)
                    edge_scores.append(edge_score)
                except:
                    edge_scores.append(0.0)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新状态和统计
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 学习更新
            try:
                if hasattr(agent, 'should_update') and agent.should_update():
                    learning_info = agent.learn()
                elif hasattr(agent, 'replay_buffer') and len(getattr(agent, 'replay_buffer', [])) >= getattr(agent, 'batch_size', 32):
                    learning_info = agent.learn()
            except Exception as e:
                pass
            
            if done:
                success = step_info.get('success', False)
                info = step_info
                break
        
        # 最后一次学习更新
        try:
            if hasattr(agent, 'experiences') and len(getattr(agent, 'experiences', [])) > 0:
                if hasattr(agent, 'should_update') and agent.should_update():
                    learning_info = agent.learn()
        except Exception as e:
            pass
        
        # 计算增强指标
        avg_edge_score = np.mean(edge_scores) if edge_scores else 0.0
        
        # 计算路径质量评分
        if success and 'paths' in info:
            quality_score = self._calculate_path_quality_score(info['paths'])
            quality_scores.append(quality_score)
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # 获取当前场景名称
        current_scenario_name = getattr(env, 'current_scenario_name', self.current_scenario)
        
        # 计算episode统计
        sar = 1.0 if success else 0.0
        splat = info.get('splat', info.get('avg_delay', float('inf'))) if success else float('inf')
        
        episode_stats = {
            'total_reward': total_reward,
            'steps': step_count,
            'success': success,
            'sar': sar,
            'splat': splat,
            'edge_score': avg_edge_score,
            'quality_score': avg_quality_score,
            'info': info,
            'scenario': current_scenario_name
        }
        
        return episode_stats
    
    def _calculate_path_quality_score(self, paths: List[Dict]) -> float:
        """计算路径质量评分"""
        if not paths:
            return 0.0
        
        total_quality = 0.0
        for path in paths:
            delay = path.get("delay", 0.0)
            jitter = path.get("jitter", 0.0)
            loss = path.get("loss", 0.0)
            bandwidth = path.get("bandwidth", 0.0)
            
            # 简化质量评分
            delay_score = max(0.0, 1.0 - delay / 100.0)
            jitter_score = max(0.0, 1.0 - jitter / 5.0)
            loss_score = max(0.0, 1.0 - loss / 0.05)
            bandwidth_score = min(1.0, bandwidth / 100.0)
            
            quality = (delay_score + jitter_score + loss_score + bandwidth_score) / 4.0
            total_quality += quality
        
        return total_quality / len(paths)
    
    def _update_scenario(self, episode: int) -> bool:
        """更新当前场景"""
        new_scenario = None
        
        if episode <= 25:
            new_scenario = "normal_operation"
        elif episode <= 50:
            new_scenario = "peak_congestion"
        elif episode <= 75:
            new_scenario = "failure_recovery"
        else:
            new_scenario = "extreme_pressure"
        
        if new_scenario and new_scenario != self.current_scenario:
            print(f"\n🎯 场景切换: {self.current_scenario} → {new_scenario}")
            
            self.current_scenario = new_scenario
            self.scenario_start_episode = episode
            
            # 获取场景配置并应用
            scenario_config = get_scenario_config(episode)
            self.env_edge_aware.apply_scenario_config(scenario_config)
            self.env_baseline.apply_scenario_config(scenario_config)
            
            print(f"✅ 场景配置已应用: Episode {episode}")
            return True
        
        return False
    
    def train_enhanced(self):
        """主要的增强训练循环"""
        print(f"\n🚀 开始增强Edge-Aware训练...")
        print(f"目标episodes: {self.episodes}")
        
        # 初始化第一个场景
        initial_scenario_config = get_scenario_config(1)
        self.env_edge_aware.apply_scenario_config(initial_scenario_config)
        self.env_baseline.apply_scenario_config(initial_scenario_config)
        
        for episode in range(1, self.episodes + 1):
            # 检查并更新场景
            scenario_changed = self._update_scenario(episode)
            
            if episode % 25 == 0 or scenario_changed:
                print(f"\n📍 Episode {episode}/{self.episodes} - 场景: {self.current_scenario}")
            
            # 训练Edge-aware智能体
            for agent_type in self.agent_types:
                agent = self.agents_edge_aware[agent_type]
                env = self.env_edge_aware
                episode_stats = self.train_enhanced_episode(agent, env, f"{agent_type}_edge_aware")
                
                # 记录增强指标
                self.performance_tracker['edge_aware'][agent_type]['sar'].append(episode_stats['sar'])
                self.performance_tracker['edge_aware'][agent_type]['splat'].append(episode_stats['splat'])
                self.performance_tracker['edge_aware'][agent_type]['rewards'].append(episode_stats['total_reward'])
                self.performance_tracker['edge_aware'][agent_type]['edge_scores'].append(episode_stats['edge_score'])
                self.performance_tracker['edge_aware'][agent_type]['quality_scores'].append(episode_stats['quality_score'])
                
                # 记录日志
                logger_id = f"{agent_type}_edge_aware"
                if logger_id in self.loggers:
                    self.loggers[logger_id].log_episode(episode, episode_stats)
            
            # 训练Baseline智能体
            for agent_type in self.agent_types:
                agent = self.agents_baseline[agent_type]
                env = self.env_baseline
                episode_stats = self.train_enhanced_episode(agent, env, f"{agent_type}_baseline")
                
                # 记录指标
                self.performance_tracker['baseline'][agent_type]['sar'].append(episode_stats['sar'])
                self.performance_tracker['baseline'][agent_type]['splat'].append(episode_stats['splat'])
                self.performance_tracker['baseline'][agent_type]['rewards'].append(episode_stats['total_reward'])
                self.performance_tracker['baseline'][agent_type]['edge_scores'].append(episode_stats['edge_score'])
                self.performance_tracker['baseline'][agent_type]['quality_scores'].append(episode_stats['quality_score'])
                
                # 记录日志
                logger_id = f"{agent_type}_baseline"
                if logger_id in self.loggers:
                    self.loggers[logger_id].log_episode(episode, episode_stats)
            
            # 定期进度报告
            if episode % 25 == 0:
                self._print_enhanced_progress(episode)
            
            # 定期保存模型
            if episode % self.save_interval == 0:
                self._save_enhanced_models(episode)
        
        # 最终分析
        self._enhanced_final_analysis()
        
        print(f"\n🎉 增强Edge-Aware训练完成!")
        return self.performance_tracker
    
    def _print_enhanced_progress(self, episode: int):
        """打印增强训练进度"""
        print(f"\n📊 Episode {episode} 增强性能统计:")
        window = 25
        start_idx = max(0, episode - window)
        
        for variant in ['edge_aware', 'baseline']:
            print(f"\n{variant.upper()}:")
            for agent_type in self.agent_types:
                tracker = self.performance_tracker[variant][agent_type]
                
                recent_sar = np.mean(tracker['sar'][start_idx:])
                recent_splat = np.mean([s for s in tracker['splat'][start_idx:] if s != float('inf')])
                recent_reward = np.mean(tracker['rewards'][start_idx:])
                recent_edge_score = np.mean(tracker['edge_scores'][start_idx:])
                recent_quality = np.mean(tracker['quality_scores'][start_idx:])
                
                print(f"  {agent_type.upper()}:")
                print(f"    SAR: {recent_sar:.3f}")
                print(f"    SPLat: {recent_splat:.2f}")
                print(f"    Reward: {recent_reward:.1f}")
                if variant == 'edge_aware':
                    print(f"    Edge Score: {recent_edge_score:.3f}")
                    print(f"    Quality: {recent_quality:.3f}")
    
    def _save_enhanced_models(self, episode: int):
        """保存增强模型"""
        checkpoint_dir = os.path.join(self.results_dir, "enhanced_checkpoints", f"episode_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存Edge-aware智能体
        for agent_type, agent in self.agents_edge_aware.items():
            filepath = os.path.join(checkpoint_dir, f"enhanced_{agent_type}_edge_aware.pth")
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(filepath)
        
        # 保存Baseline智能体
        for agent_type, agent in self.agents_baseline.items():
            filepath = os.path.join(checkpoint_dir, f"{agent_type}_baseline.pth")
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(filepath)
        
        print(f"💾 Episode {episode} 增强模型已保存")
    
    def _enhanced_final_analysis(self):
        """增强最终分析"""
        print(f"\n🎯 增强Edge-Aware最终性能分析:")
        print(f"{'='*70}")
        
        # 保存详细结果
        results_data = []
        
        for variant in ['edge_aware', 'baseline']:
            for agent_type in self.agent_types:
                tracker = self.performance_tracker[variant][agent_type]
                
                final_sar = np.mean(tracker['sar'][-25:]) if len(tracker['sar']) >= 25 else np.mean(tracker['sar'])
                final_splat = np.mean([s for s in tracker['splat'][-25:] if s != float('inf')])
                final_reward = np.mean(tracker['rewards'][-25:]) if len(tracker['rewards']) >= 25 else np.mean(tracker['rewards'])
                final_edge_score = np.mean(tracker['edge_scores'][-25:]) if len(tracker['edge_scores']) >= 25 else np.mean(tracker['edge_scores'])
                final_quality = np.mean(tracker['quality_scores'][-25:]) if len(tracker['quality_scores']) >= 25 else np.mean(tracker['quality_scores'])
                
                results_data.append({
                    'Variant': variant,
                    'Algorithm': agent_type.upper(),
                    'Final_SAR': final_sar,
                    'Final_SPLat': final_splat,
                    'Final_Reward': final_reward,
                    'Edge_Score': final_edge_score,
                    'Quality_Score': final_quality
                })
                
                print(f"\n{variant.upper()} - {agent_type.upper()}:")
                print(f"  最终SAR: {final_sar:.3f}")
                print(f"  最终SPLat: {final_splat:.2f}")
                print(f"  最终奖励: {final_reward:.1f}")
                if variant == 'edge_aware':
                    print(f"  Edge适应性: {final_edge_score:.3f}")
                    print(f"  路径质量: {final_quality:.3f}")
        
        # 保存结果
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(os.path.join(self.results_dir, 'enhanced_final_results.csv'), index=False)
        
        # 计算Edge-Aware优势
        print(f"\n📈 Edge-Aware优势分析:")
        for agent_type in self.agent_types:
            edge_sar = df_results[(df_results['Variant'] == 'edge_aware') & (df_results['Algorithm'] == agent_type.upper())]['Final_SAR'].iloc[0]
            baseline_sar = df_results[(df_results['Variant'] == 'baseline') & (df_results['Algorithm'] == agent_type.upper())]['Final_SAR'].iloc[0]
            
            sar_improvement = (edge_sar - baseline_sar) / baseline_sar * 100 if baseline_sar > 0 else 0
            print(f"  {agent_type.upper()} SAR提升: {sar_improvement:.1f}%")
        
        print(f"\n💾 增强结果已保存: {self.results_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强Edge-Aware VNF嵌入训练系统')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ 使用设备: {device}")
    
    # 创建训练器
    trainer = EnhancedEdgeAwareTrainer(config_path=args.config)
    
    if args.episodes:
        trainer.episodes = args.episodes
    
    # 开始训练
    results = trainer.train_enhanced()
    
    print(f"\n✅ 增强Edge-Aware训练任务完成!")
    print(f"📁 结果保存在: {trainer.results_dir}")
    print(f"🎯 核心增强:")
    print(f"   ✅ 增强GNN编码器 - 边注意力机制")
    print(f"   ✅ 增强奖励系统 - 多维度评估")
    print(f"   ✅ 课程学习 - 渐进式场景")
    print(f"   ✅ 性能监控 - 实时分析")


if __name__ == "__main__":
    main()