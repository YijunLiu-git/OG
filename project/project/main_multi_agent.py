#!/usr/bin/env python3
# main_multi_agent.py - macOS兼容版多智能体训练脚本

import os
import sys
import torch
import numpy as np
import random
from datetime import datetime
import traceback
import json

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def set_seeds(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class MacOSCompatibleTrainer:
    """macOS兼容的多智能体训练器"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        print("🚀 初始化macOS兼容多智能体训练器...")
        
        try:
            # 导入必要模块
            from config_loader import get_scenario_config, load_config
            from env.topology_loader import generate_topology
            from env.vnf_env_multi import EnhancedVNFEmbeddingEnv
            from agents.base_agent import create_agent
            
            # 加载配置
            self.config = load_config(config_path)
            print("✅ 配置加载成功")
            
            # 设置基本参数
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.episodes = 15  # 简化为15个episode
            self.agent_types = ['ddqn']  # 只使用DDQN
            
            # 生成拓扑
            self.graph, self.node_features, self.edge_features = generate_topology(self.config)
            print(f"✅ 拓扑生成: {len(self.graph.nodes())}节点, {len(self.graph.edges())}边")
            
            # 创建环境
            scenario_config = get_scenario_config('normal_operation')
            self.env = EnhancedVNFEmbeddingEnv(
                graph=self.graph,
                node_features=self.node_features,
                edge_features=self.edge_features,
                reward_config=scenario_config['reward'],
                config=self.config
            )
            self.env.apply_scenario_config(scenario_config)
            print("✅ 环境创建成功")
            
            # 创建智能体
            self.agent = create_agent(
                agent_type='ddqn',
                agent_id='ddqn_macos',
                state_dim=8,
                action_dim=len(self.graph.nodes()),
                edge_dim=4,
                config=self.config
            )
            print("✅ 智能体创建成功")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            traceback.print_exc()
            raise
    
    def train_episode(self, episode: int):
        """训练单个episode"""
        try:
            # 重置环境
            state = self.env.reset()
            total_reward = 0.0
            step_count = 0
            max_steps = 12
            
            while step_count < max_steps:
                # 获取有效动作
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break
                
                # 选择动作
                try:
                    action = self.agent.select_action(state, valid_actions=valid_actions)
                    if action not in valid_actions:
                        action = random.choice(valid_actions)
                except Exception as e:
                    action = random.choice(valid_actions)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                try:
                    self.agent.store_transition(state, action, reward, next_state, done)
                except Exception as e:
                    pass  # 忽略存储错误
                
                total_reward += reward
                step_count += 1
                state = next_state
                
                if done:
                    success = info.get('success', False)
                    sar = info.get('sar', 0.0)
                    splat = info.get('splat', float('inf'))
                    
                    result = {
                        'episode': episode,
                        'reward': total_reward,
                        'steps': step_count,
                        'success': success,
                        'sar': sar,
                        'splat': splat if splat != float('inf') else 0.0
                    }
                    
                    # 学习更新
                    try:
                        if hasattr(self.agent, 'learn'):
                            if hasattr(self.agent, 'replay_buffer') and len(getattr(self.agent, 'replay_buffer', [])) >= 8:
                                learning_info = self.agent.learn()
                    except Exception as e:
                        pass  # 忽略学习错误
                    
                    return result
            
            # Episode未完成的情况
            return {
                'episode': episode,
                'reward': total_reward,
                'steps': step_count,
                'success': False,
                'sar': 0.0,
                'splat': 100.0
            }
            
        except Exception as e:
            print(f"❌ Episode {episode} 训练失败: {e}")
            return {
                'episode': episode,
                'reward': -50.0,
                'steps': 0,
                'success': False,
                'sar': 0.0,
                'splat': 100.0
            }
    
    def train(self):
        """主训练循环"""
        print(f"\n🎯 开始macOS兼容多智能体训练")
        print(f"目标episodes: {self.episodes}")
        print("=" * 50)
        
        results = []
        
        for episode in range(1, self.episodes + 1):
            result = self.train_episode(episode)
            results.append(result)
            
            # 打印进度
            print(f"Episode {episode:2d}: "
                  f"奖励={result['reward']:6.1f}, "
                  f"步数={result['steps']}, "
                  f"成功={result['success']}, "
                  f"SAR={result['sar']:.3f}, "
                  f"SPLat={result['splat']:.1f}")
            
            # 每5个episode打印统计
            if episode % 5 == 0:
                recent_results = results[-5:]
                avg_reward = np.mean([r['reward'] for r in recent_results])
                avg_sar = np.mean([r['sar'] for r in recent_results])
                success_rate = np.mean([r['success'] for r in recent_results])
                
                print(f"\n📊 最近5轮统计:")
                print(f"   平均奖励: {avg_reward:.2f}")
                print(f"   平均SAR: {avg_sar:.3f}")
                print(f"   成功率: {success_rate:.3f}")
                print("-" * 50)
        
        # 总结
        if results:
            avg_reward = np.mean([r['reward'] for r in results])
            avg_sar = np.mean([r['sar'] for r in results])
            success_rate = np.mean([r['success'] for r in results])
            
            print(f"\n🎉 macOS兼容训练完成!")
            print(f"总episodes: {len(results)}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"平均SAR: {avg_sar:.3f}")
            print(f"成功率: {success_rate:.3f}")
            
            # 保存结果
            os.makedirs("../results", exist_ok=True)
            with open("../results/macos_training_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            print("✅ 结果已保存到 results/macos_training_results.json")
        
        return results

def main():
    """主函数"""
    print("🍎 macOS兼容VNF嵌入多智能体训练")
    print("=" * 50)
    
    # 设置种子
    set_seeds(42)
    
    try:
        # 创建训练器
        trainer = MacOSCompatibleTrainer()
        
        # 执行训练
        results = trainer.train()
        
        print("\n🎉 macOS兼容训练完成！")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
