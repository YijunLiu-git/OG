# agents/enhanced_base_agent.py - 修复增强智能体集成

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from torch_geometric.data import Data, Batch

from models.gnn_encoder import GNNEncoder
from models.enhanced_gnn_encoder import EnhancedEdgeAwareGNN

class EnhancedBaseAgent(ABC):
    """
    增强的基础智能体类 - 支持Enhanced GNN编码器
    
    主要修复：
    1. 正确处理增强GNN编码器的输入
    2. 处理VNF上下文和网络状态
    3. 兼容原有和增强的编码器
    """
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int, 
                 action_dim: int, 
                 edge_dim: int,
                 config: Dict[str, Any],
                 use_enhanced_gnn: bool = False):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.edge_dim = edge_dim
        self.config = config
        self.use_enhanced_gnn = use_enhanced_gnn
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Agent {agent_id} 使用设备: {self.device}")
        
        # 选择 GNN 配置
        gnn_config = config.get("gnn", {}).get("edge_aware" if "edge_aware" in agent_id else "baseline", {})
        self.hidden_dim = gnn_config.get("hidden_dim", 128)
        self.output_dim = gnn_config.get("output_dim", 256)
        self.num_layers = gnn_config.get("layers", 4)
        
        # 训练配置
        self.learning_rate = config.get("train", {}).get("lr", 0.001)
        self.gamma = config.get("train", {}).get("gamma", 0.99)
        self.batch_size = config.get("train", {}).get("batch_size", 32)
        
        # 探索配置
        self.epsilon = config.get("train", {}).get("epsilon_start", 1.0)
        self.epsilon_decay = config.get("train", {}).get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("train", {}).get("epsilon_min", 0.01)
        
        # 初始化GNN编码器
        self._setup_gnn_encoder()
        
        # 策略网络（子类实现具体结构）
        self.policy_network = None
        self.target_network = None  # DQN系列使用
        self.optimizer = None
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.is_training = True
        
        # 统计信息
        self.stats = {
            "total_reward": 0.0,
            "episodes": 0,
            "steps": 0,
            "losses": [],
            "q_values": [],
            "actions_taken": {}
        }
        
        # 多智能体协调（预留）
        self.other_agents = {}
        self.communication_enabled = False
        
    def _setup_gnn_encoder(self):
        """设置GNN编码器"""
        if self.use_enhanced_gnn:
            # 使用增强的GNN编码器
            self.gnn_encoder = EnhancedEdgeAwareGNN(
                node_dim=self.state_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                vnf_context_dim=self.config.get('dimensions', {}).get('vnf_context_dim', 6)
            ).to(self.device)
            print(f"✅ {self.agent_id}: 使用增强GNN编码器")
        else:
            # 使用标准GNN编码器
            self.gnn_encoder = GNNEncoder(
                node_dim=self.state_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers
            ).to(self.device)
            print(f"✅ {self.agent_id}: 使用标准GNN编码器")
    
    def process_state(self, state: Union[Data, Dict, np.ndarray]) -> torch.Tensor:
        """
        处理状态输入 - 增强版本
        
        Args:
            state: 可以是PyG Data对象、字典或numpy数组
            
        Returns:
            processed_state: 处理后的状态tensor [1, output_dim]
        """
        self.gnn_encoder.eval()
        
        with torch.no_grad():
            if isinstance(state, Data):
                state = state.to(self.device)
                
                if self.use_enhanced_gnn:
                    # 增强GNN编码器：直接处理原始数据
                    encoded_state = self.gnn_encoder(state)
                else:
                    # 标准GNN编码器：可能需要预处理
                    encoded_state = self.gnn_encoder(state)
                
            elif isinstance(state, dict) and 'graph_data' in state:
                graph_data = state['graph_data'].to(self.device)
                
                if self.use_enhanced_gnn:
                    encoded_state = self.gnn_encoder(graph_data)
                else:
                    encoded_state = self.gnn_encoder(graph_data)
                
            elif isinstance(state, (np.ndarray, torch.Tensor)):
                if isinstance(state, np.ndarray):
                    state = torch.tensor(state, dtype=torch.float32)
                encoded_state = state.unsqueeze(0).to(self.device)
                
            else:
                raise ValueError(f"不支持的状态格式: {type(state)}")
        
        if self.is_training:
            self.gnn_encoder.train()
            
        return encoded_state
    
    def update_target_network(self, tau: float = None):
        """更新目标网络（用于DQN系列算法）"""
        if self.target_network is None:
            return
            
        if tau is None:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        else:
            for target_param, policy_param in zip(
                self.target_network.parameters(), 
                self.policy_network.parameters()
            ):
                target_param.data.copy_(
                    tau * policy_param.data + (1 - tau) * target_param.data
                )
    
    def decay_epsilon(self):
        """更新探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """获取当前状态下的有效动作"""
        return list(range(self.action_dim))
    
    def mask_invalid_actions(self, q_values: torch.Tensor, valid_actions: List[int]) -> torch.Tensor:
        """屏蔽无效动作的Q值"""
        masked_q_values = q_values.clone()
        invalid_actions = [a for a in range(self.action_dim) if a not in valid_actions]
        
        if invalid_actions:
            masked_q_values[:, invalid_actions] = -float('inf')
        
        return masked_q_values
    
    def update_stats(self, reward: float, action: int, loss: float = None, q_values: torch.Tensor = None):
        """更新智能体统计信息"""
        self.stats["total_reward"] += reward
        self.stats["steps"] += 1
        
        if loss is not None:
            self.stats["losses"].append(loss)
        
        if q_values is not None:
            self.stats["q_values"].append(q_values.mean().item())
        
        if action not in self.stats["actions_taken"]:
            self.stats["actions_taken"][action] = 0
        self.stats["actions_taken"][action] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        stats = self.stats.copy()
        
        if stats["episodes"] > 0:
            stats["avg_reward"] = stats["total_reward"] / stats["episodes"]
        else:
            stats["avg_reward"] = 0.0
            
        if stats["losses"]:
            stats["avg_loss"] = np.mean(stats["losses"][-100:])
            
        if stats["q_values"]:
            stats["avg_q_value"] = np.mean(stats["q_values"][-100:])
            
        stats["epsilon"] = self.epsilon
        
        return stats
    
    def reset_episode_stats(self):
        """重置episode统计"""
        self.stats["total_reward"] = 0.0
        self.stats["episodes"] += 1
    
    def save_checkpoint(self, filepath: str):
        """保存智能体检查点"""
        checkpoint = {
            'agent_id': self.agent_id,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'stats': self.stats,
            'gnn_encoder_state': self.gnn_encoder.state_dict(),
            'use_enhanced_gnn': self.use_enhanced_gnn,
        }
        
        if self.policy_network is not None:
            checkpoint['policy_network_state'] = self.policy_network.state_dict()
            
        if self.target_network is not None:
            checkpoint['target_network_state'] = self.target_network.state_dict()
            
        if self.optimizer is not None:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"💾 Agent {self.agent_id} 检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载智能体检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        self.stats = checkpoint['stats']
        
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state'])
        
        if 'policy_network_state' in checkpoint and self.policy_network is not None:
            self.policy_network.load_state_dict(checkpoint['policy_network_state'])
            
        if 'target_network_state' in checkpoint and self.target_network is not None:
            self.target_network.load_state_dict(checkpoint['target_network_state'])
            
        if 'optimizer_state' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"📂 Agent {self.agent_id} 检查点已加载: {filepath}")
    
    def set_training_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.is_training = training
        
        if training:
            self.gnn_encoder.train()
            if self.policy_network is not None:
                self.policy_network.train()
        else:
            self.gnn_encoder.eval()
            if self.policy_network is not None:
                self.policy_network.eval()
    
    def register_other_agents(self, agents: Dict[str, 'EnhancedBaseAgent']):
        """注册其他智能体（用于多智能体协调）"""
        self.other_agents = agents
        self.communication_enabled = len(agents) > 0
        print(f"🤝 Agent {self.agent_id} 已注册 {len(agents)} 个其他智能体")
    
    @abstractmethod
    def select_action(self, state: Union[Data, Dict], **kwargs) -> Union[int, List[int]]:
        pass
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        pass
    
    @abstractmethod
    def learn(self) -> Dict[str, float]:
        pass


def create_enhanced_agent(agent_type: str, agent_id: str, state_dim: int, action_dim: int, 
                         edge_dim: int, config: Dict[str, Any], use_enhanced_gnn: bool = False) -> EnhancedBaseAgent:
    """
    工厂函数：创建增强版智能体
    
    Args:
        agent_type: 智能体类型 ('ddqn', 'dqn', 'ppo')
        agent_id: 智能体ID
        state_dim: 状态维度
        action_dim: 动作维度
        edge_dim: 边特征维度
        config: 配置字典
        use_enhanced_gnn: 是否使用增强GNN编码器
        
    Returns:
        agent: 创建的增强智能体实例
    """
    
    if agent_type.lower() == 'ddqn':
        from agents.enhanced_multi_ddqn_agent import EnhancedMultiDDQNAgent
        return EnhancedMultiDDQNAgent(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
    elif agent_type.lower() == 'dqn':
        from agents.enhanced_multi_dqn_agent import EnhancedMultiDQNAgent
        return EnhancedMultiDQNAgent(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
    elif agent_type.lower() == 'ppo':
        from agents.enhanced_multi_ppo_agent import EnhancedMultiPPOAgent
        return EnhancedMultiPPOAgent(agent_id, state_dim, action_dim, edge_dim, config, use_enhanced_gnn)
    else:
        raise ValueError(f"不支持的智能体类型: {agent_type}")


# 修复现有agents的兼容性
def patch_existing_agents():
    """
    为现有智能体添加增强GNN支持的补丁函数
    """
    
    # 修补DDQN智能体
    from agents.multi_ddqn_agent import MultiDDQNAgent
    
    original_init = MultiDDQNAgent.__init__
    
    def enhanced_init(self, agent_id, state_dim, action_dim, edge_dim, config):
        # 调用原始初始化
        original_init(self, agent_id, state_dim, action_dim, edge_dim, config)
        
        # 检查是否需要使用增强GNN
        if "enhanced" in agent_id or "edge_aware" in agent_id:
            from models.enhanced_gnn_encoder import EnhancedEdgeAwareGNN
            
            # 替换为增强GNN编码器
            self.gnn_encoder = EnhancedEdgeAwareGNN(
                node_dim=state_dim,
                edge_dim=edge_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                vnf_context_dim=config.get('dimensions', {}).get('vnf_context_dim', 6)
            ).to(self.device)
            
            print(f"🔄 {agent_id}: 已升级为增强GNN编码器")
    
    # 应用补丁
    MultiDDQNAgent.__init__ = enhanced_init
    
    # 修补DQN智能体
    from agents.multi_dqn_agent import MultiDQNAgent
    
    original_dqn_init = MultiDQNAgent.__init__
    
    def enhanced_dqn_init(self, agent_id, state_dim, action_dim, edge_dim, config):
        original_dqn_init(self, agent_id, state_dim, action_dim, edge_dim, config)
        
        if "enhanced" in agent_id or "edge_aware" in agent_id:
            from models.enhanced_gnn_encoder import EnhancedEdgeAwareGNN
            
            self.gnn_encoder = EnhancedEdgeAwareGNN(
                node_dim=state_dim,
                edge_dim=edge_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                vnf_context_dim=config.get('dimensions', {}).get('vnf_context_dim', 6)
            ).to(self.device)
            
            print(f"🔄 {agent_id}: 已升级为增强GNN编码器")
    
    MultiDQNAgent.__init__ = enhanced_dqn_init
    
    # 修补PPO智能体
    from agents.multi_ppo_agent import MultiPPOAgent
    
    original_ppo_init = MultiPPOAgent.__init__
    
    def enhanced_ppo_init(self, agent_id, state_dim, action_dim, edge_dim, config):
        original_ppo_init(self, agent_id, state_dim, action_dim, edge_dim, config)
        
        if "enhanced" in agent_id or "edge_aware" in agent_id:
            from models.enhanced_gnn_encoder import EnhancedEdgeAwareGNN
            
            self.gnn_encoder = EnhancedEdgeAwareGNN(
                node_dim=state_dim,
                edge_dim=edge_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                vnf_context_dim=config.get('dimensions', {}).get('vnf_context_dim', 6)
            ).to(self.device)
            
            print(f"🔄 {agent_id}: 已升级为增强GNN编码器")
    
    MultiPPOAgent.__init__ = enhanced_ppo_init
    
    print("✅ 现有智能体增强GNN补丁应用完成")


# 测试函数
def test_enhanced_agent_integration():
    """测试增强智能体集成"""
    print("🧪 测试增强智能体集成...")
    
    # 应用补丁
    patch_existing_agents()
    
    # 测试配置
    config = {
        "gnn": {
            "edge_aware": {"hidden_dim": 128, "output_dim": 256, "layers": 4},
            "baseline": {"hidden_dim": 64, "output_dim": 256, "layers": 4}
        },
        "train": {"lr": 0.001, "gamma": 0.99, "batch_size": 16},
        "dimensions": {"vnf_context_dim": 6}
    }
    
    # 测试增强智能体创建
    from agents.multi_ddqn_agent import MultiDDQNAgent
    
    # 创建增强版智能体
    enhanced_agent = MultiDDQNAgent(
        "ddqn_edge_aware_enhanced", 
        state_dim=8, 
        action_dim=10, 
        edge_dim=4, 
        config=config
    )
    
    # 创建标准版智能体
    baseline_agent = MultiDDQNAgent(
        "ddqn_baseline", 
        state_dim=8, 
        action_dim=10, 
        edge_dim=4, 
        config=config
    )
    
    print(f"✅ 增强智能体类型: {type(enhanced_agent.gnn_encoder).__name__}")
    print(f"✅ 标准智能体类型: {type(baseline_agent.gnn_encoder).__name__}")
    
    # 测试状态处理
    test_state = torch.randn(1, 256)  # 模拟处理后的状态
    
    try:
        enhanced_output = enhanced_agent.process_state(test_state)
        baseline_output = baseline_agent.process_state(test_state)
        
        print(f"✅ 增强智能体输出: {enhanced_output.shape}")
        print(f"✅ 标准智能体输出: {baseline_output.shape}")
        
    except Exception as e:
        print(f"❌ 状态处理测试失败: {e}")
    
    print("✅ 增强智能体集成测试完成!")


if __name__ == "__main__":
    test_enhanced_agent_integration()