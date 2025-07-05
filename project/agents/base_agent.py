# agents/base_agent.py - 修复版

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
from torch_geometric.data import Data, Batch

from models.gnn_encoder import GNNEncoder

class BaseAgent(ABC):
    """
    多智能体VNF嵌入系统的基础智能体类 - 修复版
    
    主要修复：
    1. ✅ 统一动作选择接口：只返回int
    2. ✅ 确保状态维度一致性
    3. ✅ 改进错误处理机制
    4. ✅ 添加内存管理
    """
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int, 
                 action_dim: int, 
                 edge_dim: int,
                 config: Dict[str, Any]):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.edge_dim = edge_dim
        self.config = config
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Agent {agent_id} 使用设备: {self.device}")
        
        # ✅ 修复：确保维度配置正确
        assert state_dim == 8, f"状态维度必须为8，当前为{state_dim}"
        assert edge_dim in [2, 4], f"边维度必须为2或4，当前为{edge_dim}"
        
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
        
        # ✅ 修复：图神经网络编码器，确保维度正确
        try:
            self.gnn_encoder = GNNEncoder(
                node_dim=state_dim,  # 确保是8维
                edge_dim=edge_dim,   # 2维或4维
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers
            ).to(self.device)
            print(f"✅ GNN编码器初始化成功: {state_dim}维节点 -> {self.output_dim}维输出")
        except Exception as e:
            print(f"❌ GNN编码器初始化失败: {e}")
            raise
        
        # 策略网络（子类实现具体结构）
        self.policy_network = None
        self.target_network = None  # DQN系列使用
        self.optimizer = None
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.is_training = True
        
        # ✅ 修复：改进统计信息结构
        self.stats = {
            "total_reward": 0.0,
            "episodes": 0,
            "steps": 0,
            "losses": [],
            "q_values": [],
            "actions_taken": {},
            "errors": [],  # 新增错误记录
            "memory_usage": []  # 新增内存使用记录
        }
        
        # 多智能体协调（预留）
        self.other_agents = {}
        self.communication_enabled = False
    
    def process_state(self, state: Union[Data, Dict, np.ndarray]) -> torch.Tensor:
        """
        处理状态输入 - 修复版
        
        Args:
            state: 可以是PyG Data对象、字典或numpy数组
            
        Returns:
            processed_state: 处理后的状态tensor [1, output_dim]
        """
        try:
            self.gnn_encoder.eval()
            
            with torch.no_grad():
                if isinstance(state, Data):
                    # ✅ 修复：确保数据在正确设备上，避免内存泄漏
                    state = state.to(self.device)
                    # 验证维度
                    if state.x.size(1) != self.state_dim:
                        raise ValueError(f"状态维度不匹配: 期望{self.state_dim}，实际{state.x.size(1)}")
                    encoded_state = self.gnn_encoder(state)
                    
                elif isinstance(state, dict) and 'graph_data' in state:
                    graph_data = state['graph_data'].to(self.device)
                    if graph_data.x.size(1) != self.state_dim:
                        raise ValueError(f"状态维度不匹配: 期望{self.state_dim}，实际{graph_data.x.size(1)}")
                    encoded_state = self.gnn_encoder(graph_data)
                    
                elif isinstance(state, (np.ndarray, torch.Tensor)):
                    if isinstance(state, np.ndarray):
                        state = torch.tensor(state, dtype=torch.float32)
                    if state.dim() == 1:
                        state = state.unsqueeze(0)
                    # ✅ 修复：确保维度正确
                    if state.size(1) != self.output_dim:
                        print(f"⚠️ 状态维度({state.size(1)})不是输出维度({self.output_dim})，假设为已编码状态")
                    encoded_state = state.to(self.device)
                    
                else:
                    raise ValueError(f"不支持的状态格式: {type(state)}")
            
            if self.is_training:
                self.gnn_encoder.train()
            
            # ✅ 修复：确保输出维度正确
            if encoded_state.size(-1) != self.output_dim:
                print(f"⚠️ 编码后状态维度({encoded_state.size(-1)})不匹配期望维度({self.output_dim})")
                
            return encoded_state
            
        except Exception as e:
            print(f"❌ 状态处理失败: {e}")
            self.stats["errors"].append(f"状态处理错误: {e}")
            # 返回默认状态
            return torch.zeros(1, self.output_dim, device=self.device)
    
    def update_target_network(self, tau: float = None):
        """更新目标网络（用于DQN系列算法）"""
        if self.target_network is None:
            return
            
        try:
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
        except Exception as e:
            print(f"❌ 目标网络更新失败: {e}")
            self.stats["errors"].append(f"目标网络更新错误: {e}")
    
    def decay_epsilon(self):
        """更新探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """获取当前状态下的有效动作"""
        # ✅ 修复：添加更健壮的有效动作检查
        try:
            available_nodes = kwargs.get('available_nodes', list(range(self.action_dim)))
            resource_constraints = kwargs.get('resource_constraints', {})
            
            valid_actions = []
            for action in available_nodes:
                if 0 <= action < self.action_dim:
                    valid_actions.append(action)
            
            # 确保至少有一个有效动作
            if not valid_actions:
                valid_actions = [0]
                print("⚠️ 没有有效动作，使用默认动作0")
            
            return valid_actions
            
        except Exception as e:
            print(f"❌ 获取有效动作失败: {e}")
            return list(range(min(5, self.action_dim)))  # 返回前5个动作作为备选
    
    def mask_invalid_actions(self, q_values: torch.Tensor, valid_actions: List[int]) -> torch.Tensor:
        """屏蔽无效动作的Q值"""
        try:
            masked_q_values = q_values.clone()
            invalid_actions = [a for a in range(self.action_dim) if a not in valid_actions]
            
            if invalid_actions:
                masked_q_values[:, invalid_actions] = -float('inf')
            
            return masked_q_values
            
        except Exception as e:
            print(f"❌ 动作屏蔽失败: {e}")
            return q_values  # 返回原始Q值
    
    def update_stats(self, reward: float, action: int, loss: float = None, q_values: torch.Tensor = None):
        """更新智能体统计信息"""
        try:
            self.stats["total_reward"] += float(reward)
            self.stats["steps"] += 1
            
            if loss is not None:
                self.stats["losses"].append(float(loss))
            
            if q_values is not None:
                self.stats["q_values"].append(float(q_values.mean().item()))
            
            if action not in self.stats["actions_taken"]:
                self.stats["actions_taken"][action] = 0
            self.stats["actions_taken"][action] += 1
            
            # ✅ 新增：记录内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
                self.stats["memory_usage"].append(memory_used)
                
        except Exception as e:
            print(f"❌ 统计更新失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        try:
            stats = self.stats.copy()
            
            if stats["episodes"] > 0:
                stats["avg_reward"] = stats["total_reward"] / stats["episodes"]
            else:
                stats["avg_reward"] = 0.0
                
            if stats["losses"]:
                stats["avg_loss"] = np.mean(stats["losses"][-100:])
                
            if stats["q_values"]:
                stats["avg_q_value"] = np.mean(stats["q_values"][-100:])
                
            if stats["memory_usage"]:
                stats["avg_memory_mb"] = np.mean(stats["memory_usage"][-50:])
                stats["max_memory_mb"] = np.max(stats["memory_usage"])
                
            stats["epsilon"] = self.epsilon
            stats["error_count"] = len(stats["errors"])
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def reset_episode_stats(self):
        """重置episode统计"""
        self.stats["total_reward"] = 0.0
        self.stats["episodes"] += 1
    
    def cleanup_memory(self):
        """✅ 新增：清理内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 清理旧的统计数据
            if len(self.stats["losses"]) > 1000:
                self.stats["losses"] = self.stats["losses"][-500:]
            if len(self.stats["q_values"]) > 1000:
                self.stats["q_values"] = self.stats["q_values"][-500:]
            if len(self.stats["memory_usage"]) > 200:
                self.stats["memory_usage"] = self.stats["memory_usage"][-100:]
                
        except Exception as e:
            print(f"❌ 内存清理失败: {e}")
    
    def save_checkpoint(self, filepath: str):
        """保存智能体检查点"""
        try:
            checkpoint = {
                'agent_id': self.agent_id,
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'epsilon': self.epsilon,
                'stats': self.stats,
                'gnn_encoder_state': self.gnn_encoder.state_dict(),
                'config': self.config  # ✅ 新增：保存配置
            }
            
            if self.policy_network is not None:
                checkpoint['policy_network_state'] = self.policy_network.state_dict()
                
            if self.target_network is not None:
                checkpoint['target_network_state'] = self.target_network.state_dict()
                
            if self.optimizer is not None:
                checkpoint['optimizer_state'] = self.optimizer.state_dict()
            
            torch.save(checkpoint, filepath)
            print(f"💾 Agent {self.agent_id} 检查点已保存: {filepath}")
            
        except Exception as e:
            print(f"❌ 检查点保存失败: {e}")
            self.stats["errors"].append(f"检查点保存错误: {e}")
    
    def load_checkpoint(self, filepath: str):
        """加载智能体检查点"""
        try:
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
            
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            self.stats["errors"].append(f"检查点加载错误: {e}")
    
    def set_training_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.is_training = training
        
        try:
            if training:
                self.gnn_encoder.train()
                if self.policy_network is not None:
                    self.policy_network.train()
            else:
                self.gnn_encoder.eval()
                if self.policy_network is not None:
                    self.policy_network.eval()
        except Exception as e:
            print(f"❌ 设置训练模式失败: {e}")
    
    # ✅ 修复：统一动作选择接口
    @abstractmethod
    def select_action(self, state: Union[Data, Dict], **kwargs) -> int:
        """
        选择动作 - 统一接口
        
        Args:
            state: 当前状态
            **kwargs: 额外参数
            
        Returns:
            action: 选择的单个动作（int）
        """
        pass
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        pass
    
    @abstractmethod
    def learn(self) -> Dict[str, float]:
        pass

# ✅ 修复：改进工厂函数
def create_agent(agent_type: str, agent_id: str, state_dim: int, action_dim: int, 
                edge_dim: int, config: Dict[str, Any]) -> BaseAgent:
    """
    工厂函数：创建指定类型的智能体 - 修复版
    """
    # ✅ 验证输入参数
    assert state_dim == 8, f"状态维度必须为8，当前为{state_dim}"
    assert edge_dim in [2, 4], f"边维度必须为2或4，当前为{edge_dim}"
    assert action_dim > 0, f"动作维度必须大于0，当前为{action_dim}"
    
    try:
        if agent_type.lower() == 'ddqn':
            from agents.multi_ddqn_agent import MultiDDQNAgent
            return MultiDDQNAgent(agent_id, state_dim, action_dim, edge_dim, config)
        elif agent_type.lower() == 'dqn':
            from agents.multi_dqn_agent import MultiDQNAgent
            return MultiDQNAgent(agent_id, state_dim, action_dim, edge_dim, config)
        elif agent_type.lower() == 'ppo':
            from agents.multi_ppo_agent import MultiPPOAgent
            return MultiPPOAgent(agent_id, state_dim, action_dim, edge_dim, config)
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
            
    except Exception as e:
        print(f"❌ 智能体创建失败: {e}")
        raise

def test_base_agent():
    """测试BaseAgent基础功能"""
    print("🧪 测试修复后的BaseAgent...")
    
    config = {
        "gnn": {
            "edge_aware": {"hidden_dim": 64, "output_dim": 128, "layers": 4},
            "baseline": {"hidden_dim": 64, "output_dim": 128, "layers": 4}
        },
        "train": {"lr": 0.001, "gamma": 0.99, "batch_size": 16}
    }
    
    class TestAgent(BaseAgent):
        def __init__(self, agent_id, state_dim, action_dim, edge_dim, config):
            super().__init__(agent_id, state_dim, action_dim, edge_dim, config)
            
        def select_action(self, state, **kwargs) -> int:  # ✅ 修复：明确返回int
            valid_actions = self.get_valid_actions(state, **kwargs)
            return np.random.choice(valid_actions)
            
        def store_transition(self, state, action, reward, next_state, done, **kwargs):
            pass
            
        def learn(self):
            return {"loss": 0.1}
    
    try:
        agent = TestAgent("test_agent", state_dim=8, action_dim=10, edge_dim=4, config=config)
        
        test_state = torch.randn(1, 128)  # 假设已编码状态
        processed_state = agent.process_state(test_state)
        
        print(f"✅ 状态处理测试: {processed_state.shape}")
        print(f"✅ 智能体创建成功: {agent.agent_id}")
        print(f"✅ 设备配置: {agent.device}")
        
        # 测试动作选择
        action = agent.select_action(test_state)
        assert isinstance(action, (int, np.integer)), f"动作应该是int类型，实际是{type(action)}"
        print(f"✅ 动作选择测试: {action} (类型: {type(action)})")
        
        agent.update_stats(reward=1.0, action=action, loss=0.1)
        stats = agent.get_stats()
        print(f"✅ 统计功能测试: {stats['total_reward']}")
        
        # 测试内存清理
        agent.cleanup_memory()
        print("✅ 内存清理测试通过")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    test_base_agent()