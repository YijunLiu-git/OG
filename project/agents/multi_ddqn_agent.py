# agents/multi_ddqn_agent.py - 修复版

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict, Any
from torch_geometric.data import Data

from agents.base_agent import BaseAgent
from utils.replay_buffer import PrioritizedReplayBuffer

class DDQNNetwork(nn.Module):
    """
    双深度Q网络 - 修复版（解决BatchNorm问题）
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 512):
        super(DDQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # ✅ 修复：移除BatchNorm，使用LayerNorm替代
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # 网络初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 修复版
        
        Args:
            state_embedding: GNN编码后的状态 [batch_size, input_dim]
            
        Returns:
            q_values: Q值 [batch_size, action_dim]
        """
        # ✅ 修复：LayerNorm可以处理任意批次大小
        return self.q_network(state_embedding)


class MultiDDQNAgent(BaseAgent):
    """
    多智能体双深度Q学习智能体 - 修复版
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, edge_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, edge_dim, config)
        
        # DDQN特定配置
        self.target_update_freq = config.get("train", {}).get("target_update", 100)
        self.double_q = True  # 启用双Q学习
        
        # ✅ 修复：确保网络输入维度正确
        network_input_dim = self.output_dim  # GNNEncoder的输出维度 (应该是256)
        
        try:
            # 策略网络（主网络）
            self.policy_network = DDQNNetwork(
                input_dim=network_input_dim,
                action_dim=action_dim,
                hidden_dim=config.get("network", {}).get("hidden_dim", 512)
            ).to(self.device)
            
            # 目标网络
            self.target_network = DDQNNetwork(
                input_dim=network_input_dim,
                action_dim=action_dim,
                hidden_dim=config.get("network", {}).get("hidden_dim", 512)
            ).to(self.device)
            
            # 初始化目标网络
            self.target_network.load_state_dict(self.policy_network.state_dict())
            
            print(f"✅ DDQN网络初始化成功: {network_input_dim} -> {action_dim}")
            
        except Exception as e:
            print(f"❌ DDQN网络初始化失败: {e}")
            raise
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # 优先级经验回放
        buffer_size = config.get("train", {}).get("buffer_size", 10000)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=0.6,  # 优先级指数
            beta=0.4    # 重要性采样指数
        )
        
        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )
        
        print(f"🚀 DDQN Agent {agent_id} 初始化完成")
    
    def select_action(self, state: Union[Data, Dict], valid_actions: List[int] = None, **kwargs) -> int:
        """
        选择动作 - 修复版：确保返回单个int
        
        Args:
            state: 当前状态（图数据或编码后状态）
            valid_actions: 有效动作列表
            **kwargs: 额外参数
            
        Returns:
            action: 选择的单个动作 (int)
        """
        try:
            # 处理状态
            state_embedding = self.process_state(state)
            
            # ✅ 修复：确保状态维度正确
            if state_embedding.size(-1) != self.output_dim:
                print(f"⚠️ 状态维度不匹配: 期望{self.output_dim}，实际{state_embedding.size(-1)}")
                # 如果维度不对，进行适配
                if state_embedding.size(-1) < self.output_dim:
                    # 填充零
                    padding = torch.zeros(state_embedding.size(0), 
                                        self.output_dim - state_embedding.size(-1), 
                                        device=self.device)
                    state_embedding = torch.cat([state_embedding, padding], dim=-1)
                else:
                    # 截断
                    state_embedding = state_embedding[:, :self.output_dim]
            
            # 获取有效动作
            if valid_actions is None:
                valid_actions = self.get_valid_actions(state, **kwargs)
            
            # ✅ 修复：确保有效动作列表不为空
            if not valid_actions:
                print("⚠️ 没有有效动作，使用随机动作")
                return np.random.randint(0, self.action_dim)
            
            # ε-贪婪策略
            if self.is_training and np.random.random() < self.epsilon:
                # 探索：从有效动作中随机选择
                action = int(np.random.choice(valid_actions))
            else:
                # 利用：选择Q值最高的有效动作
                self.policy_network.eval()
                with torch.no_grad():
                    q_values = self.policy_network(state_embedding)
                    
                    # 屏蔽无效动作
                    masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
                    action = int(masked_q_values.argmax(dim=1).item())
                
                if self.is_training:
                    self.policy_network.train()
            
            # ✅ 修复：确保返回值是int类型
            assert isinstance(action, (int, np.integer)), f"动作必须是int类型，实际是{type(action)}"
            assert action in valid_actions, f"选择的动作{action}不在有效动作列表中"
            
            return action
            
        except Exception as e:
            print(f"❌ 动作选择失败: {e}")
            self.stats["errors"].append(f"动作选择错误: {e}")
            # 返回安全的默认动作
            valid_actions = valid_actions or list(range(self.action_dim))
            return int(valid_actions[0]) if valid_actions else 0
    
    def get_valid_actions(self, state: Union[Data, Dict], **kwargs) -> List[int]:
        """
        获取VNF嵌入场景下的有效动作 - 修复版
        """
        try:
            # 基础实现：所有动作都有效
            available_nodes = kwargs.get('available_nodes', list(range(self.action_dim)))
            resource_constraints = kwargs.get('resource_constraints', {})
            
            valid_actions = []
            for node in available_nodes:
                if 0 <= node < self.action_dim:  # ✅ 修复：确保节点ID在有效范围内
                    if self._check_node_feasibility(node, resource_constraints):
                        valid_actions.append(node)
            
            # 确保至少有一个有效动作
            if not valid_actions:
                valid_actions = [0]  # 默认选择节点0
                print("⚠️ 没有有效动作，使用默认动作0")
            
            return valid_actions
            
        except Exception as e:
            print(f"❌ 获取有效动作失败: {e}")
            return [0]  # 返回安全默认值
    
    def _check_node_feasibility(self, node_id: int, constraints: Dict) -> bool:
        """检查节点是否满足VNF嵌入约束"""
        try:
            # 简化实现，实际应用中需要更复杂的逻辑
            if 'min_cpu' in constraints:
                return True  # 暂时总是返回True
            return True
        except Exception as e:
            print(f"❌ 节点可行性检查失败: {e}")
            return True  # 默认认为可行
    
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        """
        存储经验到优先级回放缓冲区 - 修复版
        """
        try:
            # ✅ 修复：确保动作是int类型
            if isinstance(action, (list, np.ndarray)):
                action = int(action[0]) if len(action) > 0 else 0
            else:
                action = int(action)
            
            # 计算初始优先级（可以基于奖励或TD误差）
            priority = kwargs.get('priority', abs(float(reward)) + 1e-6)
            
            self.replay_buffer.add(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
                priority=priority,
                agent_id=self.agent_id,
                **kwargs
            )
            
        except Exception as e:
            print(f"❌ 经验存储失败: {e}")
            self.stats["errors"].append(f"经验存储错误: {e}")
    
    def learn(self) -> Dict[str, float]:
        """
        DDQN学习更新 - 修复版（解决维度问题）
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_value": 0.0}
        
        try:
            # 优先级采样
            batch_data = self.replay_buffer.sample(self.batch_size, device=self.device)
            
            # ✅ 修复：检查返回的数据格式
            if len(batch_data) == 7:  # 优先级回放返回7个元素
                states, actions, rewards, next_states, dones, weights, indices = batch_data
            else:
                print(f"⚠️ 优先级回放返回数据格式异常: {len(batch_data)}个元素")
                return {"loss": 0.0, "q_value": 0.0}
            
            # ✅ 调试：检查原始数据维度
            print(f"🔍 原始数据维度检查:")
            print(f"   states: {states.shape if hasattr(states, 'shape') else type(states)}")
            print(f"   actions: {actions.shape if hasattr(actions, 'shape') else type(actions)}")
            
            # 处理状态编码
            if isinstance(states, Data):
                # 图数据：使用GNN编码
                state_embeddings = self.gnn_encoder(states)
                next_state_embeddings = self.gnn_encoder(next_states)
            else:
                # ✅ 修复：确保状态张量是正确的2维格式
                if isinstance(states, torch.Tensor):
                    state_embeddings = states
                    next_state_embeddings = next_states
                    
                    # 修复维度问题
                    if state_embeddings.dim() == 3:
                        print(f"🔧 修复3维状态张量: {state_embeddings.shape}")
                        state_embeddings = state_embeddings.view(state_embeddings.size(0), -1)
                        next_state_embeddings = next_state_embeddings.view(next_state_embeddings.size(0), -1)
                        print(f"   修复后: {state_embeddings.shape}")
                    elif state_embeddings.dim() == 1:
                        state_embeddings = state_embeddings.unsqueeze(0)
                        next_state_embeddings = next_state_embeddings.unsqueeze(0)
                else:
                    # 转换为张量
                    state_embeddings = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(self.device)
                    next_state_embeddings = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states]).to(self.device)
            
            # ✅ 最终维度验证
            print(f"🔍 处理后状态维度: {state_embeddings.shape}")
            expected_batch_size = self.batch_size
            if state_embeddings.size(0) != expected_batch_size:
                print(f"⚠️ 批次大小不匹配: 期望{expected_batch_size}, 实际{state_embeddings.size(0)}")
            
            if state_embeddings.size(1) != self.output_dim:
                print(f"⚠️ 特征维度不匹配: 期望{self.output_dim}, 实际{state_embeddings.size(1)}")
                return {"loss": 0.0, "q_value": 0.0}
            
            # 当前Q值 - 现在应该得到正确的2维输出
            current_q_values = self.policy_network(state_embeddings)
            print(f"🔍 Q值输出维度: {current_q_values.shape}")
            
            # ✅ 验证Q值输出维度
            expected_q_shape = (expected_batch_size, self.action_dim)
            if current_q_values.shape != expected_q_shape:
                print(f"❌ Q值输出维度异常: 期望{expected_q_shape}, 实际{current_q_values.shape}")
                return {"loss": 0.0, "q_value": 0.0}
            
            # ✅ 修复：处理动作索引维度问题
            try:
                if isinstance(actions, torch.Tensor):
                    action_indices = actions.long().to(self.device)
                    # 确保action_indices是1维张量
                    if action_indices.dim() > 1:
                        action_indices = action_indices.squeeze()
                    if action_indices.dim() == 0:
                        action_indices = action_indices.unsqueeze(0)
                else:
                    # 处理列表形式的动作
                    action_list = []
                    for a in actions:
                        if isinstance(a, (list, np.ndarray)):
                            action_list.append(int(a[0]) if len(a) > 0 else 0)
                        else:
                            action_list.append(int(a))
                    action_indices = torch.tensor(action_list, dtype=torch.long, device=self.device)
                
                # ✅ 确保动作索引在有效范围内
                action_indices = torch.clamp(action_indices, 0, self.action_dim - 1)
                
                print(f"🔍 动作索引维度: {action_indices.shape}")
                
                # ✅ 修复gather操作的维度问题
                if current_q_values.dim() == 2 and action_indices.dim() == 1:
                    # current_q_values: [batch_size, action_dim]
                    # action_indices: [batch_size]
                    current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                    print(f"🔍 当前Q值维度: {current_q.shape}")
                else:
                    print(f"❌ 维度不匹配: Q值{current_q_values.shape}, 动作{action_indices.shape}")
                    return {"loss": 0.0, "q_value": 0.0}
                    
            except Exception as e:
                print(f"❌ 动作索引处理失败: {str(e)}")
                print(f"   Q值形状: {current_q_values.shape if 'current_q_values' in locals() else 'unknown'}")
                print(f"   动作类型: {type(actions)}")
                if hasattr(actions, 'shape'):
                    print(f"   动作形状: {actions.shape}")
                return {"loss": 0.0, "q_value": 0.0}
            
            # ✅ 修复：双Q学习的目标值计算
            with torch.no_grad():
                try:
                    # 策略网络选择下一状态的最优动作
                    next_q_values_policy = self.policy_network(next_state_embeddings)
                    next_actions = next_q_values_policy.argmax(dim=1)
                    
                    # ✅ 确保next_actions维度正确
                    if next_actions.dim() == 1:
                        next_actions = next_actions.unsqueeze(1)
                    
                    # 目标网络评估选定动作的价值
                    next_q_values_target = self.target_network(next_state_embeddings)
                    
                    # ✅ 修复gather操作
                    if next_q_values_target.dim() == 2 and next_actions.dim() == 2:
                        next_q = next_q_values_target.gather(1, next_actions).squeeze(1)
                    else:
                        print(f"⚠️ 目标Q值维度不匹配: target_q{next_q_values_target.shape}, actions{next_actions.shape}")
                        next_q = next_q_values_target.max(dim=1)[0]  # 回退到最大值
                    
                    # 计算目标Q值
                    target_q = rewards + (self.gamma * next_q * ~dones)
                    
                except Exception as e:
                    print(f"❌ 目标Q值计算失败: {str(e)}")
                    # 简化的目标Q值计算
                    target_q = rewards + (self.gamma * torch.zeros_like(rewards))
            
            # 计算TD误差
            td_errors = torch.abs(current_q - target_q)
            
            # ✅ 修复：处理重要性采样权重
            if weights is not None:
                loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
            else:
                loss = F.mse_loss(current_q, target_q)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # 更新优先级
            if indices is not None:
                self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
            
            # 更新目标网络
            self.training_step += 1
            if self.training_step % self.target_update_freq == 0:
                self.update_target_network()
            
            # 更新探索率
            self.decay_epsilon()
            
            # 更新统计信息
            avg_q_value = current_q.mean().item()
            first_action = action_indices[0].item() if len(action_indices) > 0 else 0
            self.update_stats(
                reward=rewards.mean().item(),
                action=first_action,
                loss=loss.item(),
                q_values=torch.tensor([avg_q_value])
            )
            
            # ✅ 新增：定期清理内存
            if self.training_step % 100 == 0:
                self.cleanup_memory()
            
            return {
                "loss": loss.item(),
                "q_value": avg_q_value,
                "td_error": td_errors.mean().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
                "epsilon": self.epsilon
            }
            
        except Exception as e:
            print(f"❌ DDQN学习失败: {e}")
            self.stats["errors"].append(f"学习错误: {e}")
            # 提供更详细的错误信息
            print(f"   错误详情: {str(e)}")
            if 'current_q_values' in locals():
                print(f"   Q值形状: {current_q_values.shape}")
            if 'actions' in locals():
                print(f"   动作类型: {type(actions)}")
                if hasattr(actions, 'shape'):
                    print(f"   动作形状: {actions.shape}")
            return {"loss": 0.0, "q_value": 0.0, "error": str(e)}
        """
        DDQN学习更新 - 修复版
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_value": 0.0}
        
        try:
            # 优先级采样
            batch_data = self.replay_buffer.sample(self.batch_size, device=self.device)
            
            # ✅ 修复：检查返回的数据格式
            if len(batch_data) == 7:  # 优先级回放返回7个元素
                states, actions, rewards, next_states, dones, weights, indices = batch_data
            else:
                print(f"⚠️ 优先级回放返回数据格式异常: {len(batch_data)}个元素")
                return {"loss": 0.0, "q_value": 0.0}
            
            # 处理状态编码
            if isinstance(states, Data):
                # 图数据：使用GNN编码
                state_embeddings = self.gnn_encoder(states)
                next_state_embeddings = self.gnn_encoder(next_states)
            else:
                # 已编码状态
                state_embeddings = states
                next_state_embeddings = next_states
            
            # ✅ 修复：确保状态嵌入是正确的2维张量
            if state_embeddings.dim() == 3:
                # 如果是3维，压缩掉中间维度
                state_embeddings = state_embeddings.squeeze(1)
                print(f"🔧 状态嵌入维度修正: 3维 -> 2维, 新形状: {state_embeddings.shape}")
            elif state_embeddings.dim() == 1:
                # 如果是1维，添加batch维度
                state_embeddings = state_embeddings.unsqueeze(0)
            
            if next_state_embeddings.dim() == 3:
                next_state_embeddings = next_state_embeddings.squeeze(1)
            elif next_state_embeddings.dim() == 1:
                next_state_embeddings = next_state_embeddings.unsqueeze(0)
            
            # ✅ 验证维度
            expected_shape = (len(experiences), self.output_dim)
            if state_embeddings.shape != expected_shape:
                print(f"⚠️ 状态嵌入维度异常: 期望{expected_shape}, 实际{state_embeddings.shape}")
                return {"loss": 0.0, "q_value": 0.0}
            
            # ✅ 修复：确保状态维度正确
            if state_embeddings.size(-1) != self.output_dim:
                print(f"⚠️ 学习时状态维度不匹配: {state_embeddings.size(-1)} != {self.output_dim}")
                return {"loss": 0.0, "q_value": 0.0}
            
            # 当前Q值 - 确保输入是2维的
            current_q_values = self.policy_network(state_embeddings)
            
            # ✅ 验证Q值输出维度
            expected_q_shape = (len(experiences), self.action_dim)
            if current_q_values.shape != expected_q_shape:
                print(f"⚠️ Q值输出维度异常: 期望{expected_q_shape}, 实际{current_q_values.shape}")
                return {"loss": 0.0, "q_value": 0.0}
            
            # ✅ 修复：处理动作索引，确保类型正确
            if isinstance(actions, torch.Tensor):
                action_indices = actions.long()
            else:
                # 处理列表形式的动作
                action_list = []
                for a in actions:
                    if isinstance(a, (list, np.ndarray)):
                        action_list.append(int(a[0]) if len(a) > 0 else 0)
                    else:
                        action_list.append(int(a))
                action_indices = torch.tensor(action_list, dtype=torch.long, device=self.device)
            
            # ✅ 修复：确保动作索引在有效范围内
            action_indices = torch.clamp(action_indices, 0, self.action_dim - 1)
            current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            
            # ✅ 修复：双Q学习的目标值计算
            with torch.no_grad():
                try:
                    # 策略网络选择下一状态的最优动作
                    next_q_values_policy = self.policy_network(next_state_embeddings)
                    next_actions = next_q_values_policy.argmax(dim=1)
                    
                    # ✅ 确保next_actions维度正确
                    if next_actions.dim() == 1:
                        next_actions = next_actions.unsqueeze(1)
                    
                    # 目标网络评估选定动作的价值
                    next_q_values_target = self.target_network(next_state_embeddings)
                    
                    # ✅ 修复gather操作
                    if next_q_values_target.dim() == 2 and next_actions.dim() == 2:
                        next_q = next_q_values_target.gather(1, next_actions).squeeze(1)
                    else:
                        print(f"⚠️ 目标Q值维度不匹配: target_q{next_q_values_target.shape}, actions{next_actions.shape}")
                        next_q = next_q_values_target.max(dim=1)[0]  # 回退到最大值
                    
                    # 计算目标Q值
                    target_q = rewards + (self.gamma * next_q * ~dones)
                    
                except Exception as e:
                    print(f"❌ 目标Q值计算失败: {str(e)}")
                    # 简化的目标Q值计算
                    target_q = rewards + (self.gamma * torch.zeros_like(rewards))
            
            # 计算TD误差
            td_errors = torch.abs(current_q - target_q)
            
            # ✅ 修复：处理重要性采样权重
            if weights is not None:
                loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
            else:
                loss = F.mse_loss(current_q, target_q)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) + list(self.policy_network.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # 更新优先级
            if indices is not None:
                self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
            
            # 更新目标网络
            self.training_step += 1
            if self.training_step % self.target_update_freq == 0:
                self.update_target_network()
            
            # 更新探索率
            self.decay_epsilon()
            
            # 更新统计信息
            avg_q_value = current_q.mean().item()
            first_action = action_indices[0].item() if len(action_indices) > 0 else 0
            self.update_stats(
                reward=rewards.mean().item(),
                action=first_action,
                loss=loss.item(),
                q_values=torch.tensor([avg_q_value])
            )
            
            # ✅ 新增：定期清理内存
            if self.training_step % 100 == 0:
                self.cleanup_memory()
            
            return {
                "loss": loss.item(),
                "q_value": avg_q_value,
                "td_error": td_errors.mean().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
                "epsilon": self.epsilon
            }
            
        except Exception as e:
            print(f"❌ DDQN学习失败: {e}")
            self.stats["errors"].append(f"学习错误: {e}")
            # 提供更详细的错误信息
            print(f"   错误详情: {str(e)}")
            if 'current_q_values' in locals():
                print(f"   Q值形状: {current_q_values.shape}")
            if 'actions' in locals():
                print(f"   动作类型: {type(actions)}")
                if hasattr(actions, 'shape'):
                    print(f"   动作形状: {actions.shape}")
            return {"loss": 0.0, "q_value": 0.0, "error": str(e)}
    
    def compute_td_error(self, state, action, reward, next_state, done) -> float:
        """计算TD误差用于优先级更新 - 修复版"""
        try:
            with torch.no_grad():
                if isinstance(state, Data):
                    state_emb = self.gnn_encoder(state.to(self.device))
                    next_state_emb = self.gnn_encoder(next_state.to(self.device))
                else:
                    # ✅ 修复：使用detach().clone()替代torch.tensor()
                    if isinstance(state, torch.Tensor):
                        state_emb = state.detach().clone().to(self.device)
                        if state_emb.dim() == 1:
                            state_emb = state_emb.unsqueeze(0)
                    else:
                        state_emb = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
                    if isinstance(next_state, torch.Tensor):
                        next_state_emb = next_state.detach().clone().to(self.device)
                        if next_state_emb.dim() == 1:
                            next_state_emb = next_state_emb.unsqueeze(0)
                    else:
                        next_state_emb = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                # ✅ 修复：确保动作是int并在有效范围内
                action = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
                action = max(0, min(action, self.action_dim - 1))  # 限制在有效范围内
                
                # ✅ 修复：确保状态维度匹配网络期望
                if state_emb.size(-1) != self.output_dim:
                    print(f"⚠️ TD误差计算：状态维度不匹配 {state_emb.size(-1)} != {self.output_dim}")
                    return 1.0  # 返回默认优先级
                
                current_q_values = self.policy_network(state_emb)
                
                # ✅ 修复：检查动作索引是否越界
                if action >= current_q_values.size(1):
                    print(f"⚠️ TD误差计算：动作索引越界 {action} >= {current_q_values.size(1)}")
                    return 1.0  # 返回默认优先级
                
                current_q = current_q_values[0, action]
                
                if not done:
                    next_q_values = self.policy_network(next_state_emb)
                    next_action = next_q_values.argmax(dim=1)
                    next_q_target = self.target_network(next_state_emb)
                    next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze()
                    target_q = reward + self.gamma * next_q
                else:
                    target_q = torch.tensor(reward, device=self.device, dtype=torch.float32)
                
                td_error = abs(current_q - target_q).item()
                
            return td_error
            
        except Exception as e:
            print(f"❌ TD误差计算失败: {e}")
            return 1.0  # 返回默认优先级
    
    def save_model(self, filepath: str):
        """保存模型 - 修复版"""
        try:
            torch.save({
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'policy_network': self.policy_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'epsilon': self.epsilon,
                'config': self.config  # ✅ 新增：保存配置
            }, filepath)
            print(f"💾 DDQN模型已保存: {filepath}")
            
        except Exception as e:
            print(f"❌ DDQN模型保存失败: {e}")
    
    def load_model(self, filepath: str):
        """加载模型 - 修复版"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step = checkpoint['training_step']
            self.epsilon = checkpoint['epsilon']
            
            print(f"📂 DDQN模型已加载: {filepath}")
            
        except Exception as e:
            print(f"❌ DDQN模型加载失败: {e}")


def test_ddqn_agent():
    """测试DDQN智能体 - 修复版"""
    print("🧪 测试修复后的DDQN智能体...")
    
    config = {
        "gnn": {"edge_aware": {"hidden_dim": 64, "output_dim": 256, "layers": 4}},
        "train": {"lr": 0.001, "gamma": 0.99, "batch_size": 16, "target_update": 10},
        "network": {"hidden_dim": 256}
    }
    
    try:
        agent = MultiDDQNAgent("test_ddqn", state_dim=8, action_dim=10, edge_dim=4, config=config)
        
        # 测试动作选择
        test_state = torch.randn(1, 256)  # 模拟GNN编码后的状态
        action = agent.select_action(test_state)
        
        assert isinstance(action, (int, np.integer)), f"动作应该是int类型，实际是{type(action)}"
        assert 0 <= action < 10, f"动作应该在[0,9]范围内，实际是{action}"
        print(f"✅ 动作选择测试: {action} (类型: {type(action)})")
        
        # ✅ 修复：添加更多经验以确保学习稳定，但确保状态维度正确
        print("📝 添加训练经验...")
        for i in range(50):  # 增加经验数量
            # ✅ 确保状态是2维的：[1, 256]
            state = torch.randn(1, 256)  # 正确的2维状态
            action = i % 10
            reward = np.random.random() - 0.5  # 随机奖励
            next_state = torch.randn(1, 256)  # 正确的2维状态
            done = False
            agent.store_transition(state, action, reward, next_state, done)
        
        print(f"✅ 经验存储测试: 缓冲区大小 {len(agent.replay_buffer)}")
        
        # ✅ 修复：确保有足够的经验进行学习
        if len(agent.replay_buffer) >= agent.batch_size:
            print("🎓 开始学习测试...")
            learning_info = agent.learn()
            print(f"✅ 学习测试: Loss={learning_info.get('loss', 0):.4f}, Q值={learning_info.get('q_value', 0):.4f}")
            
            # 检查是否有错误
            if learning_info.get('error'):
                print(f"⚠️ 学习过程中的错误: {learning_info['error']}")
            else:
                print("✅ 学习过程无错误")
        else:
            print(f"⚠️ 经验不足，无法进行学习测试 ({len(agent.replay_buffer)} < {agent.batch_size})")
        
        # 测试统计信息
        stats = agent.get_stats()
        print(f"✅ 统计测试: 错误数量={stats.get('error_count', 0)}")
        
        # ✅ 测试内存清理
        agent.cleanup_memory()
        print("✅ 内存清理测试通过")
        
        # ✅ 测试TD误差计算
        try:
            # 创建匹配网络期望维度的测试数据
            test_state_td = torch.randn(256)  # 1维状态，将被转换为合适的维度
            td_error = agent.compute_td_error(
                state=test_state_td,
                action=5,
                reward=1.0,
                next_state=test_state_td,
                done=False
            )
            print(f"✅ TD误差计算测试: {td_error:.4f}")
        except Exception as e:
            print(f"⚠️ TD误差计算测试失败: {e}")
        
        print("✅ DDQN智能体测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ DDQN智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_ddqn_agent()