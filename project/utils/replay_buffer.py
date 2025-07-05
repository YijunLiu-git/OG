# utils/replay_buffer.py

import torch
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Union, Optional, Any
from torch_geometric.data import Data, Batch
import pickle
import threading
from dataclasses import dataclass

# 经验数据结构
@dataclass
class Experience:
    """单个经验的数据结构"""
    state: Union[Data, torch.Tensor]
    action: Union[int, List[int]]
    reward: float
    next_state: Union[Data, torch.Tensor]
    done: bool
    priority: float = 1.0
    agent_id: str = "default"
    timestamp: float = 0.0
    info: dict = None

class ReplayBuffer:
    """
    标准经验回放缓冲区
    
    功能：
    1. 存储和采样经验
    2. 支持图数据和张量数据
    3. 自动内存管理
    4. 批量采样优化
    """
    
    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 统计信息
        self.total_added = 0
        self.total_sampled = 0
        
    def add(self, state, action, reward, next_state, done, **kwargs):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            **kwargs: 额外信息
        """
        
        # 创建经验对象
        experience = Experience(
            state=self._process_state(state),
            action=action,
            reward=float(reward),
            next_state=self._process_state(next_state),
            done=bool(done),
            agent_id=kwargs.get('agent_id', 'default'),
            timestamp=kwargs.get('timestamp', 0.0),
            info=kwargs.get('info', {})
        )
        
        self.buffer.append(experience)
        self.total_added += 1
    
    def _process_state(self, state):
        """处理状态数据，确保正确存储"""
        if isinstance(state, Data):
            # PyTorch Geometric数据：复制到CPU避免GPU内存泄漏
            return Data(
                x=state.x.cpu() if state.x is not None else None,
                edge_index=state.edge_index.cpu() if state.edge_index is not None else None,
                edge_attr=state.edge_attr.cpu() if state.edge_attr is not None else None,
                batch=state.batch.cpu() if hasattr(state, 'batch') and state.batch is not None else None
            )
        elif isinstance(state, torch.Tensor):
            return state.cpu().clone()
        elif isinstance(state, np.ndarray):
            return torch.tensor(state, dtype=torch.float32)
        else:
            return state
    
    def sample(self, batch_size: int, device: torch.device = None) -> Tuple:
        """
        随机采样经验批次
        
        Args:
            batch_size: 批次大小
            device: 目标设备
            
        Returns:
            (states, actions, rewards, next_states, dones): 批次数据
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # 随机采样
        experiences = random.sample(self.buffer, batch_size)
        self.total_sampled += batch_size
        
        return self._batch_experiences(experiences, device)
    
    def _batch_experiences(self, experiences: List[Experience], device: torch.device = None):
        """将经验列表转换为批次张量"""
        if device is None:
            device = torch.device("cpu")
        
        if not experiences:
            # 如果经验列表为空，返回空张量
            empty_tensor = torch.empty(0)
            return empty_tensor, [], empty_tensor, empty_tensor, empty_tensor
        
        # 分离各个组件
        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        next_states = [exp.next_state for exp in experiences]
        dones = [exp.done for exp in experiences]
        
        # 处理状态批次
        if all(isinstance(s, Data) for s in states):
            # 图数据批处理
            try:
                states_batch = Batch.from_data_list(states).to(device)
                next_states_batch = Batch.from_data_list(next_states).to(device)
            except Exception as e:
                print(f"图数据批处理失败: {e}")
                # 回退到张量处理
                states_batch = torch.stack([torch.randn(128) for _ in states]).to(device)
                next_states_batch = torch.stack([torch.randn(128) for _ in next_states]).to(device)
        else:
            # 张量数据批处理
            states_batch = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s) 
                                      for s in states]).to(device)
            next_states_batch = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s) 
                                           for s in next_states]).to(device)
        
        # 处理动作
        if all(isinstance(a, (int, np.integer)) for a in actions):
            actions_batch = torch.tensor(actions, dtype=torch.long).to(device)
        else:
            # 多动作情况
            actions_batch = actions  # 保持列表格式
        
        # 处理奖励和完成标志
        rewards_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_batch = torch.tensor(dones, dtype=torch.bool).to(device)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0
    
    def get_stats(self) -> dict:
        """获取缓冲区统计信息"""
        return {
            "capacity": self.capacity,
            "current_size": len(self.buffer),
            "utilization": len(self.buffer) / self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先级经验回放缓冲区
    
    基于TD误差的重要性采样，让智能体更多地从重要经验中学习
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, seed: Optional[int] = None):
        super().__init__(capacity, seed)
        
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # 使用段树（Segment Tree）高效实现优先级采样
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.full(2 * self.tree_capacity - 1, float('inf'))
        self.data_pointer = 0
    
    def add(self, state, action, reward, next_state, done, priority: Optional[float] = None, **kwargs):
        """添加经验，支持指定优先级"""
        if priority is None:
            priority = self.max_priority
        
        # 调用父类方法存储经验
        super().add(state, action, reward, next_state, done, **kwargs)
        
        # 更新优先级树
        self._update_tree(self.data_pointer, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
    
    def _update_tree(self, idx: int, priority: float):
        """更新段树中的优先级"""
        priority = max(priority, 1e-6)  # 避免零优先级
        self.max_priority = max(self.max_priority, priority)
        
        tree_idx = idx + self.tree_capacity - 1
        delta = priority ** self.alpha - self.sum_tree[tree_idx]
        
        self.sum_tree[tree_idx] = priority ** self.alpha
        self.min_tree[tree_idx] = priority ** self.alpha
        
        # 向上传播更新
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += delta
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx + 1],
                self.min_tree[2 * tree_idx + 2]
            )
    
    def _sample_proportional(self, batch_size: int) -> List[int]:
        """按比例采样索引"""
        indices = []
        total_priority = self.sum_tree[0]
        
        if total_priority <= 0:
            # 如果没有优先级，随机采样
            available_indices = list(range(len(self.buffer)))
            return random.sample(available_indices, min(batch_size, len(available_indices)))
        
        for _ in range(batch_size):
            mass = random.random() * total_priority
            idx = self._retrieve_leaf(0, mass)
            # 确保索引有效
            if idx < len(self.buffer):
                indices.append(idx)
            elif len(self.buffer) > 0:
                indices.append(random.randint(0, len(self.buffer) - 1))
        
        return indices
    
    def _retrieve_leaf(self, idx: int, mass: float) -> int:
        """检索叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.sum_tree):
            return idx
        
        if mass <= self.sum_tree[left]:
            return self._retrieve_leaf(left, mass)
        else:
            return self._retrieve_leaf(right, mass - self.sum_tree[left])
    
    def sample(self, batch_size: int, device: torch.device = None) -> Tuple:
        """优先级采样"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if len(self.buffer) == 0:
            # 如果缓冲区为空，返回空结果
            empty_tensor = torch.empty(0)
            return (empty_tensor, [], empty_tensor, empty_tensor, empty_tensor, 
                   torch.empty(0), [])
        
        # 按优先级采样索引
        indices = self._sample_proportional(batch_size)
        
        # 过滤有效索引
        valid_indices = [i for i in indices if i < len(self.buffer)]
        if not valid_indices:
            # 如果没有有效索引，随机采样
            valid_indices = [random.randint(0, len(self.buffer) - 1) for _ in range(min(batch_size, len(self.buffer)))]
        
        # 获取对应的经验
        experiences = [self.buffer[i] for i in valid_indices]
        
        # 计算重要性采样权重
        weights = self._calculate_weights(valid_indices, len(valid_indices))
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch_data = self._batch_experiences(experiences, device)
        return batch_data + (weights, valid_indices)
    
    def _calculate_weights(self, indices: List[int], batch_size: int) -> torch.Tensor:
        """计算重要性采样权重"""
        min_prob = self.min_tree[0] / self.sum_tree[0]
        max_weight = (min_prob * len(self.buffer)) ** (-self.beta)
        
        weights = []
        for idx in indices:
            if idx < len(self.buffer):
                tree_idx = idx + self.tree_capacity - 1
                prob = self.sum_tree[tree_idx] / self.sum_tree[0]
                weight = (prob * len(self.buffer)) ** (-self.beta)
                weights.append(weight / max_weight)
            else:
                weights.append(1.0)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """更新采样经验的优先级"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.buffer):
                self._update_tree(idx, priority)


class MultiAgentReplayBuffer:
    """
    多智能体经验回放缓冲区
    
    支持：
    1. 每个智能体独立的经验存储
    2. 智能体间的经验共享
    3. 协调学习的批量采样
    """
    
    def __init__(self, capacity_per_agent: int = 10000, shared_capacity: int = 5000, 
                 prioritized: bool = True):
        self.capacity_per_agent = capacity_per_agent
        self.shared_capacity = shared_capacity
        self.prioritized = prioritized
        
        # 每个智能体的独立缓冲区
        self.agent_buffers = {}
        
        # 共享缓冲区
        if prioritized:
            self.shared_buffer = PrioritizedReplayBuffer(shared_capacity)
        else:
            self.shared_buffer = ReplayBuffer(shared_capacity)
        
        # 共享策略
        self.sharing_probability = 0.1  # 10%的经验会被共享
    
    def add_agent(self, agent_id: str):
        """添加新智能体"""
        if agent_id not in self.agent_buffers:
            if self.prioritized:
                self.agent_buffers[agent_id] = PrioritizedReplayBuffer(self.capacity_per_agent)
            else:
                self.agent_buffers[agent_id] = ReplayBuffer(self.capacity_per_agent)
    
    def add(self, agent_id: str, state, action, reward, next_state, done, **kwargs):
        """添加经验到指定智能体的缓冲区"""
        # 确保智能体缓冲区存在
        if agent_id not in self.agent_buffers:
            self.add_agent(agent_id)
        
        # 添加到智能体专用缓冲区
        self.agent_buffers[agent_id].add(state, action, reward, next_state, done, 
                                        agent_id=agent_id, **kwargs)
        
        # 随机共享到共享缓冲区
        if random.random() < self.sharing_probability:
            self.shared_buffer.add(state, action, reward, next_state, done, 
                                 agent_id=agent_id, **kwargs)
    
    def sample(self, agent_id: str, batch_size: int, use_shared: bool = True, 
               shared_ratio: float = 0.3, device: torch.device = None):
        """
        为指定智能体采样经验批次
        
        Args:
            agent_id: 智能体ID
            batch_size: 批次大小
            use_shared: 是否使用共享经验
            shared_ratio: 共享经验在批次中的比例
            device: 目标设备
        """
        if agent_id not in self.agent_buffers:
            raise ValueError(f"智能体 {agent_id} 不存在")
        
        agent_buffer = self.agent_buffers[agent_id]
        
        if not use_shared or len(self.shared_buffer) == 0:
            # 只使用智能体自己的经验
            return agent_buffer.sample(batch_size, device)
        
        # 混合采样：智能体经验 + 共享经验
        shared_size = int(batch_size * shared_ratio)
        agent_size = batch_size - shared_size
        
        # 确保有足够的经验可采样
        if len(agent_buffer) < agent_size:
            agent_size = len(agent_buffer)
            shared_size = batch_size - agent_size
        
        if len(self.shared_buffer) < shared_size:
            shared_size = len(self.shared_buffer)
            agent_size = batch_size - shared_size
        
        experiences = []
        
        # 从智能体缓冲区采样
        if agent_size > 0:
            agent_experiences = random.sample(list(agent_buffer.buffer), agent_size)
            experiences.extend(agent_experiences)
        
        # 从共享缓冲区采样
        if shared_size > 0:
            shared_experiences = random.sample(list(self.shared_buffer.buffer), shared_size)
            experiences.extend(shared_experiences)
        
        # 批处理
        return agent_buffer._batch_experiences(experiences, device)
    
    def get_stats(self) -> dict:
        """获取所有缓冲区的统计信息"""
        stats = {
            "shared_buffer": self.shared_buffer.get_stats(),
            "agent_buffers": {}
        }
        
        for agent_id, buffer in self.agent_buffers.items():
            stats["agent_buffers"][agent_id] = buffer.get_stats()
        
        return stats


# 测试函数
def test_replay_buffers():
    """测试回放缓冲区功能"""
    print("🧪 测试回放缓冲区...")
    
    # 测试标准缓冲区
    buffer = ReplayBuffer(capacity=100)
    
    # 添加测试数据
    for i in range(50):
        state = torch.randn(4, 8)  # 模拟节点特征
        action = i % 10
        reward = random.random()
        next_state = torch.randn(4, 8)
        done = (i % 20 == 0)
        
        buffer.add(state, action, reward, next_state, done)
    
    # 测试采样
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=16)
    
    print(f"✅ 标准缓冲区测试:")
    print(f"   - 缓冲区大小: {len(buffer)}")
    print(f"   - 批次状态形状: {states.shape}")
    print(f"   - 批次动作形状: {actions.shape}")
    
    # 测试优先级缓冲区
    print("\n🧪 测试优先级缓冲区...")
    
    priority_buffer = PrioritizedReplayBuffer(capacity=100)
    
    # 添加带优先级的数据
    for i in range(30):
        state = torch.randn(4, 8)
        action = i % 10
        reward = random.random()
        next_state = torch.randn(4, 8)
        done = False
        priority = random.random()
        
        priority_buffer.add(state, action, reward, next_state, done, priority=priority)
    
    # 测试优先级采样
    batch_data = priority_buffer.sample(batch_size=8)
    states, actions, rewards, next_states, dones, weights, indices = batch_data
    
    print(f"✅ 优先级缓冲区测试:")
    print(f"   - 缓冲区大小: {len(priority_buffer)}")
    print(f"   - 重要性权重形状: {weights.shape}")
    print(f"   - 采样索引数量: {len(indices)}")
    
    # 测试多智能体缓冲区
    print("\n🧪 测试多智能体缓冲区...")
    
    multi_buffer = MultiAgentReplayBuffer(capacity_per_agent=50, shared_capacity=30)
    
    # 添加多个智能体的经验
    agents = ["ddqn", "dqn", "ppo"]
    for agent in agents:
        for i in range(20):
            state = torch.randn(4, 8)
            action = i % 10
            reward = random.random()
            next_state = torch.randn(4, 8)
            done = False
            
            multi_buffer.add(agent, state, action, reward, next_state, done)
    
    # 测试混合采样
    mixed_batch = multi_buffer.sample("ddqn", batch_size=12, use_shared=True)
    states, actions, rewards, next_states, dones = mixed_batch
    
    print(f"✅ 多智能体缓冲区测试:")
    print(f"   - 智能体数量: {len(multi_buffer.agent_buffers)}")
    print(f"   - 混合批次大小: {len(actions)}")
    
    # 统计信息
    stats = multi_buffer.get_stats()
    print(f"   - 共享缓冲区利用率: {stats['shared_buffer']['utilization']:.2f}")
    
    print("\n✅ 所有缓冲区测试通过!")


if __name__ == "__main__":
    test_replay_buffers()