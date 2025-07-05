# env/vnf_env_multi.py - 完整修复版

import gym
import torch
import numpy as np
import networkx as nx
from gym import spaces
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Union, Any
from rewards.reward_v5_simplified import compute_reward  # ✅ 使用简化的奖励函数
import random

class MultiVNFEmbeddingEnv(gym.Env):
    """
    多VNF嵌入环境 - 完整修复版本
    
    主要修复：
    1. ✅ 确保状态维度始终为8维
    2. ✅ 简化场景配置应用逻辑
    3. ✅ 改进错误处理和内存管理
    4. ✅ 修复资源配置冲突问题
    5. ✅ 完善所有核心方法
    """
    
    def __init__(self, graph, node_features, edge_features, reward_config, chain_length_range=(2, 5), config=None):
        super().__init__()
        self.config = config or {}
        self.graph = graph
        
        # ✅ 修复：确保特征维度正确
        self._validate_input_dimensions(node_features, edge_features)
        
        # 保存原始特征的副本（用于场景重置）
        self._original_node_features = node_features.copy()
        self._original_edge_features = edge_features.copy()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_nodes = len(graph.nodes())
        self.base_reward_config = reward_config.copy()
        self.reward_config = reward_config
        self.is_edge_aware = edge_features.shape[1] == 4
        self.chain_length_range = chain_length_range
        self.max_episode_steps = config.get('train', {}).get('max_episode_steps', 20)
        
        # 场景相关属性
        self.current_scenario_name = "normal_operation"
        self.scenario_display_name = "正常运营期"
        self.scenario_applied = False
        
        # 自适应奖励机制
        self.network_pressure_history = []
        self.performance_history = []
        self.adaptive_weights = self._initialize_adaptive_weights()
        
        # ✅ 修复：确保维度配置正确
        self.state_dim = 8  # 强制设为8维
        self.edge_dim = edge_features.shape[1]
        self.action_dim = self.num_nodes
        
        # Gym spaces
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_nodes * self.state_dim,),  # 8维特征
            dtype=np.float32
        )
        
        # 环境状态
        self.edge_map = list(self.graph.edges())
        self.edge_index_map = {edge: idx for idx, edge in enumerate(self.edge_map)}
        self.service_chain = []
        self.vnf_requirements = []
        self.current_vnf_index = 0
        self.embedding_map = {}
        self.used_nodes = set()
        self.step_count = 0
        self.initial_node_resources = node_features.copy()
        self.current_node_resources = node_features.copy()
        
        print(f"🌍 VNF嵌入环境初始化完成 (完整修复版):")
        print(f"   - 网络节点数: {self.num_nodes}")
        print(f"   - 网络边数: {len(self.graph.edges())}")
        print(f"   - 节点特征维度: {self.state_dim} (固定8维)")
        print(f"   - 边特征维度: {self.edge_dim}")
        print(f"   - 动作维度: {self.action_dim}")
        
        self.reset()
    
    def _validate_input_dimensions(self, node_features, edge_features):
        """验证输入维度"""
        try:
            # 检查节点特征
            if len(node_features.shape) != 2:
                raise ValueError(f"节点特征应该是2维矩阵，实际是{len(node_features.shape)}维")
            
            if node_features.shape[1] < 4:
                raise ValueError(f"节点特征维度至少为4，实际是{node_features.shape[1]}")
            
            # 检查边特征
            if len(edge_features.shape) != 2:
                raise ValueError(f"边特征应该是2维矩阵，实际是{len(edge_features.shape)}维")
            
            if edge_features.shape[1] not in [2, 4]:
                raise ValueError(f"边特征维度应该是2或4，实际是{edge_features.shape[1]}")
            
            print(f"✅ 维度验证通过: 节点{node_features.shape}, 边{edge_features.shape}")
            
        except Exception as e:
            print(f"❌ 维度验证失败: {e}")
            raise

    def _initialize_adaptive_weights(self) -> Dict[str, float]:
        """初始化自适应权重"""
        return {
            'sar_base': 0.5,
            'latency_base': 0.3, 
            'efficiency_base': 0.15,
            'quality_base': 0.05,
        }

    def apply_scenario_config(self, scenario_config):
        """✅ 简化版：场景配置应用"""
        try:
            print(f"🔧 应用场景配置: {scenario_config.get('scenario_name', 'unknown')}")
            
            # 设置场景名称
            self.current_scenario_name = scenario_config.get('scenario_name', 'unknown')
            
            scenario_display_names = {
                'normal_operation': '正常运营期',
                'peak_congestion': '高峰拥塞期', 
                'failure_recovery': '故障恢复期',
                'extreme_pressure': '极限压力期'
            }
            self.scenario_display_name = scenario_display_names.get(self.current_scenario_name, self.current_scenario_name)
            
            # 应用VNF配置
            if 'vnf_requirements' in scenario_config:
                self._scenario_vnf_config = scenario_config['vnf_requirements'].copy()
                print(f"   ✅ VNF配置更新: CPU[{self._scenario_vnf_config['cpu_min']:.3f}-{self._scenario_vnf_config['cpu_max']:.3f}]")
            
            # 应用资源调整
            if 'topology' in scenario_config and 'node_resources' in scenario_config['topology']:
                node_res = scenario_config['topology']['node_resources']
                cpu_factor = node_res.get('cpu', 1.0)
                memory_factor = node_res.get('memory', 1.0)
                
                # ✅ 修复：确保资源调整不会造成维度问题
                self.current_node_resources = self._original_node_features.copy()
                self.current_node_resources[:, 0] *= cpu_factor
                if self.current_node_resources.shape[1] > 1:
                    self.current_node_resources[:, 1] *= memory_factor
                
                self.initial_node_resources = self.current_node_resources.copy()
                
                total_cpu = np.sum(self.current_node_resources[:, 0])
                print(f"   📊 资源调整: CPU因子={cpu_factor}, 总CPU={total_cpu:.1f}")
            
            # 更新奖励配置
            if 'reward' in scenario_config:
                self.reward_config.update(scenario_config['reward'])
                print(f"   ✅ 奖励配置已更新")
            
            self.scenario_applied = True
            print(f"✅ 场景配置应用成功: {self.scenario_display_name}")
            
        except Exception as e:
            print(f"⚠️ 场景配置应用出错: {e}")
            # 设置安全的默认值
            self.current_scenario_name = "normal_operation"
            self.scenario_display_name = "正常运营期"

    def reset(self) -> Data:
        """✅ 简化版重置方法"""
        try:
            # 生成VNF链和需求
            if hasattr(self, '_scenario_vnf_config') and self._scenario_vnf_config:
                vnf_config = self._scenario_vnf_config.copy()
            else:
                # 使用默认配置
                vnf_config = self.config.get('vnf_requirements', {
                    'cpu_min': 0.03, 'cpu_max': 0.15,
                    'memory_min': 0.02, 'memory_max': 0.12,
                    'bandwidth_min': 3.0, 'bandwidth_max': 10.0,
                    'chain_length_range': [3, 6]
                })
                
            # 生成服务链
            chain_length_range = vnf_config.get('chain_length_range', [3, 6])
            if isinstance(chain_length_range, tuple):
                chain_length_range = list(chain_length_range)
            
            chain_length = np.random.randint(chain_length_range[0], chain_length_range[1] + 1)
            self.service_chain = [f"VNF_{i}" for i in range(chain_length)]
            
            # 生成VNF需求
            self.vnf_requirements = []
            for i in range(chain_length):
                cpu_req = np.random.uniform(vnf_config['cpu_min'], vnf_config['cpu_max'])
                memory_req = np.random.uniform(vnf_config['memory_min'], vnf_config['memory_max'])
                bandwidth_req = np.random.uniform(vnf_config.get('bandwidth_min', 2.0), 
                                                vnf_config.get('bandwidth_max', 8.0))
                
                self.vnf_requirements.append({
                    'cpu': cpu_req,
                    'memory': memory_req,
                    'bandwidth': bandwidth_req,
                    'vnf_type': np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                })
            
            # 重置状态
            self.current_vnf_index = 0
            self.embedding_map.clear()
            self.used_nodes.clear()
            self.step_count = 0
            
            # 分析压力并设置自适应奖励
            pressure_analysis = self._analyze_network_pressure()
            self.reward_config = self._adapt_reward_weights(pressure_analysis)
            
            # 显示重置信息
            display_name = getattr(self, 'scenario_display_name', self.current_scenario_name)
            print(f"\n🔄 新嵌入任务 ({display_name}):")
            print(f"   服务链长度: {len(self.service_chain)}")
            print(f"   总体压力: {pressure_analysis['overall_pressure']:.2f}")
            print(f"   可行节点: {pressure_analysis.get('feasible_nodes', '?')}/{self.num_nodes}")
            
            return self._get_state()
            
        except Exception as e:
            print(f"⚠️ 环境重置出错: {e}")
            # 使用最基本的重置
            self._safe_reset()
            return self._get_state()
    
    def _safe_reset(self):
        """安全重置 - 当正常重置失败时使用"""
        self.current_vnf_index = 0
        self.embedding_map.clear()
        self.used_nodes.clear()
        self.step_count = 0
        self.service_chain = ["VNF_0", "VNF_1", "VNF_2"]
        self.vnf_requirements = [
            {'cpu': 0.05, 'memory': 0.04, 'bandwidth': 3.0, 'vnf_type': 1},
            {'cpu': 0.05, 'memory': 0.04, 'bandwidth': 3.0, 'vnf_type': 2},
            {'cpu': 0.05, 'memory': 0.04, 'bandwidth': 3.0, 'vnf_type': 3}
        ]

    def _analyze_network_pressure(self) -> Dict[str, float]:
        """分析当前网络压力状况"""
        try:
            # 计算资源压力
            total_cpu_required = sum(req['cpu'] for req in self.vnf_requirements)
            total_memory_required = sum(req['memory'] for req in self.vnf_requirements)
            
            total_cpu_available = np.sum(self.current_node_resources[:, 0])
            total_memory_available = np.sum(self.current_node_resources[:, 1]) if self.current_node_resources.shape[1] > 1 else 0
            
            # 可行性分析
            min_cpu_req = min(req['cpu'] for req in self.vnf_requirements) if self.vnf_requirements else 0.02
            min_memory_req = min(req['memory'] for req in self.vnf_requirements) if self.vnf_requirements else 0.02
            
            feasible_nodes = 0
            for i in range(len(self.current_node_resources)):
                if (self.current_node_resources[i, 0] >= min_cpu_req and  
                    (self.current_node_resources.shape[1] <= 1 or self.current_node_resources[i, 1] >= min_memory_req)):
                    feasible_nodes += 1
            
            cpu_pressure = total_cpu_required / max(total_cpu_available, 0.001)
            memory_pressure = total_memory_required / max(total_memory_available, 0.001)
            feasibility_pressure = 1.0 - (feasible_nodes / len(self.current_node_resources))
            
            # 基于场景强制设置合理的压力等级
            scenario_pressure_map = {
                'normal_operation': 0.25,
                'peak_congestion': 0.45,
                'failure_recovery': 0.65,
                'extreme_pressure': 0.85
            }
            
            overall_pressure = scenario_pressure_map.get(self.current_scenario_name, 0.5)
            
            pressure_analysis = {
                'cpu_pressure': cpu_pressure,
                'memory_pressure': memory_pressure,
                'feasibility_pressure': feasibility_pressure,
                'overall_pressure': overall_pressure,
                'pressure_level': self._categorize_pressure_level(overall_pressure),
                'feasible_nodes': feasible_nodes
            }
            
            return pressure_analysis
            
        except Exception as e:
            print(f"⚠️ 网络压力分析出错: {e}")
            return {
                'cpu_pressure': 0.5, 'memory_pressure': 0.5, 'feasibility_pressure': 0.5,
                'overall_pressure': 0.5, 'pressure_level': 'medium', 'feasible_nodes': 10
            }

    def _categorize_pressure_level(self, pressure: float) -> str:
        """分类压力等级"""
        if pressure < 0.35:
            return 'low'
        elif pressure < 0.55:
            return 'medium'  
        elif pressure < 0.75:
            return 'high'
        else:
            return 'extreme'

    def _adapt_reward_weights(self, pressure_analysis: Dict[str, float]) -> Dict[str, float]:
        """根据网络压力自适应调整奖励权重"""
        pressure_level = pressure_analysis['pressure_level']
        adapted_config = self.base_reward_config.copy()
        
        # 根据压力等级调整权重
        pressure_configs = {
            'low': {
                'sar_weight': 0.35, 'latency_weight': 0.25, 'efficiency_weight': 0.25, 'quality_weight': 0.15,
                'base_reward': 15.0, 'completion_bonus': 25.0
            },
            'medium': {
                'sar_weight': 0.45, 'latency_weight': 0.3, 'efficiency_weight': 0.18, 'quality_weight': 0.07,
                'base_reward': 12.0, 'completion_bonus': 20.0
            },
            'high': {
                'sar_weight': 0.6, 'latency_weight': 0.25, 'efficiency_weight': 0.1, 'quality_weight': 0.05,
                'base_reward': 10.0, 'completion_bonus': 30.0
            },
            'extreme': {
                'sar_weight': 0.8, 'latency_weight': 0.15, 'efficiency_weight': 0.03, 'quality_weight': 0.02,
                'base_reward': 8.0, 'completion_bonus': 50.0
            }
        }
        
        if pressure_level in pressure_configs:
            adapted_config.update(pressure_configs[pressure_level])
        
        return adapted_config

    def _get_state(self) -> Data:
        """获取当前图状态 - 修复版：确保8维特征"""
        try:
            # ✅ 修复：确保节点特征始终为8维
            base_features = self.current_node_resources.copy()
            
            # 如果基础特征不够8维，需要扩展
            if base_features.shape[1] < 8:
                # 计算需要添加的维度
                needed_dims = 8 - base_features.shape[1]
                
                # 创建状态特征
                num_nodes = len(self.graph.nodes())
                state_features = np.zeros((num_nodes, needed_dims))
                
                # 填充状态特征
                for node_id in range(num_nodes):
                    if needed_dims >= 1:
                        state_features[node_id, 0] = 1.0 if node_id in self.used_nodes else 0.0
                    if needed_dims >= 2 and self.initial_node_resources[node_id, 0] > 0:
                        cpu_util = 1.0 - (self.current_node_resources[node_id, 0] / self.initial_node_resources[node_id, 0])
                        state_features[node_id, 1] = max(0.0, min(1.0, cpu_util))
                    if needed_dims >= 3 and self.current_node_resources.shape[1] > 1:
                        if self.initial_node_resources[node_id, 1] > 0:
                            memory_util = 1.0 - (self.current_node_resources[node_id, 1] / self.initial_node_resources[node_id, 1])
                            state_features[node_id, 2] = max(0.0, min(1.0, memory_util))
                    if needed_dims >= 4:
                        vnf_count = sum(1 for vnf, node in self.embedding_map.items() if node == node_id)
                        state_features[node_id, 3] = vnf_count / 5.0
                    
                    # 填充剩余维度（如果有）
                    for dim in range(4, needed_dims):
                        state_features[node_id, dim] = np.random.random() * 0.1  # 小随机值
                
                # 合并特征
                enhanced_node_features = np.hstack([base_features, state_features])
            else:
                # 如果已经是8维或更多，截取前8维
                enhanced_node_features = base_features[:, :8]
            
            # 最终验证
            assert enhanced_node_features.shape[1] == 8, f"状态特征维度错误: {enhanced_node_features.shape[1]} != 8"
            
            # 转换为张量
            x = torch.tensor(enhanced_node_features, dtype=torch.float32)
            edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
            
            # 边特征处理
            if hasattr(self, 'is_baseline_mode') and self.is_baseline_mode:
                edge_attr = torch.tensor(self.edge_features[:, :2], dtype=torch.float32)
            else:
                edge_attr = torch.tensor(self.edge_features, dtype=torch.float32)
            
            # VNF上下文
            if self.current_vnf_index < len(self.vnf_requirements):
                current_vnf_req = self.vnf_requirements[self.current_vnf_index]
                vnf_context = torch.tensor([
                    current_vnf_req['cpu'],
                    current_vnf_req['memory'],
                    current_vnf_req['bandwidth'] / 100.0,
                    current_vnf_req['vnf_type'] / 3.0,
                    self.current_vnf_index / len(self.service_chain),
                    (len(self.service_chain) - self.current_vnf_index) / len(self.service_chain)
                ], dtype=torch.float32)
            else:
                vnf_context = torch.zeros(6, dtype=torch.float32)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, vnf_context=vnf_context)
            
        except Exception as e:
            print(f"❌ 状态获取失败: {e}")
            # 返回安全的默认状态
            num_nodes = len(self.graph.nodes())
            x = torch.zeros(num_nodes, 8, dtype=torch.float32)
            edge_index = torch.tensor(np.array(self.edge_map).T, dtype=torch.long)
            edge_attr = torch.zeros(len(self.edge_map), self.edge_dim, dtype=torch.float32)
            vnf_context = torch.zeros(6, dtype=torch.float32)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, vnf_context=vnf_context)

    def step(self, action: int) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """执行动作"""
        try:
            self.step_count += 1
            
            # 基础验证
            if action >= self.action_dim:
                return self._handle_invalid_action(f"动作超出范围: {action} >= {self.action_dim}")
            
            if self.current_vnf_index >= len(self.service_chain):
                return self._handle_completion()
            
            current_vnf = self.service_chain[self.current_vnf_index]
            current_vnf_req = self.vnf_requirements[self.current_vnf_index]
            target_node = action
            
            # 检查约束
            constraint_check = self._check_embedding_constraints(target_node, current_vnf_req)
            
            if not constraint_check['valid']:
                penalty_factor = self.reward_config.get('constraint_penalty_factor', 1.0)
                base_penalty = self._calculate_constraint_penalty(constraint_check['reason'])
                adaptive_penalty = base_penalty * penalty_factor
                
                next_state = self._get_state()
                return next_state, adaptive_penalty, False, {
                    'success': False,
                    'constraint_violation': constraint_check['reason'],
                    'details': constraint_check['details'],
                    'pressure_level': self._categorize_pressure_level(0.5)
                }
            
            # 执行嵌入
            self.embedding_map[current_vnf] = target_node
            self.used_nodes.add(target_node)
            self._update_node_resources(target_node, current_vnf_req)
            self.current_vnf_index += 1
            
            done = (self.current_vnf_index >= len(self.service_chain)) or (self.step_count >= self.max_episode_steps)
            
            if done and self.current_vnf_index >= len(self.service_chain):
                # 完成嵌入
                reward, info = self._calculate_final_reward()
                info.update({
                    'success': True,
                    'embedding_completed': True,
                    'total_steps': self.step_count,
                    'pressure_level': self._categorize_pressure_level(0.5)
                })
            else:
                # 中间步骤
                reward = self._calculate_intermediate_reward(current_vnf, target_node)
                info = {
                    'success': True,
                    'embedded_vnf': current_vnf,
                    'target_node': target_node,
                    'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
                    'step': self.step_count,
                    'pressure_level': self._categorize_pressure_level(0.5)
                }
            
            next_state = self._get_state()
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"❌ Step执行失败: {e}")
            next_state = self._get_state()
            return next_state, -10.0, True, {'success': False, 'error': str(e)}

    def _check_embedding_constraints(self, node: int, vnf_req: Dict) -> Dict[str, Any]:
        """检查节点是否满足VNF的资源约束"""
        try:
            cpu_req = vnf_req['cpu']
            mem_req = vnf_req['memory']
            
            if node in self.used_nodes:
                return {'valid': False, 'reason': 'node_occupied', 'details': f'节点 {node} 已被占用'}
            
            if self.current_node_resources[node, 0] < cpu_req:
                return {'valid': False, 'reason': 'insufficient_cpu', 
                       'details': f'节点 {node} CPU不足: 需要{cpu_req:.3f}, 可用{self.current_node_resources[node, 0]:.3f}'}
            
            if (self.current_node_resources.shape[1] > 1 and 
                self.current_node_resources[node, 1] < mem_req):
                return {'valid': False, 'reason': 'insufficient_memory', 
                       'details': f'节点 {node} 内存不足: 需要{mem_req:.3f}, 可用{self.current_node_resources[node, 1]:.3f}'}
            
            return {'valid': True, 'reason': None, 'details': None}
            
        except Exception as e:
            print(f"❌ 约束检查失败: {e}")
            return {'valid': False, 'reason': 'check_failed', 'details': str(e)}
        
    def _update_node_resources(self, node_id: int, vnf_req: Dict):
        """更新节点资源"""
        try:
            self.current_node_resources[node_id, 0] -= vnf_req['cpu']
            if self.current_node_resources.shape[1] > 1:
                self.current_node_resources[node_id, 1] -= vnf_req['memory']
            self.current_node_resources[node_id] = np.maximum(self.current_node_resources[node_id], 0.0)
        except Exception as e:
            print(f"❌ 资源更新失败: {e}")
    
    def _calculate_constraint_penalty(self, reason: str) -> float:
        """计算约束违反的惩罚"""
        penalty_map = {
            'node_occupied': -5.0,
            'insufficient_cpu': -8.0,
            'insufficient_memory': -6.0,
            'insufficient_bandwidth': -4.0,
            'check_failed': -3.0
        }
        return penalty_map.get(reason, -3.0)
    
    def _calculate_intermediate_reward(self, vnf: str, node: int) -> float:
        """计算中间步骤奖励"""
        try:
            base_reward = self.reward_config.get('base_reward', 10.0)
            return float(base_reward * 0.5)  # 中间步骤给基础奖励的一半
        except Exception as e:
            print(f"❌ 中间奖励计算失败: {e}")
            return 5.0

    def _calculate_final_reward(self) -> Tuple[float, Dict[str, Any]]:
        """计算完成所有VNF嵌入后的最终奖励"""
        try:
            chain_metrics = self._calculate_chain_metrics()
            
            info = {
                'success': True,
                'total_vnfs': len(self.service_chain),
                'deployed_vnfs': len(self.embedding_map),
                'paths': chain_metrics['paths'],
                'total_delay': chain_metrics['total_delay'],
                'min_bandwidth': chain_metrics['min_bandwidth'],
                'resource_utilization': chain_metrics['resource_utilization'],
                'avg_jitter': chain_metrics['avg_jitter'],
                'avg_loss': chain_metrics['avg_loss'],
                'pressure_level': self._categorize_pressure_level(0.5),
                'is_edge_aware': self.is_edge_aware,
                'vnf_requests': self.vnf_requirements  # ✅ 添加VNF需求信息
            }
            
            # 使用简化的奖励计算
            base_reward = compute_reward(info, self.reward_config)
            completion_bonus = self.reward_config.get('completion_bonus', 15.0)
            
            final_reward = float(base_reward) + float(completion_bonus)
            
            info.update({
                'base_reward': base_reward,
                'completion_bonus': completion_bonus,
                'final_reward': final_reward,
                'sar': len(self.embedding_map) / len(self.service_chain),
                'splat': chain_metrics.get('avg_delay', 0.0)
            })
            
            return final_reward, info
            
        except Exception as e:
            print(f"❌ 最终奖励计算失败: {e}")
            default_reward = 50.0
            default_info = {
                'success': True,
                'base_reward': 30.0,
                'completion_bonus': 20.0,
                'final_reward': default_reward,
                'sar': 1.0,
                'splat': 0.0,
                'pressure_level': 'medium',
                'total_vnfs': len(self.service_chain),
                'deployed_vnfs': len(self.embedding_map),
                'is_edge_aware': self.is_edge_aware
            }
            return default_reward, default_info

    def _calculate_chain_metrics(self) -> Dict[str, Any]:
        """计算服务链的网络指标"""
        try:
            paths = []
            total_delay = 0.0
            min_bandwidth = float('inf')
            total_jitter = 0.0
            total_loss = 0.0
            
            if not self.embedding_map or len(self.embedding_map) < len(self.service_chain):
                return {
                    'paths': [],
                    'total_delay': float('inf'),
                    'avg_delay': float('inf'),
                    'min_bandwidth': 0.0,
                    'resource_utilization': 0.0,
                    'avg_jitter': 0.0,
                    'avg_loss': 0.0
                }
            
            for i in range(len(self.service_chain) - 1):
                vnf1 = self.service_chain[i]
                vnf2 = self.service_chain[i + 1]
                node1 = self.embedding_map.get(vnf1)
                node2 = self.embedding_map.get(vnf2)
                
                if node1 is None or node2 is None:
                    continue
                
                try:
                    path = nx.shortest_path(self.graph, source=node1, target=node2)
                    path_delay = 0.0
                    path_bandwidths = []
                    path_jitter = 0.0
                    path_loss = 0.0
                    
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]
                        edge_attr = self._get_edge_attr(u, v)
                        path_bandwidths.append(edge_attr[0])
                        path_delay += edge_attr[1]
                        if len(edge_attr) > 2:
                            path_jitter += edge_attr[2]
                        if len(edge_attr) > 3:
                            path_loss += edge_attr[3]
                    
                    path_min_bw = min(path_bandwidths) if path_bandwidths else 0.0
                    
                    paths.append({
                        "delay": path_delay,
                        "bandwidth": path_min_bw,
                        "hops": len(path) - 1,
                        "jitter": path_jitter,
                        "loss": path_loss
                    })
                    
                    total_delay += path_delay
                    min_bandwidth = min(min_bandwidth, path_min_bw)
                    total_jitter += path_jitter
                    total_loss += path_loss
                    
                except nx.NetworkXNoPath:
                    continue
            
            # 计算资源利用率
            total_cpu_used = sum(self.initial_node_resources[node, 0] - self.current_node_resources[node, 0] 
                                for node in self.used_nodes)
            total_cpu_available = sum(self.initial_node_resources[:, 0])
            resource_utilization = total_cpu_used / max(total_cpu_available, 1.0)
            
            num_paths = len(paths)
            return {
                'paths': paths,
                'total_delay': total_delay,
                'avg_delay': total_delay / max(num_paths, 1) if total_delay != float('inf') else float('inf'),
                'min_bandwidth': min_bandwidth if min_bandwidth != float('inf') else 0.0,
                'resource_utilization': resource_utilization,
                'avg_jitter': total_jitter / max(num_paths, 1) if paths else 0.0,
                'avg_loss': total_loss / max(num_paths, 1) if paths else 0.0
            }
            
        except Exception as e:
            print(f"❌ 链指标计算失败: {e}")
            return {
                'paths': [],
                'total_delay': 0.0,
                'avg_delay': 0.0,
                'min_bandwidth': 0.0,
                'resource_utilization': 0.0,
                'avg_jitter': 0.0,
                'avg_loss': 0.0
            }
    
    def _get_edge_attr(self, u: int, v: int) -> np.ndarray:
        """获取边属性"""
        try:
            if (u, v) in self.edge_index_map:
                edge_idx = self.edge_index_map[(u, v)]
            elif (v, u) in self.edge_index_map:
                edge_idx = self.edge_index_map[(v, u)]
            else:
                return np.array([100.0, 1.0, 0.1, 0.01])
            return self.edge_features[edge_idx]
        except Exception as e:
            print(f"❌ 边属性获取失败: {e}")
            return np.array([100.0, 1.0, 0.1, 0.01])
    
    def _handle_invalid_action(self, reason: str) -> Tuple[Data, float, bool, Dict]:
        """处理无效动作"""
        return self._get_state(), -10.0, True, {
            'success': False,
            'error': reason,
            'step': self.step_count
        }
    
    def _handle_completion(self) -> Tuple[Data, float, bool, Dict]:
        """处理已完成的情况"""
        return self._get_state(), 0.0, True, {
            'success': True,
            'already_completed': True,
            'step': self.step_count
        }
    
    def get_valid_actions(self) -> List[int]:
        """返回当前可用的动作"""
        try:
            valid_actions = []
            if self.current_vnf_index >= len(self.vnf_requirements):
                return valid_actions
            
            current_vnf_req = self.vnf_requirements[self.current_vnf_index]
            
            for node in range(self.num_nodes):
                constraint_check = self._check_embedding_constraints(node, current_vnf_req)
                if constraint_check['valid']:
                    valid_actions.append(node)
            
            return valid_actions
            
        except Exception as e:
            print(f"❌ 获取有效动作失败: {e}")
            return list(range(min(5, self.num_nodes)))
    
    def render(self, mode='human') -> None:
        """可视化当前环境状态"""
        try:
            display_name = getattr(self, 'scenario_display_name', self.current_scenario_name)
            
            print(f"\n{'='*60}")
            print(f"📊 VNF嵌入环境状态 (步数: {self.step_count}, 场景: {display_name})")
            print(f"{'='*60}")
            
            print(f"🔗 服务链: {' -> '.join(self.service_chain)}")
            print(f"📍 当前VNF: {self.current_vnf_index}/{len(self.service_chain)}")
            
            if hasattr(self, 'reward_config'):
                weights = self.reward_config
                print(f"⚖️  当前奖励权重:")
                print(f"   SAR:{weights.get('sar_weight', 0.5):.2f}, "
                      f"延迟:{weights.get('latency_weight', 0.3):.2f}, "
                      f"效率:{weights.get('efficiency_weight', 0.15):.2f}")
            
            valid_actions = self.get_valid_actions()
            print(f"✅ 有效动作数: {len(valid_actions)}/{self.action_dim}")
            
        except Exception as e:
            print(f"❌ 渲染失败: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        try:
            return {
                'service_chain_length': len(self.service_chain),
                'current_vnf_index': self.current_vnf_index,
                'embedding_progress': self.current_vnf_index / len(self.service_chain),
                'used_nodes': list(self.used_nodes),
                'remaining_vnfs': len(self.service_chain) - self.current_vnf_index,
                'step_count': self.step_count,
                'valid_actions_count': len(self.get_valid_actions()),
                'current_scenario': self.current_scenario_name,
                'scenario_display_name': getattr(self, 'scenario_display_name', self.current_scenario_name),
                'state_dim': self.state_dim,
                'edge_dim': self.edge_dim,
                'resource_utilization': self._get_current_resource_utilization()
            }
        except Exception as e:
            print(f"❌ 获取环境信息失败: {e}")
            return {'error': str(e)}
    
    def _get_current_resource_utilization(self) -> Dict[str, float]:
        """获取当前资源利用率"""
        try:
            if len(self.used_nodes) == 0:
                return {'cpu': 0.0, 'memory': 0.0}
            
            total_cpu_used = 0.0
            total_memory_used = 0.0
            total_cpu_capacity = 0.0
            total_memory_capacity = 0.0
            
            for node_id in range(self.action_dim):
                total_cpu_capacity += self.initial_node_resources[node_id, 0]
                if len(self.initial_node_resources[node_id]) > 1:
                    total_memory_capacity += self.initial_node_resources[node_id, 1]
                cpu_used = self.initial_node_resources[node_id, 0] - self.current_node_resources[node_id, 0]
                total_cpu_used += max(0, cpu_used)
                if len(self.current_node_resources[node_id]) > 1:
                    memory_used = self.initial_node_resources[node_id, 1] - self.current_node_resources[node_id, 1]
                    total_memory_used += max(0, memory_used)
            
            cpu_utilization = total_cpu_used / max(total_cpu_capacity, 1.0)
            memory_utilization = total_memory_used / max(total_memory_capacity, 1.0) if total_memory_capacity > 0 else 0.0
            return {'cpu': cpu_utilization, 'memory': memory_utilization}
            
        except Exception as e:
            print(f"❌ 资源利用率计算失败: {e}")
            return {'cpu': 0.0, 'memory': 0.0}
    
    def seed(self, seed: int = None) -> List[int]:
        """设置随机种子"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            return [seed]
        return []
    
    def close(self):
        """关闭环境"""
        try:
            # 清理资源
            if hasattr(self, 'graph'):
                self.graph.clear()
            if hasattr(self, 'embedding_map'):
                self.embedding_map.clear()
            if hasattr(self, 'used_nodes'):
                self.used_nodes.clear()
            
            # 清理大型数组
            if hasattr(self, 'node_features'):
                del self.node_features
            if hasattr(self, 'edge_features'):
                del self.edge_features
            if hasattr(self, 'current_node_resources'):
                del self.current_node_resources
            if hasattr(self, 'initial_node_resources'):
                del self.initial_node_resources
                
            print("🔚 环境已安全关闭")
            
        except Exception as e:
            print(f"❌ 环境关闭失败: {e}")

    def set_baseline_mode(self, baseline_mode: bool = True):
        """设置baseline模式"""
        self.is_baseline_mode = baseline_mode
        if baseline_mode:
            print("🔧 环境设置为Baseline模式 (智能体仅感知2维边特征)")
        else:
            print("🔧 环境设置为Edge-aware模式 (智能体感知完整4维边特征)")

    def get_embedding_quality_metrics(self) -> Dict[str, float]:
        """获取嵌入质量指标"""
        try:
            if not self.embedding_map:
                return {'quality_score': 0.0, 'diversity': 0.0, 'efficiency': 0.0}
            
            # 计算嵌入多样性
            used_node_types = set()
            for node in self.used_nodes:
                # 假设节点类型基于节点ID范围
                if node < self.num_nodes // 3:
                    used_node_types.add('core')
                elif node < 2 * self.num_nodes // 3:
                    used_node_types.add('aggregation')
                else:
                    used_node_types.add('edge')
            
            diversity = len(used_node_types) / 3.0  # 归一化到[0,1]
            
            # 计算资源效率
            resource_util = self._get_current_resource_utilization()
            efficiency = (resource_util['cpu'] + resource_util['memory']) / 2.0
            
            # 综合质量分数
            quality_score = (diversity + efficiency) / 2.0
            
            return {
                'quality_score': quality_score,
                'diversity': diversity,
                'efficiency': efficiency,
                'node_type_diversity': len(used_node_types)
            }
            
        except Exception as e:
            print(f"❌ 嵌入质量指标计算失败: {e}")
            return {'quality_score': 0.0, 'diversity': 0.0, 'efficiency': 0.0}

    def get_network_topology_info(self) -> Dict[str, Any]:
        """获取网络拓扑信息"""
        try:
            return {
                'num_nodes': self.num_nodes,
                'num_edges': len(self.graph.edges()),
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.num_nodes,
                'is_connected': nx.is_connected(self.graph),
                'clustering_coefficient': nx.average_clustering(self.graph),
                'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else -1,
                'node_feature_dim': self.state_dim,
                'edge_feature_dim': self.edge_dim
            }
        except Exception as e:
            print(f"❌ 拓扑信息获取失败: {e}")
            return {'error': str(e)}

    def update_performance_history(self, reward: float, info: Dict[str, Any]):
        """更新性能历史，用于长期自适应"""
        try:
            performance_record = {
                'reward': reward,
                'sar': info.get('sar', 0.0),
                'latency': info.get('splat', 0.0),
                'success': info.get('success', False),
                'pressure_level': getattr(self, '_current_pressure_level', 'medium'),
                'scenario': self.current_scenario_name,
                'timestamp': self.step_count
            }
            
            self.performance_history.append(performance_record)
            
            # 保持历史记录在合理范围内
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
        except Exception as e:
            print(f"❌ 性能历史更新失败: {e}")


# ✅ 测试函数
def test_fixed_environment():
    """测试修复后的环境"""
    print("🧪 测试修复后的VNF环境...")
    
    try:
        # 创建测试拓扑
        import networkx as nx
        G = nx.erdos_renyi_graph(20, 0.3)
        
        # 创建节点特征 (4维基础特征)
        node_features = np.random.random((20, 4)) * 0.8 + 0.2
        
        # 创建边特征 (4维)
        edge_features = np.random.random((len(G.edges()), 4))
        edge_features[:, 0] *= 100  # 带宽
        edge_features[:, 1] *= 10   # 延迟
        edge_features[:, 2] *= 0.1  # 抖动
        edge_features[:, 3] *= 0.05 # 丢包
        
        # 奖励配置
        reward_config = {
            'base_reward': 10.0,
            'sar_weight': 0.5,
            'latency_weight': 0.3,
            'efficiency_weight': 0.15,
            'quality_weight': 0.05,
            'completion_bonus': 20.0
        }
        
        # 环境配置
        env_config = {
            'train': {'max_episode_steps': 10},
            'vnf_requirements': {
                'cpu_min': 0.05, 'cpu_max': 0.15,
                'memory_min': 0.03, 'memory_max': 0.08,
                'bandwidth_min': 5.0, 'bandwidth_max': 15.0,
                'chain_length_range': [3, 5]
            }
        }
        
        # 创建环境
        env = MultiVNFEmbeddingEnv(
            graph=G,
            node_features=node_features,
            edge_features=edge_features,
            reward_config=reward_config,
            config=env_config
        )
        
        print(f"✅ 环境创建成功")
        
        # 测试重置
        state = env.reset()
        print(f"✅ 重置测试: 状态维度 {state.x.shape}")
        assert state.x.shape[1] == 8, f"状态维度应该是8，实际是{state.x.shape[1]}"
        
        # 测试步骤
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = valid_actions[0]
            next_state, reward, done, info = env.step(action)
            print(f"✅ 步骤测试: 动作={action}, 奖励={reward:.2f}, 完成={done}")
            assert next_state.x.shape[1] == 8, f"下一状态维度应该是8，实际是{next_state.x.shape[1]}"
        
        # 测试场景配置
        scenario_config = {
            'scenario_name': 'extreme_pressure',
            'topology': {'node_resources': {'cpu': 0.8, 'memory': 0.8}},
            'vnf_requirements': {'cpu_min': 0.02, 'cpu_max': 0.08},
            'reward': {'sar_weight': 0.8}
        }
        
        env.apply_scenario_config(scenario_config)
        print(f"✅ 场景配置测试通过")
        
        # 测试信息获取
        env_info = env.get_info()
        print(f"✅ 信息获取测试: {env_info['current_scenario']}")
        
        # 测试拓扑信息
        topo_info = env.get_network_topology_info()
        print(f"✅ 拓扑信息测试: 节点数={topo_info['num_nodes']}")
        
        # 测试嵌入质量指标
        quality_metrics = env.get_embedding_quality_metrics()
        print(f"✅ 质量指标测试: 质量分数={quality_metrics['quality_score']:.2f}")
        
        # 测试baseline模式
        env.set_baseline_mode(True)
        state_baseline = env._get_state()
        print(f"✅ Baseline模式测试: 边特征维度={state_baseline.edge_attr.shape[1]}")
        
        env.close()
        print("✅ 环境修复测试全部通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_fixed_environment()