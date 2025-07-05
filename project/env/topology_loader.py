# env/topology_loader.py

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List
import random

def generate_realistic_backbone_topology(config: Dict) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    生成真实的骨干网络拓扑
    
    特点：
    1. 分层结构：Core -> Aggregation -> Edge
    2. 真实的资源异构性
    3. 动态负载和故障模拟
    4. 地理位置影响的延迟
    
    边特征：[Available_Bandwidth, Current_Latency, Jitter, Packet_Loss]
    """
    
    # 网络层次配置
    core_nodes = 6      # 核心路由器
    agg_nodes = 12      # 汇聚路由器  
    edge_nodes = 24     # 边缘路由器
    total_nodes = core_nodes + agg_nodes + edge_nodes
    
    # 创建图
    G = nx.Graph()
    
    # 节点类型标记
    node_types = {}
    node_locations = {}  # 地理位置
    
    # 1. 添加核心层节点
    for i in range(core_nodes):
        G.add_node(i)
        node_types[i] = 'core'
        node_locations[i] = generate_core_location(i)
    
    # 2. 添加汇聚层节点
    for i in range(core_nodes, core_nodes + agg_nodes):
        G.add_node(i)
        node_types[i] = 'aggregation'
        node_locations[i] = generate_aggregation_location(i - core_nodes)
    
    # 3. 添加边缘层节点
    for i in range(core_nodes + agg_nodes, total_nodes):
        G.add_node(i)
        node_types[i] = 'edge'
        node_locations[i] = generate_edge_location(i - core_nodes - agg_nodes)
    
    # 4. 构建真实的连接模式
    _add_realistic_connections(G, core_nodes, agg_nodes, edge_nodes, node_types)
    
    # 5. 生成异构节点特征
    node_features = _generate_heterogeneous_node_features(total_nodes, node_types, config)
    
    # 6. 生成真实的边特征
    edge_features = _generate_realistic_edge_features(G, node_types, node_locations, config)
    
    return G, node_features, edge_features

def generate_core_location(node_id: int) -> Tuple[float, float]:
    """生成核心节点的地理位置（主要城市）"""
    major_cities = [
        (40.7128, -74.0060),   # 纽约
        (37.7749, -122.4194),  # 旧金山
        (51.5074, -0.1278),    # 伦敦
        (35.6762, 139.6503),   # 东京
        (52.5200, 13.4050),    # 柏林
        (39.9042, 116.4074)    # 北京
    ]
    return major_cities[node_id % len(major_cities)]

def generate_aggregation_location(node_id: int) -> Tuple[float, float]:
    """生成汇聚节点位置（区域中心）"""
    base_locations = [
        (41.8781, -87.6298),   # 芝加哥
        (34.0522, -118.2437),  # 洛杉矶
        (48.8566, 2.3522),     # 巴黎
        (35.6895, 139.6917),   # 东京周边
        (53.5511, 9.9937),     # 汉堡
        (31.2304, 121.4737)    # 上海
    ]
    base_lat, base_lon = base_locations[node_id % len(base_locations)]
    lat_offset = np.random.uniform(-2, 2)
    lon_offset = np.random.uniform(-2, 2)
    return (base_lat + lat_offset, base_lon + lon_offset)

def generate_edge_location(node_id: int) -> Tuple[float, float]:
    """生成边缘节点位置（本地接入点）"""
    lat = np.random.uniform(25, 60)  # 北纬25-60度
    lon = np.random.uniform(-130, 140)  # 经度范围
    return (lat, lon)

def _add_realistic_connections(G: nx.Graph, core_nodes: int, agg_nodes: int, 
                              edge_nodes: int, node_types: Dict):
    """添加真实的网络连接"""
    
    # 1. 核心层全连接
    for i in range(core_nodes):
        for j in range(i + 1, core_nodes):
            G.add_edge(i, j, connection_type='core_core')
    
    # 2. 核心层到汇聚层
    for agg_id in range(core_nodes, core_nodes + agg_nodes):
        core_connections = np.random.choice(
            range(core_nodes), 
            size=np.random.randint(2, 4), 
            replace=False
        )
        for core_id in core_connections:
            G.add_edge(core_id, agg_id, connection_type='core_agg')
    
    # 3. 汇聚层内部连接
    agg_start = core_nodes
    agg_end = core_nodes + agg_nodes
    for i in range(agg_start, agg_end):
        for j in range(i + 1, agg_end):
            if np.random.random() < 0.3:
                G.add_edge(i, j, connection_type='agg_agg')
    
    # 4. 汇聚层到边缘层
    edge_start = core_nodes + agg_nodes
    edge_end = core_nodes + agg_nodes + edge_nodes
    for edge_id in range(edge_start, edge_end):
        agg_connections = np.random.choice(
            range(agg_start, agg_end),
            size=np.random.randint(1, 3),
            replace=False
        )
        for agg_id in agg_connections:
            G.add_edge(agg_id, edge_id, connection_type='agg_edge')
    
    # 5. 边缘层部分互联
    for i in range(edge_start, edge_end):
        for j in range(i + 1, edge_end):
            if np.random.random() < 0.1:
                G.add_edge(i, j, connection_type='edge_edge')

def _generate_heterogeneous_node_features(total_nodes: int, node_types: Dict, config: Dict) -> np.ndarray:
    """
    生成异构节点特征
    
    特征维度：[CPU, Memory, Storage, Network_Capacity]
    """
    
    features = np.zeros((total_nodes, 4))
    
    for node_id in range(total_nodes):
        node_type = node_types[node_id]
        
        if node_type == 'core':
            features[node_id] = [
                np.random.uniform(0.8, 1.0),    # CPU
                np.random.uniform(0.8, 1.0),    # Memory
                np.random.uniform(0.7, 1.0),    # Storage
                np.random.uniform(0.9, 1.0)     # Network_Capacity
            ]
        elif node_type == 'aggregation':
            features[node_id] = [
                np.random.uniform(0.6, 0.8),    # CPU
                np.random.uniform(0.6, 0.8),    # Memory
                np.random.uniform(0.5, 0.8),    # Storage
                np.random.uniform(0.7, 0.9)     # Network_Capacity
            ]
        else:  # edge
            features[node_id] = [
                np.random.uniform(0.3, 0.6),    # CPU
                np.random.uniform(0.3, 0.6),    # Memory
                np.random.uniform(0.2, 0.6),    # Storage
                np.random.uniform(0.4, 0.7)     # Network_Capacity
            ]
        
        noise = np.random.normal(0, 0.05, 4)
        features[node_id] += noise
        features[node_id] = np.clip(features[node_id], 0.01, 1.0)
    
    return features

def _generate_realistic_edge_features(G: nx.Graph, node_types: Dict, 
                                    node_locations: Dict, config: Dict) -> np.ndarray:
    """
    🔧 修复版：生成真实的边特征并同步存储到图
    
    特征维度：[Available_Bandwidth, Current_Latency, Jitter, Packet_Loss]
    """
    
    edges = list(G.edges())
    edge_features = np.zeros((len(edges), 4))
    
    for idx, (u, v) in enumerate(edges):
        connection_type = G.edges[u, v].get('connection_type', 'unknown')
        
        # 计算地理距离影响的延迟
        lat1, lon1 = node_locations[u]
        lat2, lon2 = node_locations[v]
        geo_distance = _calculate_distance(lat1, lon1, lat2, lon2)
        base_latency = max(1.0, geo_distance / 200000)
        
        if connection_type == 'core_core':
            bandwidth = np.random.uniform(80, 100)
            latency = base_latency + np.random.uniform(0.1, 0.5)
            jitter = np.random.uniform(0.01, 0.05)
            packet_loss = np.random.uniform(0.0001, 0.001)
            
        elif connection_type == 'core_agg':
            bandwidth = np.random.uniform(60, 90)
            latency = base_latency + np.random.uniform(0.5, 1.5)
            jitter = np.random.uniform(0.02, 0.08)
            packet_loss = np.random.uniform(0.001, 0.005)
            
        elif connection_type == 'agg_edge':
            bandwidth = np.random.uniform(30, 70)
            latency = base_latency + np.random.uniform(1.0, 3.0)
            jitter = np.random.uniform(0.05, 0.15)
            packet_loss = np.random.uniform(0.005, 0.02)
            
        else:  # 其他连接类型
            bandwidth = np.random.uniform(20, 60)
            latency = base_latency + np.random.uniform(2.0, 5.0)
            jitter = np.random.uniform(0.1, 0.3)
            packet_loss = np.random.uniform(0.01, 0.05)
        
        # 🔧 关键修复：同时更新edge_features矩阵和图的边属性
        edge_features[idx] = [bandwidth, latency, jitter, packet_loss]
        
        # ✅ 将属性存储到图的边中（这是原始代码缺失的关键部分！）
        G.edges[u, v]['bandwidth'] = bandwidth
        G.edges[u, v]['latency'] = latency
        G.edges[u, v]['jitter'] = jitter
        G.edges[u, v]['packet_loss'] = packet_loss
        G.edges[u, v]['available_bandwidth'] = bandwidth  # 初始可用带宽等于总带宽
    
    return edge_features

def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的地理距离（米）"""
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # 地球半径（米）
    
    return c * r

def simulate_dynamic_network_conditions(G: nx.Graph, edge_features: np.ndarray, 
                                      time_step: int) -> np.ndarray:
    """
    模拟动态网络条件
    
    包括：
    1. 流量波动
    2. 链路故障
    """
    
    dynamic_features = edge_features.copy()
    
    daily_cycle = np.sin(2 * np.pi * time_step / 24) * 0.3 + 1.0
    traffic_spikes = np.random.exponential(0.1, len(dynamic_features))
    failure_mask = np.random.random(len(dynamic_features)) < 0.05
    
    for idx in range(len(dynamic_features)):
        traffic_factor = daily_cycle * (1 + traffic_spikes[idx])
        dynamic_features[idx, 0] *= (1 / traffic_factor)  # 带宽
        dynamic_features[idx, 1] *= (1 + traffic_spikes[idx] * 2)  # 延迟
        dynamic_features[idx, 2] *= (1 + traffic_spikes[idx])  # 抖动
        if failure_mask[idx]:
            dynamic_features[idx, 0] *= 0.1  # 带宽下降
            dynamic_features[idx, 1] *= 5.0  # 延迟增加
            dynamic_features[idx, 3] *= 10.0 # 丢包率增加
    
    dynamic_features = np.clip(dynamic_features, 0.001, 1000)
    
    return dynamic_features

def generate_topology(config=None, num_nodes=20, prob=0.3):
    """🔧 修复版：生成网络拓扑，支持 config 参数"""
    if config is None:
        # 默认配置
        config = {
            'topology': {
                'num_nodes': num_nodes,
                'prob': prob,
                'node_resources': {'cpu': 2.0, 'memory': 1.5},
                'edge_resources': {
                    'bandwidth_min': 10.0,
                    'bandwidth_max': 50.0,
                    'latency_min': 1.0,
                    'latency_max': 10.0,
                    'jitter_min': 0.1,
                    'jitter_max': 0.5,
                    'packet_loss_min': 0.0,
                    'packet_loss_max': 0.05
                },
                'node_types': {'core': 0.3, 'aggregation': 0.3, 'edge': 0.4}
            }
        }
    
    topology_config = config.get('topology', {})
    use_realistic = topology_config.get('use_realistic', False)
    
    if use_realistic:
        return generate_realistic_backbone_topology(config)
    else:
        G = nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=None, directed=False)
        
        # 节点特征：[CPU, Memory, Storage, Network_Capacity]
        node_resources = topology_config.get('node_resources', {'cpu': 2.0, 'memory': 1.5})
        node_features = np.zeros((len(G.nodes()), 4))
        for i in range(len(G.nodes())):
            node_features[i, 0] = node_resources.get('cpu', 2.0)
            node_features[i, 1] = node_resources.get('memory', 1.5)
            node_features[i, 2] = np.random.choice([0, 1, 2], p=list(topology_config.get('node_types', {'core': 0.3, 'aggregation': 0.3, 'edge': 0.4}).values()))
            node_features[i, 3] = np.random.uniform(50, 100)
        
        # 🔧 关键修复：边特征生成并同步到图
        edge_resources = topology_config.get('edge_resources', {})
        edge_features = np.zeros((len(G.edges()), 4))
        
        for idx, (u, v) in enumerate(G.edges()):
            bandwidth = np.random.uniform(
                edge_resources.get('bandwidth_min', 10.0),
                edge_resources.get('bandwidth_max', 50.0)
            )
            latency = np.random.uniform(
                edge_resources.get('latency_min', 1.0),
                edge_resources.get('latency_max', 10.0)
            )
            jitter = np.random.uniform(
                edge_resources.get('jitter_min', 0.1),
                edge_resources.get('jitter_max', 0.5)
            )
            packet_loss = np.random.uniform(
                edge_resources.get('packet_loss_min', 0.0),
                edge_resources.get('packet_loss_max', 0.05)
            )
            
            # 同时更新edge_features矩阵和图的边属性
            edge_features[idx] = [bandwidth, latency, jitter, packet_loss]
            
            # ✅ 存储到图的边中（原始代码缺失的关键部分！）
            G.edges[u, v]['bandwidth'] = bandwidth
            G.edges[u, v]['latency'] = latency
            G.edges[u, v]['jitter'] = jitter
            G.edges[u, v]['packet_loss'] = packet_loss
            G.edges[u, v]['available_bandwidth'] = bandwidth
            G.edges[u, v]['connection_type'] = 'default'
        
        return G, node_features, edge_features