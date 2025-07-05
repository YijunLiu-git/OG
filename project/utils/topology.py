import networkx as nx
import numpy as np

def generate_topology(config):
    """
    根据配置生成网络拓扑，目前仅支持 fat-tree。
    参数:
        config (dict): 配置字典，应包含 "type" 和 "k"
    返回:
        graph: networkx 图对象
        node_features: 节点特征矩阵 (N, 8)
        edge_features: 边特征矩阵 (E, 8)
    """
    topo_type = config.get("type", "fat-tree")
    k = config.get("k", 4)

    if topo_type == "fat-tree":
        return generate_fat_tree_topology(k)
    else:
        raise ValueError(f"Unsupported topology type: {topo_type}")


def generate_fat_tree_topology(k):
    """
    生成 fat-tree 拓扑，返回图及其特征。
    节点特征维度: [CPU利用率, MEM利用率, 随机特征1~6]
    边特征维度: [带宽, 延迟, 抖动, 丢包率, 链路利用率, 可靠性, pad1, pad2]
    """
    G = nx.Graph()
    num_core = (k // 2) ** 2
    num_pods = k
    num_agg_per_pod = k // 2
    num_edge_per_pod = k // 2

    core_start = 0
    agg_start = num_core
    edge_start = agg_start + num_pods * num_agg_per_pod
    host_start = edge_start + num_pods * num_edge_per_pod

    # 添加 Core 节点
    for i in range(num_core):
        G.add_node(i, type="core")

    # 添加 Pod 中的 Agg 和 Edge 节点及连接
    for p in range(num_pods):
        for a in range(num_agg_per_pod):
            agg_id = agg_start + p * num_agg_per_pod + a
            G.add_node(agg_id, type="agg")
            for c in range(a * (k // 2), (a + 1) * (k // 2)):
                G.add_edge(agg_id, c)

        for e in range(num_edge_per_pod):
            edge_id = edge_start + p * num_edge_per_pod + e
            G.add_node(edge_id, type="edge")
            for a in range(num_agg_per_pod):
                agg_id = agg_start + p * num_agg_per_pod + a
                G.add_edge(edge_id, agg_id)

    # 节点特征构造
    num_nodes = G.number_of_nodes()
    cpu_util = np.random.uniform(0.2, 0.8, size=(num_nodes, 1))
    mem_util = np.random.uniform(0.2, 0.8, size=(num_nodes, 1))
    pad_node = np.random.rand(num_nodes, 6)  # 占位或自定义特征
    node_features = np.hstack([cpu_util, mem_util, pad_node]).astype(np.float32)

    # 边特征构造
    num_edges = G.number_of_edges()
    bandwidth = np.random.uniform(10.0, 100.0, size=(num_edges, 1))     # Mbps
    delay = np.random.uniform(1.0, 10.0, size=(num_edges, 1))           # ms
    jitter = np.random.uniform(0.1, 1.0, size=(num_edges, 1))           # ms
    packet_loss = np.random.uniform(0.0, 0.05, size=(num_edges, 1))     # 丢包率
    utilization = np.random.uniform(0.1, 0.9, size=(num_edges, 1))      # 链路负载
    reliability = np.random.uniform(0.8, 1.0, size=(num_edges, 1))      # 链路可靠性
    pad_edge = np.random.rand(num_edges, 2)                             # 扩展占位

    edge_features = np.hstack([
        bandwidth, delay, jitter, packet_loss,
        utilization, reliability, pad_edge
    ]).astype(np.float32)

    return G, node_features, edge_features