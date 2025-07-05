# models/gnn_encoder.py - 修复版：统一8维节点特征

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    """
    修复版GNN编码器 - 解决维度不一致问题
    
    ✅ 修复要点：
    1. 节点特征统一为8维：[CPU, Memory, Storage, Network_Capacity, is_used, cpu_util, memory_util, vnf_count]
    2. 边特征支持4维(edge-aware)和2维(baseline)两种模式
    3. 确保与环境状态生成逻辑完全一致
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3):
        super(GNNEncoder, self).__init__()
        
        # ✅ 关键修复：节点维度统一为8
        self.node_dim = node_dim  # 8维：4基础特征 + 4状态特征
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # ✅ 节点嵌入层：处理8维输入
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # ✅ 边嵌入层：支持可变维度
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # GAT卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, 
                       edge_dim=hidden_dim, dropout=0.1)
            )
        
        # 全局池化
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        print(f"✅ GNN编码器初始化: 节点{node_dim}维 -> 隐藏{hidden_dim}维 -> 输出{output_dim}维")
        
    def forward(self, data):
        """
        前向传播
        
        输入:
        - data.x: [N, 8] 节点特征矩阵
        - data.edge_index: [2, E] 边索引  
        - data.edge_attr: [E, edge_dim] 边特征矩阵
        """
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # ✅ 维度验证
        if x.size(1) != self.node_dim:
            raise ValueError(f"❌ 节点特征维度不匹配: 期望{self.node_dim}维，实际{x.size(1)}维")
        
        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            # ✅ 支持维度自适应：如果边特征是2维但期望4维，用零填充
            if self.edge_dim == 4 and edge_attr.size(1) == 2:
                padding = torch.zeros(edge_attr.size(0), 2, device=edge_attr.device)
                edge_attr = torch.cat([edge_attr, padding], dim=1)
                print(f"🔧 边特征自动扩展: 2维 -> 4维")
            else:
                raise ValueError(f"❌ 边特征维度不匹配: 期望{self.edge_dim}维，实际{edge_attr.size(1)}维")
        
        # 特征嵌入
        x = self.node_embedding(x)
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
            edge_attr = F.normalize(edge_attr, p=2, dim=1)
        
        # GNN卷积
        for i, conv in enumerate(self.conv_layers):
            x_residual = x
            
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            # 批归一化
            if batch is not None:
                x = self.batch_norms[i](x)
            else:
                if x.size(0) > 1:
                    x = self.batch_norms[i](x)
            
            x = F.relu(x)
            
            # 残差连接
            if x_residual.size() == x.size():
                x = x + x_residual
        
        # 全局池化
        if batch is not None:
            graph_embedding = self.global_pool(x, batch)
        else:
            batch_single = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = self.global_pool(x, batch_single)
        
        # 输出层
        graph_embedding = self.output_layers(graph_embedding)
        graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
        
        return graph_embedding
    
    def encode_network_state(self, graph, node_features, edge_features):
        """
        编码网络状态为固定维度向量
        
        ✅ 修复版：确保输入特征维度正确
        """
        edge_list = list(graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # ✅ 确保节点特征是8维
        if node_features.shape[1] != 8:
            print(f"⚠️ 节点特征维度({node_features.shape[1]})不是8，需要在环境中修复")
        
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        with torch.no_grad():
            encoded_state = self.forward(data)
        
        return encoded_state.squeeze(0)
    
    def get_edge_importance(self, data):
        """计算边的重要性权重"""
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            return torch.ones(data.edge_attr.size(0))
        else:
            return torch.ones(data.edge_index.size(1))

class EdgeAwareGNNEncoder(GNNEncoder):
    """
    边感知GNN编码器 - 修复版
    
    ✅ 增强功能：
    1. VNF需求上下文编码
    2. 边重要性评估
    3. 动态特征融合
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3):
        super(EdgeAwareGNNEncoder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers)
        
        # ✅ VNF需求编码器：处理6维VNF上下文
        self.vnf_requirement_encoder = nn.Linear(6, hidden_dim)  # [cpu, memory, bandwidth/100, vnf_type/3, progress, remaining]
        
        # 边重要性网络
        self.edge_importance_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"✅ EdgeAware编码器初始化: VNF上下文支持")
        
    def forward_with_vnf_context(self, data, vnf_context=None):
        """
        带VNF上下文的前向传播
        
        参数:
        - data: 图数据
        - vnf_context: [6] VNF需求向量
        """
        # 基础图编码
        graph_embedding = self.forward(data)
        
        # VNF上下文融合
        if vnf_context is not None:
            if isinstance(vnf_context, torch.Tensor):
                vnf_tensor = vnf_context.float()
            else:
                vnf_tensor = torch.tensor(vnf_context, dtype=torch.float32)
            
            if vnf_tensor.dim() == 1:
                vnf_tensor = vnf_tensor.unsqueeze(0)
            
            # ✅ VNF上下文编码
            vnf_embedding = self.vnf_requirement_encoder(vnf_tensor)
            
            # 特征融合
            if graph_embedding.size(0) == vnf_embedding.size(0):
                fused_features = torch.cat([graph_embedding, vnf_embedding], dim=1)
                enhanced_embedding = self.feature_fusion(fused_features)
            else:
                # 广播处理
                enhanced_embedding = graph_embedding + 0.3 * vnf_embedding.mean(dim=0, keepdim=True)
            
            return enhanced_embedding
        else:
            return graph_embedding
    
    def compute_edge_attention(self, data):
        """计算边注意力权重"""
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            attention_weights = self.edge_importance_net(data.edge_attr.float())
            return attention_weights.squeeze(-1)
        else:
            return torch.ones(data.edge_index.size(1), device=data.edge_index.device)

def create_gnn_encoder(config: dict, mode: str = 'edge_aware'):
    """
    创建GNN编码器的工厂函数
    
    参数:
    - config: 配置字典
    - mode: 'edge_aware' 或 'baseline'
    """
    if mode == 'edge_aware':
        gnn_config = config.get('gnn', {}).get('edge_aware', {})
        encoder = EdgeAwareGNNEncoder(
            node_dim=8,  # ✅ 统一为8维
            edge_dim=gnn_config.get('edge_dim', 4),
            hidden_dim=gnn_config.get('hidden_dim', 128),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('layers', 6)
        )
    else:  # baseline
        gnn_config = config.get('gnn', {}).get('baseline', {})
        encoder = GNNEncoder(
            node_dim=8,  # ✅ 统一为8维
            edge_dim=gnn_config.get('edge_dim', 2),
            hidden_dim=gnn_config.get('hidden_dim', 64),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('layers', 4)
        )
    
    print(f"✅ 创建{mode}模式GNN编码器")
    return encoder

def test_gnn_encoder_fixed():
    """
    测试修复版GNN编码器
    """
    print("🧪 测试修复版GNN编码器...")
    print("=" * 50)
    
    # 测试参数
    num_nodes = 10
    num_edges = 20
    node_dim = 8  # ✅ 修复：使用8维节点特征
    edge_dim_full = 4  # edge-aware模式
    edge_dim_baseline = 2  # baseline模式
    
    # 生成测试数据
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_full = torch.randn(num_edges, edge_dim_full)
    edge_attr_baseline = torch.randn(num_edges, edge_dim_baseline)
    
    # 测试1: EdgeAware模式
    print("\n1. 测试EdgeAware模式:")
    data_full = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_full)
    encoder_full = EdgeAwareGNNEncoder(node_dim=node_dim, edge_dim=edge_dim_full)
    
    with torch.no_grad():
        output_full = encoder_full(data_full)
        print(f"   ✅ 输入: {num_nodes}节点×{node_dim}维, {num_edges}边×{edge_dim_full}维")
        print(f"   ✅ 输出: {output_full.shape}")
    
    # 测试2: Baseline模式
    print("\n2. 测试Baseline模式:")
    data_baseline = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_baseline)
    encoder_baseline = GNNEncoder(node_dim=node_dim, edge_dim=edge_dim_baseline)
    
    with torch.no_grad():
        output_baseline = encoder_baseline(data_baseline)
        print(f"   ✅ 输入: {num_nodes}节点×{node_dim}维, {num_edges}边×{edge_dim_baseline}维")
        print(f"   ✅ 输出: {output_baseline.shape}")
    
    # 测试3: VNF上下文
    print("\n3. 测试VNF上下文融合:")
    vnf_context = torch.tensor([0.05, 0.03, 0.04, 0.33, 0.5, 0.5])  # 6维VNF上下文
    
    with torch.no_grad():
        output_with_context = encoder_full.forward_with_vnf_context(data_full, vnf_context)
        print(f"   ✅ VNF上下文: {vnf_context.shape}")
        print(f"   ✅ 融合输出: {output_with_context.shape}")
    
    # 测试4: 维度一致性验证
    print("\n4. 维度一致性验证:")
    assert output_full.shape == output_baseline.shape == output_with_context.shape, "输出维度不一致!"
    print(f"   ✅ 所有模式输出维度一致: {output_full.shape}")
    
    # 测试5: 边特征自适应
    print("\n5. 测试边特征自适应:")
    data_adaptive = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_baseline)
    encoder_adaptive = GNNEncoder(node_dim=node_dim, edge_dim=edge_dim_full)  # 期望4维但输入2维
    
    with torch.no_grad():
        output_adaptive = encoder_adaptive(data_adaptive)
        print(f"   ✅ 自适应处理: 2维边特征 -> 4维编码器")
        print(f"   ✅ 输出: {output_adaptive.shape}")
    
    print(f"\n🎉 GNN编码器修复版测试通过!")
    print(f"   - 支持8维节点特征 ✅")
    print(f"   - 支持4维/2维边特征 ✅") 
    print(f"   - VNF上下文融合 ✅")
    print(f"   - 维度自适应 ✅")
    print(f"   - 输出维度固定 ✅")

if __name__ == "__main__":
    test_gnn_encoder_fixed()