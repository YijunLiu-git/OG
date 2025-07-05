# models/gnn_encoder.py - 修复版：解决维度匹配问题

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    """
    修复版GNN编码器 - 解决维度匹配问题
    """
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3):
        super(GNNEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 节点嵌入层
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # 边嵌入层：支持可变维度
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # GAT卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, 
                       edge_dim=hidden_dim, dropout=0.1)
            )
        
        # 🔧 修复：全局池化层使用正确的维度
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)
        
        # 🔧 修复：输出层输入维度应该是 2 * hidden_dim（Set2Set的输出）
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Set2Set输出是2倍hidden_dim
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
        """前向传播"""
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 维度验证
        if x.size(1) != self.node_dim:
            raise ValueError(f"❌ 节点特征维度不匹配: 期望{self.node_dim}维，实际{x.size(1)}维")
        
        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            # 支持维度自适应
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
        
        # 🔧 修复：确保graph_embedding维度为 2*hidden_dim
        print(f"🔍 池化后维度: {graph_embedding.shape}, 期望: [batch_size, {2*self.hidden_dim}]")
        
        # 输出层
        graph_embedding = self.output_layers(graph_embedding)
        graph_embedding = F.normalize(graph_embedding, p=2, dim=1)
        
        return graph_embedding


class EdgeAwareGNNEncoder(GNNEncoder):
    """边感知GNN编码器 - 修复版"""
    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=3):
        super(EdgeAwareGNNEncoder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers)
        
        # VNF需求编码器
        self.vnf_requirement_encoder = nn.Linear(6, hidden_dim)
        
        # 边重要性网络
        self.edge_importance_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 🔧 修复：特征融合网络输入维度
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim + hidden_dim, output_dim),  # output_dim + vnf_embedding_dim
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"✅ EdgeAware编码器初始化: VNF上下文支持")
        
    def forward_with_vnf_context(self, data, vnf_context=None):
        """带VNF上下文的前向传播"""
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
            
            # VNF上下文编码
            vnf_embedding = self.vnf_requirement_encoder(vnf_tensor)
            
            # 🔧 修复：特征融合维度匹配
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
    """创建GNN编码器的工厂函数"""
    if mode == 'edge_aware':
        gnn_config = config.get('gnn', {}).get('edge_aware', {})
        encoder = EdgeAwareGNNEncoder(
            node_dim=8,
            edge_dim=gnn_config.get('edge_dim', 4),
            hidden_dim=gnn_config.get('hidden_dim', 128),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('layers', 6)
        )
    else:  # baseline
        gnn_config = config.get('gnn', {}).get('baseline', {})
        encoder = GNNEncoder(
            node_dim=8,
            edge_dim=gnn_config.get('edge_dim', 2),
            hidden_dim=gnn_config.get('hidden_dim', 64),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('layers', 4)
        )
    
    print(f"✅ 创建{mode}模式GNN编码器")
    return encoder


def test_gnn_encoder_fixed():
    """测试修复版GNN编码器"""
    print("🧪 测试修复版GNN编码器...")
    print("=" * 50)
    
    # 测试参数
    num_nodes = 10
    num_edges = 20
    node_dim = 8
    edge_dim_full = 4
    edge_dim_baseline = 2
    
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
    vnf_context = torch.tensor([0.05, 0.03, 0.04, 0.33, 0.5, 0.5])
    
    with torch.no_grad():
        output_with_context = encoder_full.forward_with_vnf_context(data_full, vnf_context)
        print(f"   ✅ VNF上下文: {vnf_context.shape}")
        print(f"   ✅ 融合输出: {output_with_context.shape}")
    
    # 测试4: 维度一致性验证
    print("\n4. 维度一致性验证:")
    assert output_full.shape == output_baseline.shape == output_with_context.shape, "输出维度不一致!"
    print(f"   ✅ 所有模式输出维度一致: {output_full.shape}")
    
    print(f"\n🎉 GNN编码器修复版测试通过!")
    return True

if __name__ == "__main__":
    test_gnn_encoder_fixed()
