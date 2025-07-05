# models/enhanced_gnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, Set2Set
from torch_geometric.data import Data, Batch
import numpy as np

class EdgeAttentionLayer(nn.Module):
    def __init__(self, edge_dim, hidden_dim, vnf_context_dim=6):
        super(EdgeAttentionLayer, self).__init__()
        
        self.edge_dim = edge_dim  # 输入边特征维度
        self.hidden_dim = hidden_dim
        self.vnf_context_dim = vnf_context_dim
        
        # 边特征投影
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # VNF需求编码器
        self.vnf_encoder = nn.Linear(vnf_context_dim, hidden_dim)
        
        # 注意力计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 边重要性分类器
        self.importance_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3个重要性等级：低、中、高
            nn.Softmax(dim=-1)
        )
        
        # 全局网络状态感知
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, edge_attr, edge_index, vnf_context=None, network_state=None):
        if edge_attr.size(1) != self.edge_dim:
            raise ValueError(f"边特征维度不匹配: 期望 {self.edge_dim}, 实际 {edge_attr.size(1)}")
        
        # 边特征编码
        edge_features = self.edge_proj(edge_attr)  # [E, hidden_dim]
        
        # VNF上下文融合
        if vnf_context is not None:
            if vnf_context.dim() == 1:
                vnf_context = vnf_context.unsqueeze(0)
            vnf_embedding = self.vnf_encoder(vnf_context)  # [1, hidden_dim]
            vnf_broadcast = vnf_embedding.expand(edge_features.size(0), -1)
            combined_features = torch.cat([edge_features, vnf_broadcast], dim=-1)
            edge_attention = self.attention_net(combined_features).squeeze(-1)
        else:
            edge_attention = torch.ones(edge_features.size(0), device=edge_features.device)
        
        # 全局网络状态感知
        if network_state is not None:
            if network_state.dim() == 1:
                network_state = network_state.unsqueeze(0)
            network_state_encoded = self.network_state_encoder(network_state)
            edge_features_expanded = edge_features.unsqueeze(0)
            network_state_expanded = network_state_encoded.unsqueeze(0)
            attended_features, _ = self.global_attention(
                edge_features_expanded, 
                network_state_expanded, 
                network_state_expanded
            )
            edge_features = attended_features.squeeze(0)
        
        # 边重要性分类
        edge_importance = self.importance_classifier(edge_features)
        weighted_edge_features = edge_features * edge_attention.unsqueeze(-1)
        
        return edge_attention, weighted_edge_features, edge_importance

class PathQualityAwareConv(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4):
        super(PathQualityAwareConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim  # 输入边特征维度
        self.heads = heads
        
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=0.1
        )
        
        self.path_quality_encoder = nn.Sequential(
            nn.Linear(4, edge_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_dim // 2, edge_dim)
        )
        
        self.constraint_awareness = nn.Sequential(
            nn.Linear(out_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, edge_index, edge_attr, path_quality_info=None):
        if edge_attr.size(1) != self.edge_dim:
            raise ValueError(f"边特征维度不匹配: 期望 {self.edge_dim}, 实际 {edge_attr.size(1)}")
        
        x_conv = self.gat(x, edge_index, edge_attr)
        
        if path_quality_info is not None:
            quality_features = self._extract_path_quality_features(edge_index, edge_attr, path_quality_info)
            enhanced_edge_attr = edge_attr + quality_features
            x_conv = self.gat(x, edge_index, enhanced_edge_attr)
        
        if path_quality_info is not None:
            constraint_features = self._compute_constraint_features(x_conv, path_quality_info)
            x_out = self.constraint_awareness(torch.cat([x_conv, constraint_features], dim=-1))
        else:
            x_out = x_conv
        
        return x_out
    
    def _extract_path_quality_features(self, edge_index, edge_attr, path_quality_info):
        quality_matrix = path_quality_info.get('path_quality_matrix', {})
        quality_features = torch.zeros_like(edge_attr)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            src_idx, dst_idx = src.item(), dst.item()
            path_info = quality_matrix.get((src_idx, dst_idx), {})
            
            if path_info:
                bandwidth = path_info.get('bandwidth', 0.0)
                latency = path_info.get('latency', 0.0)
                jitter = path_info.get('jitter', 0.0)
                loss = path_info.get('packet_loss', 0.0)
                
                quality_vec = torch.tensor([bandwidth/100.0, latency/100.0, jitter/5.0, loss], 
                                         device=edge_attr.device, dtype=torch.float32)
                quality_encoded = self.path_quality_encoder(quality_vec)
                quality_features[i] = quality_encoded
        
        return quality_features
    
    def _compute_constraint_features(self, node_features, path_quality_info):
        network_state = path_quality_info.get('network_state_vector', torch.zeros(8))
        
        if isinstance(network_state, np.ndarray):
            network_state = torch.tensor(network_state, device=node_features.device, dtype=torch.float32)
        
        constraint_features = network_state.unsqueeze(0).expand(node_features.size(0), -1)
        
        if constraint_features.size(-1) != self.edge_dim:
            constraint_features = F.adaptive_avg_pool1d(
                constraint_features.unsqueeze(1), self.edge_dim
            ).squeeze(1)
        
        return constraint_features

class EnhancedEdgeAwareGNN(nn.Module):
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=256, num_layers=6, vnf_context_dim=6):
        super(EnhancedEdgeAwareGNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.vnf_context_dim = vnf_context_dim
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        self.edge_attention = EdgeAttentionLayer(edge_dim, hidden_dim, vnf_context_dim)
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                PathQualityAwareConv(hidden_dim, hidden_dim, hidden_dim, heads=4)
            )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.vnf_context_encoder = nn.Sequential(
            nn.Linear(vnf_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)
        
        self.output_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU()
        )
        
        self.network_state_encoder = nn.Linear(8, hidden_dim)
        
        print(f"🚀 增强Edge-Aware GNN编码器初始化完成")
        print(f"   - 节点维度: {node_dim} -> {hidden_dim}")
        print(f"   - 边维度: {edge_dim} -> {hidden_dim}")
        print(f"   - 卷积层数: {num_layers}")
        print(f"   - 输出维度: {output_dim}")
        print(f"   - 核心创新: 边注意力 + 路径质量感知")
        
    def forward(self, data):
        if isinstance(data, list):
            data = Batch.from_data_list(data)
        
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        vnf_context = getattr(data, 'vnf_context', None)
        network_state = getattr(data, 'network_state', None)
        enhanced_info = getattr(data, 'enhanced_info', None)
        
        self._validate_input_dimensions(x, edge_attr)
        
        x = self.node_embedding(x)
        if edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
        
        edge_attention, enhanced_edge_features, edge_importance = self.edge_attention(
            edge_attr, edge_index, vnf_context, network_state
        )
        
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_residual = x
            x = conv(x, edge_index, enhanced_edge_features, enhanced_info)
            x = norm(x)
            if x_residual.size() == x.size():
                x = x + x_residual
            x = F.relu(x)
        
        if batch is not None:
            graph_embedding = self.global_pool(x, batch)
        else:
            batch_single = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_embedding = self.global_pool(x, batch_single)
        
        if vnf_context is not None:
            vnf_embedding = self.vnf_context_encoder(vnf_context.float())
            if vnf_embedding.dim() == 1:
                vnf_embedding = vnf_embedding.unsqueeze(0)
            if vnf_embedding.size(0) != graph_embedding.size(0):
                vnf_embedding = vnf_embedding.expand(graph_embedding.size(0), -1)
            combined_embedding = torch.cat([graph_embedding, vnf_embedding], dim=-1)
        else:
            zero_context = torch.zeros(graph_embedding.size(0), self.hidden_dim, 
                                     device=graph_embedding.device)
            combined_embedding = torch.cat([graph_embedding, zero_context], dim=-1)
        
        final_embedding = self.output_net(combined_embedding)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)
        
        return final_embedding
    
    def _validate_input_dimensions(self, x, edge_attr):
        if x.size(1) != self.node_dim:
            raise ValueError(f"节点特征维度不匹配: 期望{self.node_dim}, 实际{x.size(1)}")
        if edge_attr is not None and edge_attr.size(1) != self.edge_dim:
            raise ValueError(f"边特征维度不匹配: 期望{self.edge_dim}, 实际{edge_attr.size(1)}")
    
    def compute_edge_importance_map(self, data):
        with torch.no_grad():
            edge_index = data.edge_index
            edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
            vnf_context = getattr(data, 'vnf_context', None)
            network_state = getattr(data, 'network_state', None)
            
            if edge_attr is not None:
                edge_attr = self.edge_embedding(edge_attr)
            
            edge_attention, _, edge_importance = self.edge_attention(
                edge_attr, edge_index, vnf_context, network_state
            )
            
            edge_importance_map = {}
            for i, (src, dst) in enumerate(edge_index.t()):
                edge_key = (src.item(), dst.item())
                edge_importance_map[edge_key] = {
                    'attention_weight': edge_attention[i].item(),
                    'importance_scores': edge_importance[i].cpu().numpy(),
                    'importance_level': edge_importance[i].argmax().item()
                }
            
            return edge_importance_map