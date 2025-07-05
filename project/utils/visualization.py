# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import os

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_curves(results: Dict[str, Any], save_path: str = None, show: bool = True):
    """
    绘制训练曲线对比图
    
    Args:
        results: 训练结果字典
        save_path: 保存路径
        show: 是否显示图片
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 定义颜色和线型
    colors = {'ddqn': '#1f77b4', 'dqn': '#ff7f0e', 'ppo': '#2ca02c'}
    linestyles = {'edge_aware': '-', 'baseline': '--'}
    
    # 1. SAR对比
    ax = axes[0, 0]
    ax.set_title('Service Acceptance Rate (SAR)', fontsize=14, fontweight='bold')
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            for agent_type in results[variant]:
                sar_data = results[variant][agent_type].get('sar', [])
                if sar_data:
                    episodes = list(range(1, len(sar_data) + 1))
                    # 计算滑动平均以平滑曲线
                    smoothed_sar = _smooth_curve(sar_data, window=20)
                    
                    label = f'{agent_type.upper()} ({variant.replace("_", "-")})'
                    ax.plot(episodes[:len(smoothed_sar)], smoothed_sar, 
                           color=colors.get(agent_type, 'gray'),
                           linestyle=linestyles[variant],
                           label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('SAR')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 2. SPLat对比
    ax = axes[0, 1]
    ax.set_title('Service Path Latency (SPLat)', fontsize=14, fontweight='bold')
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            for agent_type in results[variant]:
                splat_data = [s for s in results[variant][agent_type].get('splat', []) 
                             if s != float('inf')]
                if splat_data:
                    episodes = list(range(1, len(splat_data) + 1))
                    smoothed_splat = _smooth_curve(splat_data, window=20)
                    
                    label = f'{agent_type.upper()} ({variant.replace("_", "-")})'
                    ax.plot(episodes[:len(smoothed_splat)], smoothed_splat,
                           color=colors.get(agent_type, 'gray'),
                           linestyle=linestyles[variant],
                           label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('SPLat')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 3. 奖励对比
    ax = axes[1, 0]
    ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            for agent_type in results[variant]:
                reward_data = results[variant][agent_type].get('rewards', [])
                if reward_data:
                    episodes = list(range(1, len(reward_data) + 1))
                    smoothed_rewards = _smooth_curve(reward_data, window=20)
                    
                    label = f'{agent_type.upper()} ({variant.replace("_", "-")})'
                    ax.plot(episodes[:len(smoothed_rewards)], smoothed_rewards,
                           color=colors.get(agent_type, 'gray'),
                           linestyle=linestyles[variant],
                           label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. 成功率对比
    ax = axes[1, 1]
    ax.set_title('Success Rate (Rolling Average)', fontsize=14, fontweight='bold')
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            for agent_type in results[variant]:
                success_data = results[variant][agent_type].get('success', [])
                if success_data:
                    # 计算滑动平均成功率
                    rolling_success = _rolling_average([float(s) for s in success_data], 30)
                    episodes = list(range(30, 30 + len(rolling_success)))
                    
                    label = f'{agent_type.upper()} ({variant.replace("_", "-")})'
                    ax.plot(episodes, rolling_success,
                           color=colors.get(agent_type, 'gray'),
                           linestyle=linestyles[variant],
                           label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 训练曲线已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_performance_comparison(results: Dict[str, Any], save_path: str = None, show: bool = True):
    """
    绘制性能对比柱状图
    
    Args:
        results: 训练结果字典
        save_path: 保存路径
        show: 是否显示图片
    """
    # 准备数据
    metrics_data = []
    window = 50  # 使用最后50个episode的平均值
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            for agent_type in results[variant]:
                agent_results = results[variant][agent_type]
                
                # 计算平均指标
                recent_sar = np.mean(agent_results.get('sar', [0])[-window:])
                recent_splat = np.mean([s for s in agent_results.get('splat', [float('inf')])[-window:] 
                                      if s != float('inf')])
                recent_reward = np.mean(agent_results.get('rewards', [0])[-window:])
                
                metrics_data.append({
                    'Variant': variant.replace('_', '-'),
                    'Algorithm': agent_type.upper(),
                    'SAR': recent_sar,
                    'SPLat': recent_splat if recent_splat != float('inf') else 0,
                    'Reward': recent_reward
                })
    
    df = pd.DataFrame(metrics_data)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # SAR对比
    ax = axes[0]
    sns.barplot(data=df, x='Algorithm', y='SAR', hue='Variant', ax=ax)
    ax.set_title('Service Acceptance Rate (SAR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('SAR')
    ax.set_ylim(0, 1.0)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10)
    
    # SPLat对比
    ax = axes[1]
    sns.barplot(data=df, x='Algorithm', y='SPLat', hue='Variant', ax=ax)
    ax.set_title('Service Path Latency (SPLat)', fontsize=14, fontweight='bold')
    ax.set_ylabel('SPLat')
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Reward对比
    ax = axes[2]
    sns.barplot(data=df, x='Algorithm', y='Reward', hue='Variant', ax=ax)
    ax.set_title('Average Reward', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward')
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 性能对比图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_improvement_analysis(results: Dict[str, Any], save_path: str = None, show: bool = True):
    """
    绘制改进分析图
    
    Args:
        results: 训练结果字典
        save_path: 保存路径
        show: 是否显示图片
    """
    if 'edge_aware' not in results or 'baseline' not in results:
        print("❌ 缺少edge_aware或baseline结果，无法进行改进分析")
        return
    
    # 计算改进幅度
    improvements = {}
    window = 50
    
    for agent_type in results['edge_aware']:
        if agent_type in results['baseline']:
            # SAR改进
            edge_sar = np.mean(results['edge_aware'][agent_type].get('sar', [0])[-window:])
            baseline_sar = np.mean(results['baseline'][agent_type].get('sar', [0])[-window:])
            sar_improvement = ((edge_sar - baseline_sar) / baseline_sar * 100) if baseline_sar > 0 else 0
            
            # SPLat改进（越低越好）
            edge_splat = np.mean([s for s in results['edge_aware'][agent_type].get('splat', [float('inf')])[-window:] 
                                if s != float('inf')])
            baseline_splat = np.mean([s for s in results['baseline'][agent_type].get('splat', [float('inf')])[-window:] 
                                    if s != float('inf')])
            splat_improvement = ((baseline_splat - edge_splat) / baseline_splat * 100) if baseline_splat > 0 else 0
            
            # Reward改进
            edge_reward = np.mean(results['edge_aware'][agent_type].get('rewards', [0])[-window:])
            baseline_reward = np.mean(results['baseline'][agent_type].get('rewards', [0])[-window:])
            reward_improvement = ((edge_reward - baseline_reward) / baseline_reward * 100) if baseline_reward != 0 else 0
            
            improvements[agent_type.upper()] = {
                'SAR': sar_improvement,
                'SPLat': splat_improvement,
                'Reward': reward_improvement
            }
    
    # 绘制改进幅度图
    algorithms = list(improvements.keys())
    metrics = ['SAR', 'SPLat', 'Reward']
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        values = [improvements[alg][metric] for alg in algorithms]
        bars = ax.bar(x + i * width, values, width, label=f'{metric} Improvement', 
                     color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                   f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Edge-Aware vs Baseline Performance Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 改进分析图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_convergence_analysis(results: Dict[str, Any], save_path: str = None, show: bool = True):
    """
    绘制收敛性分析图
    
    Args:
        results: 训练结果字典
        save_path: 保存路径
        show: 是否显示图片
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义颜色
    colors = {'ddqn': '#1f77b4', 'dqn': '#ff7f0e', 'ppo': '#2ca02c'}
    linestyles = {'edge_aware': '-', 'baseline': '--'}
    
    # 左图：训练曲线的标准差（稳定性）
    ax = axes[0]
    ax.set_title('Training Stability (Reward Std)', fontsize=14, fontweight='bold')
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            for agent_type in results[variant]:
                reward_data = results[variant][agent_type].get('rewards', [])
                if len(reward_data) > 50:
                    # 计算滑动标准差
                    window = 30
                    rolling_std = []
                    for i in range(window, len(reward_data)):
                        window_data = reward_data[i-window:i]
                        rolling_std.append(np.std(window_data))
                    
                    episodes = list(range(window, window + len(rolling_std)))
                    
                    label = f'{agent_type.upper()} ({variant.replace("_", "-")})'
                    ax.plot(episodes, rolling_std,
                           color=colors.get(agent_type, 'gray'),
                           linestyle=linestyles[variant],
                           label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Std (30-episode window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右图：学习速度（到达目标性能的episode数）
    ax = axes[1]
    ax.set_title('Learning Speed (Episodes to Target SAR)', fontsize=14, fontweight='bold')
    
    target_sar = 0.7  # 目标SAR阈值
    learning_speeds = {}
    
    for variant in ['edge_aware', 'baseline']:
        if variant in results:
            learning_speeds[variant] = {}
            for agent_type in results[variant]:
                sar_data = results[variant][agent_type].get('sar', [])
                
                # 找到首次达到目标SAR的episode
                convergence_episode = -1
                window = 10
                for i in range(window, len(sar_data)):
                    recent_sar = np.mean(sar_data[i-window:i])
                    if recent_sar >= target_sar:
                        convergence_episode = i
                        break
                
                learning_speeds[variant][agent_type] = convergence_episode if convergence_episode != -1 else len(sar_data)
    
    # 绘制柱状图
    algorithms = list(set().union(*[learning_speeds[v].keys() for v in learning_speeds.keys()]))
    x = np.arange(len(algorithms))
    width = 0.35
    
    edge_speeds = [learning_speeds.get('edge_aware', {}).get(alg, 0) for alg in algorithms]
    baseline_speeds = [learning_speeds.get('baseline', {}).get(alg, 0) for alg in algorithms]
    
    bars1 = ax.bar(x - width/2, edge_speeds, width, label='Edge-aware', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_speeds, width, label='Baseline', alpha=0.8, color='orange')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(f'Episodes to SAR ≥ {target_sar}')
    ax.set_xticks(x)
    ax.set_xticklabels([alg.upper() for alg in algorithms])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 收敛分析图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_network_topology(graph, node_features=None, embedding_map=None, 
                         save_path: str = None, show: bool = True):
    """
    绘制网络拓扑图
    
    Args:
        graph: NetworkX图对象
        node_features: 节点特征矩阵
        embedding_map: VNF嵌入映射
        save_path: 保存路径
        show: 是否显示图片
    """
    import networkx as nx
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 计算节点位置
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1, edge_color='gray', ax=ax)
    
    # 绘制节点
    if embedding_map:
        # 区分已使用和未使用的节点
        used_nodes = set(embedding_map.values())
        unused_nodes = set(graph.nodes()) - used_nodes
        
        # 绘制未使用的节点
        nx.draw_networkx_nodes(graph, pos, nodelist=list(unused_nodes), 
                              node_color='lightblue', node_size=300, alpha=0.7, ax=ax)
        
        # 绘制已使用的节点
        nx.draw_networkx_nodes(graph, pos, nodelist=list(used_nodes), 
                              node_color='red', node_size=500, alpha=0.8, ax=ax)
        
        # 添加VNF标签
        vnf_labels = {}
        for vnf, node in embedding_map.items():
            vnf_labels[node] = f"{vnf}\n({node})"
        
        nx.draw_networkx_labels(graph, pos, labels=vnf_labels, 
                               font_size=8, font_weight='bold', ax=ax)
        
        # 绘制未使用节点的标签
        unused_labels = {node: str(node) for node in unused_nodes}
        nx.draw_networkx_labels(graph, pos, labels=unused_labels, 
                               font_size=8, ax=ax)
    else:
        # 普通节点绘制
        node_colors = []
        if node_features is not None:
            # 根据CPU资源着色
            cpu_values = node_features[:, 0] if len(node_features.shape) > 1 else node_features
            node_colors = cpu_values
        else:
            node_colors = 'lightblue'
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.7, cmap='YlOrRd', ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    
    ax.set_title('Network Topology' + (' with VNF Embedding' if embedding_map else ''), 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if embedding_map:
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='Available Nodes'),
            Patch(facecolor='red', alpha=0.8, label='VNF Embedded Nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"🌐 网络拓扑图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def _smooth_curve(data: List[float], window: int = 10) -> List[float]:
    """平滑曲线"""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return smoothed

def _rolling_average(data: List[float], window: int) -> List[float]:
    """滑动平均"""
    if len(data) < window:
        return []
    
    rolling_avg = []
    for i in range(len(data) - window + 1):
        rolling_avg.append(np.mean(data[i:i + window]))
    
    return rolling_avg

def create_comprehensive_report(results: Dict[str, Any], output_dir: str):
    """
    创建综合报告
    
    Args:
        results: 训练结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📋 生成综合报告...")
    
    # 1. 训练曲线图
    plot_training_curves(results, 
                         save_path=os.path.join(output_dir, 'training_curves.png'), 
                         show=False)
    
    # 2. 性能对比图
    plot_performance_comparison(results, 
                               save_path=os.path.join(output_dir, 'performance_comparison.png'), 
                               show=False)
    
    # 3. 改进分析图
    plot_improvement_analysis(results, 
                             save_path=os.path.join(output_dir, 'improvement_analysis.png'), 
                             show=False)
    
    # 4. 收敛分析图
    plot_convergence_analysis(results, 
                             save_path=os.path.join(output_dir, 'convergence_analysis.png'), 
                             show=False)
    
    # 5. 生成数据摘要
    from utils.metrics import generate_performance_summary, export_metrics_comparison
    
    summary = generate_performance_summary(results)
    
    # 保存摘要
    import json
    with open(os.path.join(output_dir, 'performance_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 导出对比表
    if 'edge_aware' in results and 'baseline' in results:
        export_metrics_comparison(
            results['edge_aware'], 
            results['baseline'],
            os.path.join(output_dir, 'metrics_comparison.csv')
        )
    
    print(f"✅ 综合报告已生成: {output_dir}")
    print(f"   包含文件:")
    print(f"   - training_curves.png")
    print(f"   - performance_comparison.png") 
    print(f"   - improvement_analysis.png")
    print(f"   - convergence_analysis.png")
    print(f"   - performance_summary.json")
    print(f"   - metrics_comparison.csv")