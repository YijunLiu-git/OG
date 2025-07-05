# utils/metrics.py

import numpy as np
from typing import List, Dict, Any, Tuple

def calculate_sar(success_list: List[bool], window_size: int = None) -> float:
    """
    计算服务接受率 (Service Acceptance Rate)
    
    Args:
        success_list: 成功标志列表
        window_size: 滑动窗口大小，None表示使用全部数据
        
    Returns:
        sar: 服务接受率 (0.0 - 1.0)
    """
    if not success_list:
        return 0.0
    
    if window_size is not None and len(success_list) > window_size:
        success_list = success_list[-window_size:]
    
    successes = sum(1 for s in success_list if s)  # 明确统计True值
    total = len(success_list)
    sar = successes / total if total > 0 else 0.0
    print(f"计算SAR: 成功={successes}, 总计={total}, SAR={sar:.3f}")  # ✅ 添加调试信息
    return sar

def calculate_splat(latency_list: List[float], window_size: int = None) -> float:
    """
    计算平均服务路径延迟 (Service Path Latency)
    
    Args:
        latency_list: 延迟值列表
        window_size: 滑动窗口大小，None表示使用全部数据
        
    Returns:
        splat: 平均路径延迟
    """
    if not latency_list:
        return float('inf')
    
    # 过滤无效值
    valid_latencies = [lat for lat in latency_list if lat != float('inf') and not np.isnan(lat)]
    
    if not valid_latencies:
        return float('inf')
    
    if window_size is not None and len(valid_latencies) > window_size:
        valid_latencies = valid_latencies[-window_size:]
    
    return np.mean(valid_latencies)

def calculate_resource_utilization(embedding_map: Dict[str, int], 
                                 initial_resources: np.ndarray,
                                 current_resources: np.ndarray) -> Dict[str, float]:
    """
    计算资源利用率
    
    Args:
        embedding_map: VNF到节点的映射
        initial_resources: 初始资源矩阵
        current_resources: 当前资源矩阵
        
    Returns:
        utilization: 资源利用率字典
    """
    if len(embedding_map) == 0:
        return {'cpu': 0.0, 'memory': 0.0, 'overall': 0.0}
    
    # 计算CPU利用率
    total_cpu_capacity = np.sum(initial_resources[:, 0])
    used_cpu = np.sum(initial_resources[:, 0] - current_resources[:, 0])
    cpu_utilization = used_cpu / total_cpu_capacity if total_cpu_capacity > 0 else 0.0
    
    # 计算内存利用率（如果存在）
    memory_utilization = 0.0
    if initial_resources.shape[1] > 1:
        total_memory_capacity = np.sum(initial_resources[:, 1])
        used_memory = np.sum(initial_resources[:, 1] - current_resources[:, 1])
        memory_utilization = used_memory / total_memory_capacity if total_memory_capacity > 0 else 0.0
    
    # 整体利用率
    overall_utilization = (cpu_utilization + memory_utilization) / 2
    
    return {
        'cpu': cpu_utilization,
        'memory': memory_utilization,
        'overall': overall_utilization
    }

def calculate_network_efficiency(paths_info: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算网络效率指标
    
    Args:
        paths_info: 路径信息列表
        
    Returns:
        efficiency: 网络效率指标
    """
    if not paths_info:
        return {
            'avg_delay': float('inf'),
            'avg_hops': float('inf'),
            'min_bandwidth': 0.0,
            'avg_jitter': float('inf'),
            'avg_loss': 1.0
        }
    
    delays = [path.get('delay', 0) for path in paths_info]
    hops = [path.get('hops', 0) for path in paths_info]
    bandwidths = [path.get('bandwidth', 0) for path in paths_info]
    jitters = [path.get('jitter', 0) for path in paths_info]
    losses = [path.get('loss', 0) for path in paths_info]
    
    return {
        'avg_delay': np.mean(delays) if delays else float('inf'),
        'avg_hops': np.mean(hops) if hops else float('inf'),
        'min_bandwidth': min(bandwidths) if bandwidths else 0.0,
        'avg_jitter': np.mean(jitters) if jitters else float('inf'),
        'avg_loss': np.mean(losses) if losses else 1.0
    }

def calculate_improvement_percentage(baseline_value: float, improved_value: float, 
                                   higher_is_better: bool = True) -> float:
    """
    计算改进百分比
    
    Args:
        baseline_value: 基线值
        improved_value: 改进后的值
        higher_is_better: True表示数值越高越好，False表示越低越好
        
    Returns:
        improvement_pct: 改进百分比
    """
    if baseline_value == 0:
        return 0.0
    
    if higher_is_better:
        return ((improved_value - baseline_value) / baseline_value) * 100
    else:
        return ((baseline_value - improved_value) / baseline_value) * 100

def calculate_rolling_average(data: List[float], window_size: int) -> List[float]:
    """
    计算滑动平均
    
    Args:
        data: 数据列表
        window_size: 窗口大小
        
    Returns:
        rolling_avg: 滑动平均列表
    """
    if len(data) < window_size:
        return data.copy()
    
    rolling_avg = []
    for i in range(len(data) - window_size + 1):
        window_data = data[i:i + window_size]
        avg = np.mean(window_data)
        rolling_avg.append(avg)
    
    return rolling_avg

def generate_performance_summary(results: Dict[str, Any], window_size: int = 50) -> Dict[str, Any]:
    """
    生成性能摘要报告
    
    Args:
        results: 训练结果字典
        window_size: 统计窗口大小
        
    Returns:
        summary: 性能摘要
    """
    summary = {}
    
    for variant in ['edge_aware', 'baseline']:
        if variant not in results:
            continue
            
        summary[variant] = {}
        
        for agent_type in results[variant]:
            agent_results = results[variant][agent_type]
            
            # 计算最近window_size个episode的统计
            recent_sar = calculate_sar(agent_results.get('success', []), window_size)
            recent_splat = calculate_splat(agent_results.get('splat', []), window_size)
            recent_rewards = agent_results.get('rewards', [])
            
            if recent_rewards:
                recent_avg_reward = np.mean(recent_rewards[-window_size:])
                recent_std_reward = np.std(recent_rewards[-window_size:])
            else:
                recent_avg_reward = 0.0
                recent_std_reward = 0.0
            
            summary[variant][agent_type] = {
                'sar': recent_sar,
                'splat': recent_splat,
                'avg_reward': recent_avg_reward,
                'std_reward': recent_std_reward,
                'total_episodes': len(agent_results.get('rewards', [])),
                'convergence_episode': find_convergence_point(agent_results.get('rewards', []))
            }
    
    # 计算改进幅度
    if 'edge_aware' in summary and 'baseline' in summary:
        summary['improvements'] = {}
        
        for agent_type in summary['edge_aware']:
            if agent_type in summary['baseline']:
                edge_sar = summary['edge_aware'][agent_type]['sar']
                baseline_sar = summary['baseline'][agent_type]['sar']
                sar_improvement = calculate_improvement_percentage(baseline_sar, edge_sar, True)
                
                edge_splat = summary['edge_aware'][agent_type]['splat']
                baseline_splat = summary['baseline'][agent_type]['splat']
                splat_improvement = calculate_improvement_percentage(baseline_splat, edge_splat, False)
                
                summary['improvements'][agent_type] = {
                    'sar_improvement_pct': sar_improvement,
                    'splat_improvement_pct': splat_improvement
                }
    
    return summary

def find_convergence_point(rewards: List[float], window_size: int = 20, 
                          threshold: float = 0.1) -> int:
    """
    寻找收敛点
    
    Args:
        rewards: 奖励序列
        window_size: 滑动窗口大小
        threshold: 收敛阈值（变异系数）
        
    Returns:
        convergence_episode: 收敛的episode编号，-1表示未收敛
    """
    if len(rewards) < window_size * 2:
        return -1
    
    rolling_means = calculate_rolling_average(rewards, window_size)
    rolling_stds = []
    
    for i in range(len(rewards) - window_size + 1):
        window_data = rewards[i:i + window_size]
        rolling_stds.append(np.std(window_data))
    
    # 寻找变异系数小于阈值的点
    for i in range(len(rolling_means)):
        if rolling_means[i] != 0:
            cv = rolling_stds[i] / abs(rolling_means[i])  # 变异系数
            if cv < threshold:
                return i + window_size
    
    return -1

def export_metrics_comparison(edge_aware_results: Dict, baseline_results: Dict, 
                            output_path: str):
    """
    导出指标对比表
    
    Args:
        edge_aware_results: Edge-aware结果
        baseline_results: Baseline结果
        output_path: 输出文件路径
    """
    import pandas as pd
    
    comparison_data = []
    
    for agent_type in edge_aware_results:
        if agent_type in baseline_results:
            # Edge-aware指标
            edge_sar = calculate_sar(edge_aware_results[agent_type].get('success', []))
            edge_splat = calculate_splat(edge_aware_results[agent_type].get('splat', []))
            edge_reward = np.mean(edge_aware_results[agent_type].get('rewards', [0]))
            
            # Baseline指标
            baseline_sar = calculate_sar(baseline_results[agent_type].get('success', []))
            baseline_splat = calculate_splat(baseline_results[agent_type].get('splat', []))
            baseline_reward = np.mean(baseline_results[agent_type].get('rewards', [0]))
            
            # 改进幅度
            sar_improvement = calculate_improvement_percentage(baseline_sar, edge_sar, True)
            splat_improvement = calculate_improvement_percentage(baseline_splat, edge_splat, False)
            reward_improvement = calculate_improvement_percentage(baseline_reward, edge_reward, True)
            
            comparison_data.append({
                'Algorithm': agent_type.upper(),
                'Edge_Aware_SAR': edge_sar,
                'Baseline_SAR': baseline_sar,
                'SAR_Improvement_%': sar_improvement,
                'Edge_Aware_SPLat': edge_splat,
                'Baseline_SPLat': baseline_splat,
                'SPLat_Improvement_%': splat_improvement,
                'Edge_Aware_Reward': edge_reward,
                'Baseline_Reward': baseline_reward,
                'Reward_Improvement_%': reward_improvement
            })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_path, index=False)
    print(f"📊 指标对比表已导出: {output_path}")