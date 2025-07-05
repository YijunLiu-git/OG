# utils/logger.py

import os
import csv
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List

class Logger:
    """
    训练日志记录器
    
    功能：
    1. 记录episode级别的训练统计
    2. 保存为CSV和JSON格式
    3. 支持多种指标记录
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV文件路径
        self.csv_path = os.path.join(log_dir, 'training_log.csv')
        self.json_path = os.path.join(log_dir, 'training_log.json')
        
        # 初始化CSV文件
        self.csv_headers = [
            'episode', 'timestamp', 'total_reward', 'steps', 'success', 
            'sar', 'splat', 'learning_info'
        ]
        
        # 检查CSV文件是否存在，如果不存在则创建
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
        
        # JSON日志存储
        self.json_logs = []
        
        print(f"📊 日志记录器初始化完成: {log_dir}")
    
    def log_episode(self, episode: int, episode_stats: Dict[str, Any]):
        """
        记录episode统计信息
        
        Args:
            episode: episode编号
            episode_stats: episode统计数据
        """
        timestamp = datetime.now().isoformat()
        
        # 处理JSON序列化问题：转换numpy和boolean类型
        def make_json_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        # 清理episode_stats
        clean_stats = make_json_serializable(episode_stats)
        
        # 准备CSV行数据
        csv_row = [
            episode,
            timestamp,
            clean_stats.get('total_reward', 0.0),
            clean_stats.get('steps', 0),
            clean_stats.get('success', False),
            clean_stats.get('sar', 0.0),
            clean_stats.get('splat', float('inf')),
            json.dumps(clean_stats.get('learning_info', {}))
        ]
        
        # 写入CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
        
        # 准备JSON数据
        json_entry = {
            'episode': episode,
            'timestamp': timestamp,
            'stats': clean_stats
        }
        
        self.json_logs.append(json_entry)
        
        # 写入JSON（覆盖整个文件）
        with open(self.json_path, 'w') as f:
            json.dump(self.json_logs, f, indent=2)
    
    def log_custom(self, data: Dict[str, Any], filename: str = None):
        """
        记录自定义数据
        
        Args:
            data: 要记录的数据
            filename: 自定义文件名
        """
        if filename is None:
            filename = f"custom_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """获取所有日志数据"""
        return self.json_logs.copy()
    
    def export_csv_summary(self, metrics: List[str] = None):
        """
        导出指标摘要CSV
        
        Args:
            metrics: 要导出的指标列表
        """
        if metrics is None:
            metrics = ['total_reward', 'sar', 'splat', 'success']
        
        summary_path = os.path.join(self.log_dir, 'metrics_summary.csv')
        
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = ['episode'] + metrics
            writer.writerow(headers)
            
            # 写入数据
            for log_entry in self.json_logs:
                episode = log_entry['episode']
                stats = log_entry['stats']
                
                row = [episode]
                for metric in metrics:
                    row.append(stats.get(metric, 0.0))
                
                writer.writerow(row)
        
        print(f"📈 指标摘要已导出: {summary_path}")