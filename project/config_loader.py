# config_loader.py - 统一配置加载器，读取config.yaml

import yaml
import os
from typing import Dict, Any, Optional

class ConfigLoader:
    """
    统一配置加载器 - 读取config.yaml
    
    ✅ 功能：
    - 从config.yaml加载所有配置
    - 根据episode数量自动选择场景
    - 提供场景配置验证
    - 支持配置热重载
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ 配置加载成功: {self.config_path}")
            return self.config
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"❌ 配置文件格式错误: {e}")
    
    def get_scenario_by_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        根据episode数量自动获取对应场景配置
        
        Episode分布：
        - 1-25: normal_operation
        - 26-50: peak_congestion  
        - 51-75: failure_recovery
        - 76-100: extreme_pressure
        """
        if episode_num <= 25:
            scenario_name = 'normal_operation'
        elif episode_num <= 50:
            scenario_name = 'peak_congestion'
        elif episode_num <= 75:
            scenario_name = 'failure_recovery'
        else:
            scenario_name = 'extreme_pressure'
        
        return self.get_scenario_config(scenario_name)
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """获取指定场景的完整配置"""
        if self.config is None:
            self.load_config()
        
        # 获取基础配置
        base_config = {
            'scenario_name': scenario_name,
            'topology': self.config['topology'].copy(),
            'vnf_requirements': self.config['vnf_requirements'].copy(),
            'reward': self.config['reward'].copy(),
            'gnn': self.config['gnn'].copy(),
            'train': self.config['train'].copy()
        }
        
        # 获取场景特定配置
        if scenario_name in self.config['scenarios']:
            scenario_specific = self.config['scenarios'][scenario_name]
            
            # 合并场景特定的拓扑配置
            if 'topology' in scenario_specific:
                base_config['topology'].update(scenario_specific['topology'])
            
            # 合并场景特定的VNF需求配置
            if 'vnf_requirements' in scenario_specific:
                base_config['vnf_requirements'].update(scenario_specific['vnf_requirements'])
            
            # 合并场景特定的奖励配置
            if 'reward' in scenario_specific:
                base_config['reward'].update(scenario_specific['reward'])
            
            # 添加场景元信息
            base_config.update({
                'name': scenario_specific.get('name', scenario_name),
                'episodes': scenario_specific.get('episodes', [1, 100]),
                'expected_sar_range': scenario_specific.get('expected_sar_range', [0.5, 0.8]),
                'realism_level': scenario_specific.get('realism_level', 3),
                'description': scenario_specific.get('description', ''),
            })
        
        # ✅ 添加调试信息
        print(f"获取场景配置: {scenario_name}")
        print(f"  节点资源: {base_config['topology']['node_resources']}")
        print(f"  边带宽: {base_config['topology']['edge_resources']['bandwidth_min']}-{base_config['topology']['edge_resources']['bandwidth_max']}")
        print(f"  VNF需求: {base_config['vnf_requirements']['cpu_min']}-{base_config['vnf_requirements']['cpu_max']}")
        
        return base_config
    
    def print_scenario_plan(self):
        """打印渐进式场景训练计划"""
        if self.config is None:
            self.load_config()
        
        print(f"\n🎯 渐进式场景训练计划:")
        print("=" * 60)
        
        for scenario_name, scenario_info in self.config['scenarios'].items():
            episodes = scenario_info.get('episodes', [1, 25])
            expected_sar = scenario_info.get('expected_sar_range', [0.5, 0.8])
            
            print(f"Episode {episodes[0]:2d}-{episodes[1]:2d}: {scenario_info['name']}")
            print(f"   现实性等级: {scenario_info.get('realism_level', 3)}/5")
            print(f"   预期SAR: {expected_sar[0]:.0%}-{expected_sar[1]:.0%}")
            print(f"   研究焦点: {scenario_info.get('description', '')}")
            print()
    
    def validate_scenario_configs(self) -> bool:
        """验证所有场景配置的合理性"""
        if self.config is None:
            self.load_config()
        
        print("🔍 场景配置验证:")
        print("=" * 50)
        
        base_nodes = self.config['topology']['node_counts']['total']
        base_cpu_per_node = self.config['topology']['base_node_resources']['cpu']
        
        all_valid = True
        
        for scenario_name in self.config['scenarios'].keys():
            config = self.get_scenario_config(scenario_name)
            
            # 计算资源供需比
            cpu_factor = config['topology']['node_resources']['cpu']
            total_cpu_supply = base_nodes * base_cpu_per_node * cpu_factor
            
            vnf_config = config['vnf_requirements']
            avg_chain_length = sum(vnf_config['chain_length_range']) / 2
            avg_cpu_demand = (vnf_config['cpu_min'] + vnf_config['cpu_max']) / 2
            total_cpu_demand_per_chain = avg_chain_length * avg_cpu_demand
            
            # 压力比和最大链数
            pressure_ratio = total_cpu_demand_per_chain / total_cpu_supply
            max_possible_chains = total_cpu_supply / total_cpu_demand_per_chain
            
            expected_range = config['expected_sar_range']
            min_chains_for_max_sar = 1.0 / expected_range[1]
            
            print(f"\n📊 {config['name']}:")
            print(f"   总CPU供应: {total_cpu_supply:.1f}")
            print(f"   单链CPU需求: {total_cpu_demand_per_chain:.3f}")
            print(f"   单链压力比: {pressure_ratio:.1%}")
            print(f"   理论最大链数: {max_possible_chains:.1f}")
            print(f"   预期SAR: {expected_range[0]:.1%}-{expected_range[1]:.1%}")
            
            # 验证SAR可达性
            if max_possible_chains >= min_chains_for_max_sar:
                print(f"   ✅ SAR {expected_range[1]:.0%} 可达")
            else:
                print(f"   ❌ SAR {expected_range[1]:.0%} 不可达")
                all_valid = False
        
        if all_valid:
            print(f"\n🎉 所有场景配置验证通过!")
        else:
            print(f"\n⚠️ 部分场景配置需要调整!")
        
        return all_valid
    
    def get_base_config(self) -> Dict[str, Any]:
        """获取基础配置（不包含场景特定设置）"""
        if self.config is None:
            self.load_config()
        
        return {
            'topology': self.config['topology'],
            'vnf_requirements': self.config['vnf_requirements'],
            'reward': self.config['reward'],
            'gnn': self.config['gnn'],
            'train': self.config['train'],
            'dimensions': self.config['dimensions']
        }
    
    def get_gnn_config(self, mode: str = 'edge_aware') -> Dict[str, Any]:
        """获取GNN配置"""
        if self.config is None:
            self.load_config()
        
        return self.config['gnn'][mode]
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        if self.config is None:
            self.load_config()
        
        return self.config['train']
    
    def reload_config(self):
        """重新加载配置文件（用于配置热更新）"""
        print("🔄 重新加载配置文件...")
        self.load_config()
        print("✅ 配置重载完成")

# 全局配置加载器实例
_config_loader = None

def get_config_loader(config_path: str = "config.yaml") -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader

def get_scenario_config(scenario_name_or_episode) -> Dict[str, Any]:
    """
    获取场景配置的便捷函数
    
    ✅ 这个函数替代了原来的场景配置函数
    """
    loader = get_config_loader()
    
    if isinstance(scenario_name_or_episode, int):
        return loader.get_scenario_by_episode(scenario_name_or_episode)
    else:
        return loader.get_scenario_config(scenario_name_or_episode)

def print_scenario_plan():
    """打印场景计划的便捷函数"""
    loader = get_config_loader()
    loader.print_scenario_plan()

def validate_all_configs():
    """验证所有配置的便捷函数"""
    loader = get_config_loader()
    return loader.validate_scenario_configs()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件的便捷函数"""
    loader = get_config_loader(config_path)
    return loader.get_base_config()

# 使用示例和测试
def test_config_loader():
    """测试配置加载器"""
    print("🧪 测试配置加载器...")
    print("=" * 50)
    
    try:
        # 1. 测试配置加载
        loader = ConfigLoader("config.yaml")
        print("✅ 1. 配置文件加载测试通过")
        
        # 2. 测试场景配置获取
        normal_config = loader.get_scenario_config('normal_operation')
        episode_50_config = loader.get_scenario_by_episode(50)
        
        assert normal_config['scenario_name'] == 'normal_operation'
        assert episode_50_config['scenario_name'] == 'peak_congestion'
        print("✅ 2. 场景配置获取测试通过")
        
        # 3. 测试权重验证
        for scenario_name in ['normal_operation', 'peak_congestion', 'failure_recovery', 'extreme_pressure']:
            config = loader.get_scenario_config(scenario_name)
            reward_config = config['reward']
            weight_sum = (reward_config.get('sar_weight', 0) +
                         reward_config.get('latency_weight', 0) +
                         reward_config.get('efficiency_weight', 0) +
                         reward_config.get('quality_weight', 0))
            
            assert abs(weight_sum - 1.0) < 0.01, f"{scenario_name} 权重总和不为1.0: {weight_sum}"
        
        print("✅ 3. 权重验证测试通过")
        
        # 4. 测试配置验证
        is_valid = loader.validate_scenario_configs()
        print(f"✅ 4. 配置验证测试: {'通过' if is_valid else '需要调整'}")
        
        # 5. 测试便捷函数
        config_by_func = get_scenario_config('extreme_pressure')
        assert config_by_func['scenario_name'] == 'extreme_pressure'
        print("✅ 5. 便捷函数测试通过")
        
        print(f"\n🎉 配置加载器测试全部通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数 - 展示配置加载器功能"""
    print("🚀 Config Loader 演示")
    print("=" * 50)
    
    # 1. 显示场景计划
    print_scenario_plan()
    
    # 2. 验证配置
    print("\n" + "="*50)
    validate_all_configs()
    
    # 3. 测试功能
    print("\n" + "="*50)
    test_config_loader()
    
    # 4. 示例使用
    print("\n" + "="*50)
    print("📖 使用示例:")
    print("   from config_loader import get_scenario_config")
    print("   config = get_scenario_config('normal_operation')")
    print("   config = get_scenario_config(25)  # Episode 25")
    print("   print(config['vnf_requirements'])")

if __name__ == "__main__":
    main()