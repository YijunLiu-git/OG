def get_reward_function(version):
    if version == 'v1':
        from .reward_v1 import basic_reward
        return basic_reward
    elif version == 'v2_hierarchical':
        from .reward_v2_hierarchical import hierarchical_reward
        return hierarchical_reward
    else:
        raise ValueError(f"Unsupported reward version: {version}")