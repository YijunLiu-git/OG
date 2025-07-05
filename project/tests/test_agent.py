# tests/test_agent.py

from agents.ddqn_agent import DDQNAgent
from env.vnf_env import VNFEnv
from env.topology_loader import load_topology
from models.gnn_encoder import GNNEncoder
import torch

if __name__ == "__main__":
    from config.config import config
    G, edge_attr = load_topology("data/topology/sample_topo.json")
    env = VNFEnv(G, edge_attr)

    agent = DDQNAgent(env.state_dim, env.action_dim, config)
    state = env.reset()
    action = agent.select_action(state)
    print("Sample action:", action)
    next_state, reward, done, _ = env.step(action)
    agent.store_transition(state, action, reward, next_state, done)
    agent.learn()
    print("Agent training step completed.")
