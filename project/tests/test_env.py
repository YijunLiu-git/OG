# tests/test_env.py

from env.vnf_env import VNFEnv
from env.topology_loader import load_topology
from models.gnn_encoder import GNNEncoder

if __name__ == "__main__":
    G, edge_attr = load_topology("data/topology/sample_topo.json")
    env = VNFEnv(G, edge_attr)
    print("Environment initialized.")
    print(f"State dim: {env.state_dim}, Edge dim: {env.edge_dim}, Action dim: {env.action_dim}")
    obs = env.reset()
    print("Initial observation:", obs)
