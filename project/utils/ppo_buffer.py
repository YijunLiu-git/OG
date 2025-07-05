from torch_geometric.data import Batch
import torch
import random
from collections import deque

class PPORolloutBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, log_probs = zip(*samples)

        state_batch = Batch.from_data_list(states)
        next_state_batch = Batch.from_data_list(next_states)

        return (
            state_batch,
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            next_state_batch,
            torch.tensor(dones, dtype=torch.float),
            torch.stack(log_probs)
        )

    def __len__(self):
        return len(self.buffer)