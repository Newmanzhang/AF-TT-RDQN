import random
import torch

class ReplayMemory:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done, gamma_n):
        transition = (state.cpu(), action, next_state.cpu(), reward, done, gamma_n)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones, gammas = zip(*samples)

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1).float()
        dones = torch.tensor(dones, device=self.device).unsqueeze(1).float()
        gammas  = torch.tensor(gammas,  device=self.device).unsqueeze(1).float()

        return states, actions, next_states, rewards, dones, gammas

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, idxs, td_errors):
        pass

    def decay_all(self, factor):
        pass
