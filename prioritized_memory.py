import random, math, numpy as np, torch

class SumTree:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = [None] * capacity
        self.write = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = idx * 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[1]

    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(1, s)
        dataIdx = idx - self.capacity
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayMemory:
    def __init__(self, capacity: int, device, α=0.5, β0=0.4,
                 β_frames=5_000, ε=1e-6):
        self.capacity, self.device = capacity, device
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.len = 0
        self.α, self.β0, self.β_frames, self.ε = α, β0, β_frames, ε
        self.frame = 1

    def __len__(self): return self.len

    def store(self, state, action, next_state, reward, done, gamma_n):
        transition = (state.cpu(), action, next_state.cpu(), reward, done, gamma_n)
        self.tree.add(self.max_priority, transition)
        self.len = min(self.len + 1, self.capacity)

    def _beta(self):
        return self.β0 + (1.0 - self.β0) * min(1.0, self.frame / self.β_frames)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        seg = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data); idxs.append(idx); priorities.append(p)

        states, actions, next_states, rewards, dones, gammas = zip(*batch)
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1).float()
        dones = torch.tensor(dones, device=self.device).unsqueeze(1).float()
        gammas  = torch.tensor(gammas,  device=self.device).unsqueeze(1).float()

        priorities = torch.tensor(priorities, device=self.device)
        probs = priorities / self.tree.total()
        beta = self._beta(); self.frame += 1
        weights = (self.len * probs).pow(-beta)
        weights /= weights.max() + 1e-9

        return (states, actions, next_states, rewards, dones,
                gammas,idxs, weights.unsqueeze(1))

    def update_priorities(self, idxs, td_errors):
        td_errors = td_errors.detach().abs().cpu().numpy()
        td_errors = np.asarray(td_errors).flatten()
        for idx, err in zip(idxs, td_errors):
            p = (float(err) + self.ε) ** self.α
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def decay_all(self, factor: float = 0.995):
        self.tree.tree[self.capacity:] *= factor

        for i in range(self.capacity - 1, 0, -1):
            self.tree.tree[i] = (
                    self.tree.tree[i * 2] + self.tree.tree[i * 2 + 1]
            )

