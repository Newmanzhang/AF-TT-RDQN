from functools import lru_cache
from typing import Tuple, List
import math
import torch
import torch.nn as nn
import numpy as np
from replaymemory import ReplayMemory
from prioritized_memory import PrioritizedReplayMemory
from state_action_qnet import StateActionQNet
from tokenizer import action2tokens_and_mask
from itertools import combinations

N_STEP = 3


class QTrainbowDQN:
    def __init__(self, train_mode, device, NUM_STATES, vocab_size, M, token_len, MEMORY_CAPACITY, LR, BATCH_SIZE, GAMMA,
                 N, K, use_per: bool = True, per_alpha=0.5, per_beta0=0.4, per_beta_frames=20000):
        super(QTrainbowDQN, self).__init__()
        self.device = device
        self.M, self.N, self.K = M, N, K
        self.n_atoms = 51
        self.v_min = -1500.0
        self.v_max = 1.0
        self.support  = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = float(self.v_max - self.v_min) / (self.n_atoms - 1)
        self.eval_net, self.target_net = StateActionQNet(NUM_STATES, vocab_size, M, N, token_len, self.device,n_atoms=self.n_atoms).to(
            self.device), StateActionQNet(NUM_STATES, vocab_size, M, N, token_len, self.device,n_atoms=self.n_atoms).to(self.device)
        self.target_net.eval()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.LR = LR
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma = 0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20000, eta_min=3e-5)
        self.loss_func = nn.MSELoss()
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.flag = 0
        self.epsilon = 0.0

        if use_per:
            self.memory = PrioritizedReplayMemory(
                capacity=MEMORY_CAPACITY, device=self.device,
                α=per_alpha, β0=per_beta0, β_frames=per_beta_frames
            )
            self.use_per = True
        else:
            self.memory = ReplayMemory(capacity=MEMORY_CAPACITY, device=self.device)
            self.use_per = False

        self._init_action_cache()
        if not hasattr(self.target_net, "reset_noise"):
            self.target_net.reset_noise = lambda: None

        if not train_mode:
            self.eval_net.eval()

    def _init_action_cache(self):
        self.node_channel_combinations = {}
        for node in range(1, self.N + 1):
            self.node_channel_combinations[node] = []
            available_channels = list(range(1, self.K + 1))
            for r in range(1, len(available_channels) + 1):
                for subset in combinations(available_channels, r):
                    self.node_channel_combinations[node].append(subset)

        self.base_candidates = [(0, ())]  # local offloading
        for node in range(1, self.N + 1):
            for channels in self.node_channel_combinations[node]:
                self.base_candidates.append((node, channels))

    @lru_cache(maxsize=1024)
    def _get_available_candidates(self, used_channels_tuple: Tuple):

        used_channels = set(used_channels_tuple)
        available_candidates = [(0, ())]  # local always available

        for node in range(1, self.N + 1):
            for channels in self.node_channel_combinations[node]:

                conflict = any((node, ch) in used_channels for ch in channels)
                if not conflict:
                    available_candidates.append((node, channels))

        return available_candidates

    def update_target_net(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

    def update_epsilon(self, *, global_step: int):
        return

    def update_LR(self):
        self.scheduler.step()

    def get_weights(self):
        return {k: v.cpu() for k, v in self.eval_net.state_dict().items()}

    def set_weights(self, state_dict):
        self.eval_net.load_state_dict(state_dict, strict=False)
        self.target_net.load_state_dict(state_dict, strict=False)
        self.target_net.eval()

    def choose_action(self, state, beam_width: int = 5):
        if hasattr(self.eval_net, "reset_noise"):
            self.eval_net.reset_noise()
        return self.beam_search_selection(state, beam_width=beam_width)

    def _greedy_selection_single(self, state):
        self.eval_net.eval()
        with torch.no_grad():
            state = state.to(self.device)
            best_action, used = [], set()
            for task_idx in range(self.M):
                used_tuple = tuple(sorted(list(used)))
                candidates = self._get_available_candidates(used_tuple)
                if not candidates:
                    candidates = [(0, ())]

                candidate_actions = []
                for candidate in candidates:
                    temp_action = best_action + [candidate] + [(0, ())] * (self.M - len(best_action) - 1)
                    candidate_actions.append(temp_action)

                tok_li, msk_li = zip(*(action2tokens_and_mask(action) for action in candidate_actions))
                tok = torch.from_numpy(np.array(tok_li)).long().to(self.device)
                msk = torch.from_numpy(np.array(msk_li)).bool().to(self.device)

                st = state.expand(tok.shape[0], -1)

                q_logits = self.eval_net(st, tok, msk)
                q_values = (torch.softmax(q_logits, dim=-1) * self.support).sum(dim=-1)

                idx = q_values.argmax().item()
                best_sub = candidates[idx]

                best_action.append(best_sub)
                if best_sub[0] > 0:
                    for ch in best_sub[1]:
                        used.add((best_sub[0], ch))
        self.eval_net.train()
        return best_action

    def _batch_greedy_selection(self, state_batch: torch.Tensor) -> List[List[Tuple[int, Tuple[int, ...]]]]:

        self.eval_net.eval()
        with torch.no_grad():
            batch_size = state_batch.size(0)


            final_actions = [[] for _ in range(batch_size)]
            used_channels_batch = [set() for _ in range(batch_size)]


            for task_idx in range(self.M):

                mega_batch_tokens = []
                mega_batch_masks = []
                mega_batch_states_indices = []
                candidates_per_sample = []

                for i in range(batch_size):
                    used_tuple = tuple(sorted(list(used_channels_batch[i])))
                    candidates = self._get_available_candidates(used_tuple)
                    if not candidates:
                        candidates = [(0, ())]

                    candidates_per_sample.append(candidates)

                    for candidate in candidates:

                        temp_action = final_actions[i] + [candidate] + [(0, ())] * (self.M - task_idx - 1)
                        tokens, mask = action2tokens_and_mask(temp_action)
                        mega_batch_tokens.append(tokens)
                        mega_batch_masks.append(mask)
                        mega_batch_states_indices.append(i)


                if not mega_batch_tokens:
                    break


                tok_tensor = torch.from_numpy(np.array(mega_batch_tokens)).long().to(self.device)
                msk_tensor = torch.from_numpy(np.array(mega_batch_masks)).bool().to(self.device)


                st_tensor = state_batch[mega_batch_states_indices]


                q_logits = self.eval_net(st_tensor, tok_tensor, msk_tensor)
                q_values = (torch.softmax(q_logits, dim=-1) * self.support).sum(dim=-1)


                current_q_idx = 0
                for i in range(batch_size):
                    candidates = candidates_per_sample[i]
                    num_candidates = len(candidates)


                    sample_q_values = q_values[current_q_idx: current_q_idx + num_candidates]


                    best_candidate_idx = sample_q_values.argmax().item()
                    best_sub_action = candidates[best_candidate_idx]


                    final_actions[i].append(best_sub_action)
                    if best_sub_action[0] > 0:
                        for ch in best_sub_action[1]:
                            used_channels_batch[i].add((best_sub_action[0], ch))

                    current_q_idx += num_candidates

        self.eval_net.train()
        return final_actions

    def beam_search_selection(self, state: torch.Tensor, beam_width: int = 5):

        self.eval_net.eval()
        with torch.no_grad():
            state = state.to(self.device)
            beams = [(score, [], set()) for score in [0.0]]
            for task_idx in range(self.M):
                all_candidates_with_scores = []

                for _, prev_seq, prev_used in beams:
                    used_tuple = tuple(sorted(list(prev_used)))
                    available_subs = self._get_available_candidates(used_tuple) or [(0, ())]
                    candidate_actions_for_eval = []
                    temp_info = []
                    for sub_action in available_subs:
                        full_action = prev_seq + [sub_action] + [(0, ())] * (self.M - task_idx - 1)
                        candidate_actions_for_eval.append(full_action)
                        temp_info.append({'prev_seq': prev_seq, 'prev_used': prev_used, 'sub_action': sub_action})

                    if not candidate_actions_for_eval:
                        continue

                    tok_li, msk_li = zip(*(action2tokens_and_mask(a) for a in candidate_actions_for_eval))
                    tok = torch.from_numpy(np.array(tok_li)).long().to(self.device)
                    msk = torch.from_numpy(np.array(msk_li)).bool().to(self.device)
                    st = state.expand(tok.size(0), -1)

                    q_logits = self.eval_net(st, tok, msk)
                    q_values = (torch.softmax(q_logits, dim=-1) * self.support).sum(dim=-1)
                    for i, q_value in enumerate(q_values):
                        info = temp_info[i]
                        score = q_value.item()
                        new_seq = info['prev_seq'] + [info['sub_action']]
                        new_used = info['prev_used'].copy()
                        if info['sub_action'][0] > 0:
                            new_used.update((info['sub_action'][0], ch) for ch in info['sub_action'][1])
                        all_candidates_with_scores.append((score, new_seq, new_used))

                if not all_candidates_with_scores:
                    break

                all_candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates_with_scores[:beam_width]

            if not beams:
                best_seq = []
            else:
                best_seq = beams[0][1]

            best_seq += [(0, ())] * (self.M - len(best_seq))

        self.eval_net.train()
        return best_seq

    def exhaustive_search_selection(self, state: torch.Tensor):
        
        self.eval_net.eval()
        with torch.no_grad():
            state_tensor = state.to(self.device)  

            @lru_cache(maxsize=None)
            def solve(task_idx: int, used_channels_tuple: Tuple) -> Tuple[float, list]:
                
                if task_idx == self.M:
                    return 0.0, []

                best_q_so_far = -float('inf')
                best_action_suffix = []  

                used_channels_set = set(used_channels_tuple)
                available_sub_actions = self._get_available_candidates(used_channels_tuple)
                if not available_sub_actions:
                    available_sub_actions = [(0, ())]

                for sub_action in available_sub_actions:

                    new_used_set = used_channels_set.copy()
                    if sub_action[0] > 0:
                        new_used_set.update((sub_action[0], ch) for ch in sub_action[1])


                    new_used_tuple = tuple(sorted(list(new_used_set)))


                    q_from_future, actions_from_future = solve(task_idx + 1, new_used_tuple)


                    current_full_action = [sub_action] + actions_from_future

                    current_full_action += [(0, ())] * (self.M - len(current_full_action))

                   
                    tok, msk = action2tokens_and_mask(current_full_action)
                    tok_tensor = torch.tensor([tok], dtype=torch.long, device=self.device)
                    msk_tensor = torch.tensor([msk], dtype=torch.bool, device=self.device)

                    q_logits = self.eval_net(state_tensor, tok_tensor, msk_tensor)
                    current_q = (torch.softmax(q_logits, dim=-1) * self.support).sum().item()

                    
                    if current_q > best_q_so_far:
                        best_q_so_far = current_q
                        best_action_suffix = [sub_action] + actions_from_future

                return best_q_so_far, best_action_suffix

           
            final_q, final_action = solve(0, tuple())

            solve.cache_clear()

            if not final_action:
                final_action = [(0, ())] * self.M

            final_action += [(0, ())] * (self.M - len(final_action))

        self.eval_net.train()
        return final_action

    def _random_action_selection(self):
        action = []
        used_channels = set()
        for _ in range(self.M):
            node = np.random.randint(0, self.N + 1)
            if node == 0:
                chs = ()
            else:
                available_chs = [k for k in range(1, self.K + 1)
                                 if (node, k) not in used_channels]
                if not available_chs:
                    node = 0
                    chs = ()
                else:
                    num_chs = np.random.randint(1, len(available_chs) + 1)
                    chs = tuple(np.random.choice(available_chs, size=num_chs, replace=False))
            action.append((node, chs))
            if node > 0:
                for ch in chs:
                    used_channels.add((node, ch))
        return action

    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None
        self.eval_net.train()
        self.learn_step_counter += 1
        if self.use_per:
            (state_batch, actions, next_state_batch,
             rewards, dones, gammas, idxs, weights) = self.memory.sample(self.BATCH_SIZE)
        else:
            state_batch, actions, next_state_batch, rewards, dones, gammas = \
                self.memory.sample(self.BATCH_SIZE)
            weights = torch.ones_like(rewards, device=self.device)
            idxs = None
        if hasattr(self.eval_net, "reset_noise"):
            self.eval_net.reset_noise()

        token_batch, mask_batch = self._batch_tokenize_actions(actions)

        logits = self.eval_net(state_batch, token_batch, mask_batch)
        log_prob = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():


            next_actions = self._batch_greedy_selection(next_state_batch)
            tok_n, msk_n = self._batch_tokenize_actions(next_actions)

            next_logits = self.target_net(next_state_batch, tok_n, msk_n)
            next_prob = torch.softmax(next_logits, dim=-1)

            rewards = rewards.squeeze(-1)
            dones = dones.squeeze(-1)
            gammas = gammas.squeeze(-1)
            proj_target = self._project_distribution(next_prob, rewards, dones, gammas)

        loss = - (proj_target * log_prob).sum(dim=-1)
        loss = (weights.squeeze(-1) * loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        if self.use_per:
            td_errors = loss.detach().clone()
            self.memory.update_priorities(idxs, td_errors)
        grad_norm = 0.0
        for p in self.eval_net.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(grad_norm)

        return {
            "loss": loss.item(),
            "td_mean": (log_prob.exp().cpu() - proj_target.cpu()).abs().sum(dim=-1).mean().item(),
            "td_max":  (log_prob.exp().cpu() - proj_target.cpu()).abs().sum(dim=-1).max().item(),
            "grad_norm": grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _project_distribution(self, next_prob, rewards, dones, gammas):
        batch_size = rewards.shape[0]
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        gammas = gammas.unsqueeze(1)
        t_z = rewards + (1 - dones) * gammas * self.support.unsqueeze(0)
        t_z = t_z.clamp(min=self.v_min, max=self.v_max)
        b = (t_z - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        u_eq_l = (u == l)
        u_p = (b - l.float())
        l_p = (u.float() - b)
        u_p[u_eq_l] = 0.0
        l_p[u_eq_l] = 1.0
        proj_dist = torch.zeros_like(next_prob)
        offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size, device=self.device).long().unsqueeze(1)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_prob * l_p).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_prob * u_p).view(-1))
        return proj_dist

    def _batch_tokenize_actions(self, actions):

        token_batch, mask_batch = [], []
        for a in actions:
            t, m = action2tokens_and_mask(a)
            token_batch.append(t)
            mask_batch.append(m)
        token_batch = torch.from_numpy(np.array(token_batch)).long().to(self.device)
        mask_batch = torch.from_numpy(np.array(mask_batch)).bool().to(self.device)
        return token_batch, mask_batch

    def save_model(self, filename):
        self.eval_net.save_model(filename)

    def load_model(self, filename):
        self.eval_net.load_model(filename=filename, device=self.device)





