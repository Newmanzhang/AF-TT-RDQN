from __future__ import annotations
import math
from random import random, randint, sample
import torch
import numpy as np
from itertools import combinations
from typing import List, Tuple, Set
Action = List[Tuple[int, Tuple[int, ...]]]   # [(node, (chs,)), …]


M=5
N=3
K=2
N_ATOMS = 51
V_MIN = -1500.0
V_MAX = 1.0
ATOMS = np.linspace(V_MIN, V_MAX, N_ATOMS, dtype=np.float32)

M_MAX, N_MAX, K_MAX = 5, 3, 2

def set_system_size(m: int, n: int, k: int) -> None:

    global M, N, K
    assert 1 <= m <= M_MAX and 1 <= n <= N_MAX and 1 <= k <= K_MAX, \
        f"require 1≤M≤{M_MAX}, 1≤N≤{N_MAX}, 1≤K≤{K_MAX}"
    M, N, K = m, n, k

CLS = ['poor','old','high','normal','medium']

CLS_PARAM = {
    'poor'  : dict(C=220, B=2, f=250, z=6.0e-28, ps= 7.0),
    'old'   : dict(C=260, B=3, f=270, z=1.0e-27, ps= 8.0),
    'high'  : dict(C=420, B=6, f=500, z=4.0e-27, ps=17.0),
    'normal': dict(C=320, B=4, f=320, z=1.5e-27, ps=11.0),
    'medium': dict(C=360, B=5, f=380, z=2.0e-27, ps=12.0),
}

ALPHA_RULE = {
    'poor'  : (0.85, 1.00),
    'old'   : (0.95, 1.00),
    'high'  : (1.5 , 3.0 ),
    'normal': (0.95, 1.8),
    'medium': (0.95, 1.5),
}

#_grid = np.linspace(0.5, 9.5, 10)
#local_location = [(float(x), float(y)) for y in _grid for x in _grid][:50]
local_location =[(-2.5,2.8),(0.5,0.8),(1.5,2.5),(3.5,2.0),(4.5,1.5)]

local_C, local_B, local_f, local_z, psend_max = [],[],[],[],[]
for c in CLS:
    p = CLS_PARAM[c]
    local_C.append(p['C'])
    local_B.append(p['B'])
    local_f.append(p['f'])
    local_z.append(p['z'])
    psend_max.append(p['ps'])

power_settings = {
    'old':    [1.5, 1.2, 1.0],
    'poor':   [2.5, 2.2, 2.0],
    'high':   [6.0, 5.5, 5.0],
    'normal': [4.0, 3.5, 3.0],
    'medium': [4.5, 4.0, 3.5]
}

local_p = np.zeros((M_MAX, N_MAX, K_MAX), dtype=np.float32)

for i, ship_class in enumerate(CLS):
    for j in range(N):
        power_val = power_settings[ship_class][j]
        local_p[i, j, :] = power_val

edge_location = [(0.0,0.0),(3.0,0.5),(-3.0,2.0)]
edge_f   = [30000,16000,8500]
edge_Nmax= [20,8,4]

edge_wavelength = np.array([

    [0.0857, 0.0852],


    [0.1667, 0.1657],


    [0.6522, 0.6383]
], dtype=np.float32)

def _slice(x, n):
    return x[:n] if isinstance(x, list) else x[:n]

def _view(var, axis=None):
    if isinstance(var, list):
        return var[:M] if axis=='M' else var[:N]
    if isinstance(var, np.ndarray):
        if axis=='M':   return var[:M]
        if axis=='N':   return var[:,:N]
        if axis=='K':   return var[:,:,:K]
    return var

local_C       = _slice(local_C, M_MAX)
local_B       = _slice(local_B, M_MAX)
local_f       = _slice(local_f, M_MAX)
local_z       = _slice(local_z, M_MAX)
psend_max     = _slice(psend_max, M_MAX)
local_location= _slice(local_location, M_MAX)

edge_location = _slice(edge_location, N_MAX)
edge_f        = _slice(edge_f, N_MAX)
edge_Nmax     = _slice(edge_Nmax, N_MAX)

local_p       = local_p[:M_MAX,:N_MAX,:K_MAX]
edge_wavelength = edge_wavelength[:N_MAX,:K_MAX]

def get_local_g():
    lambda_rates = np.random.uniform(0.2, 0.6, M)
    g = [min(np.random.poisson(lam * 10), 10) for lam in lambda_rates]
    return g

local_g=get_local_g()

def get_tlimit(local_g: list[int]) -> list[float]:

    tlimit = []
    for i, cls in enumerate(CLS[:M]):
        p     = CLS_PARAM[cls]
        t0    = local_g[i] * (p['C'] / p['f'])
        lo,hi = ALPHA_RULE[cls]
        alpha = lo if lo == hi else np.random.uniform(lo, hi)
        tlimit.append(round(alpha * t0, 2))
    return tlimit

def generate_rician_h_batch():
    KR=5

    real = np.random.normal(0, np.sqrt(0.5), size=(M, N, K))
    imag = np.random.normal(0, np.sqrt(0.5), size=(M, N, K))
    s = real + 1j * imag

    h = np.sqrt(KR / (KR + 1)) + np.sqrt(1 / (KR + 1)) * s

    return h

h_ijk=generate_rician_h_batch()


def enumerate_actions() -> List[Action]:

    phys: Set[Tuple[int, int]] = {(n, k)
                                  for n in range(1, N+1)
                                  for k in range(1, K+1)}
    actions: List[Action] = []

    def backtrack(i: int, cur: Action, free_set: Set[Tuple[int, int]]):
        if i == M:
            actions.append(cur.copy())
            return

        cur.append((0, ()))
        backtrack(i+1, cur, free_set)
        cur.pop()


        for node in range(1, N+1):
            avail = [ch for (n, ch) in free_set if n == node]
            if not avail:
                continue
            for r in range(1, len(avail)+1):
                for subset in combinations(avail, r):
                    for ch in subset:
                        free_set.remove((node, ch))
                    cur.append((node, subset))
                    backtrack(i+1, cur, free_set)
                    cur.pop()
                    for ch in subset:
                        free_set.add((node, ch))

    backtrack(0, [], phys.copy())
    return actions


def sample_random_action(max_ch_per_ship: int = K) -> Action:

    free: Set[Tuple[int, int]] = {(n, k) for n in range(1, N+1) for k in range(1, K+1)}
    act: Action = []

    for _ in range(M):

        if random() < 0.3:
            act.append((0, ()))
            continue


        node = randint(1, N)
        avail_ch = [ch for (n, ch) in free if n == node]
        if not avail_ch:
            act.append((0, ()))
            continue


        r = randint(1, min(len(avail_ch), max_ch_per_ship))
        chosen = sample(avail_ch, r)
        for ch in chosen:
            free.remove((node, ch))
        act.append((node, tuple(sorted(chosen))))

    return act
#acts = generate_actions()

def parse_action_to_edge_status(act: Action,
                                local_g: list,
                                local_C: list,
                                edge_Nmax: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    M = len(local_g)
    N = len(edge_Nmax)

    edge_g = np.zeros(N, dtype=np.float32)
    task_list = [[] for _ in range(N)]

    for i in range(M):
        node_id, channels = act[i]
        if node_id == 0:
            continue
        node_idx = node_id - 1
        edge_g[node_idx] += local_g[i]
        task_list[node_idx].extend([local_C[i]] * local_g[i])

    edge_nexe = np.zeros(N, dtype=np.float32)
    edge_cave = np.zeros(N, dtype=np.float32)

    for j in range(N):
        total_tasks = len(task_list[j])
        if total_tasks == 0:
            continue
        nexe = min(total_tasks, edge_Nmax[j])
        edge_nexe[j] = nexe
        executing_tasks = task_list[j][:int(nexe)]
        if len(executing_tasks) > 0:
            edge_cave[j] = np.mean(executing_tasks)

    return edge_g, edge_nexe, edge_cave

tlimit_init = get_tlimit(local_g)
class StateBuilder:
    def __init__(self):
        self.max_task_arrival     = max(3, int(np.max(local_g) * 1.2))
        self.max_task_compute = 500
        self.max_task_data = 10
        self.max_freq_local = 500
        self.max_energy_coef = 4.0e-27
        self.max_location = 5
        self.max_time_limit       = max(8, int(np.max(tlimit_init)  * 1.2))

        self.max_freq_edge = max(edge_f[:N])
        self.max_parallel = max(edge_Nmax[:N])
        relax = 1.2  # 20% safety margin
        self.max_tasks_arrival_edge = max(
            1,
            math.ceil(self.max_task_arrival * M / N * relax)
        )
        self.max_avg_compute_edge = self.max_task_compute

        self.max_wavelength = 0.7
        self.max_channel_gain = 5.0
        self.max_psend_max = max(psend_max)
        self.max_power = np.max(local_p[:M, :N, :K])
        self.rho_max = 1.5

    def build_state(self,
                                 local_location, local_C, local_B, local_f, local_g, local_z, tlimit,psend_max,
                                 edge_location, edge_f, edge_Nmax,
                                 edge_wavelength,
                                 h_ijk,local_p,rho_j_prev,edge_cave_j_prev):

        ships_feat = []
        for i in range(M):
            g = local_g[i] / self.max_task_arrival
            c = local_C[i] / self.max_task_compute
            b = local_B[i] / self.max_task_data
            f = local_f[i] / self.max_freq_local
            z = local_z[i] / self.max_energy_coef
            x, y = local_location[i][0]/self.max_location, local_location[i][1]/self.max_location
            t = tlimit[i] / self.max_time_limit
            p = psend_max[i] / self.max_psend_max
            ships_feat.extend([g, c, b, f, z, x, y, t,p])

        ships_feat = torch.tensor(ships_feat, dtype=torch.float32).view(1, -1)


        edges_feat = []
        for j in range(N):
            f_edge = edge_f[j] / self.max_freq_edge
            x_edge, y_edge = edge_location[j][0]/self.max_location, edge_location[j][1]/self.max_location
            Nmax = edge_Nmax[j] / self.max_parallel
            rho_prev =  min(rho_j_prev[j] / self.rho_max, 1.0)
            edge_cave_prev=edge_cave_j_prev[j]/ self.max_avg_compute_edge
            edges_feat.extend([f_edge, x_edge, y_edge, Nmax, rho_prev, edge_cave_prev])

        edges_feat = torch.tensor(edges_feat, dtype=torch.float32).view(1, -1)


        wave_feat = torch.tensor(edge_wavelength.flatten(), dtype=torch.float32).view(1, -1)
        wave_feat = wave_feat / self.max_wavelength


        h_abs_square = np.abs(h_ijk)**2
        h_feat = torch.tensor(h_abs_square.flatten(), dtype=torch.float32).view(1, -1)
        h_feat = torch.clamp(h_feat / self.max_channel_gain, 0, 1)

        power_feat = (local_p / self.max_power).astype(np.float32).flatten()
        power_feat = torch.tensor(power_feat).view(1, -1)

        state_vec = torch.cat([ships_feat, edges_feat, wave_feat, h_feat,power_feat], dim=1)

        return state_vec





