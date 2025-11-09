import numpy as np
import torch
import copy
from constants import StateBuilder,get_local_g,get_tlimit,generate_rician_h_batch,parse_action_to_edge_status
from typing import Dict, Any, Tuple, List

DEFAULT_CONFIG = {
    "bw": 10e6,
    "noise_power_dbm": -100,
    "reward_alpha": 0.5,
    "reward_beta": 0.5,
    "violation_penalty": -20.0,
    "max_steps": 500,
    "task_dynamics": True,
    "channel_dynamics": True,
    "norm_T_min": 4.0,
    "norm_T_max": 40.0,
    "norm_E_min": 0.5,
    "norm_E_max": 50.0,
}

class Environment:
    def __init__(self, M: int, N: int, K: int,local_location: List, local_C: List, local_B: List, local_f: List, local_z: List, psend_max: List,edge_location: List, edge_f: List, edge_Nmax: List, edge_wavelength: np.ndarray, local_p: np.ndarray,config: Dict[str, Any] = None):

        self.M=M
        self.N=N
        self.K=K
        self.local_location = local_location
        self.local_C = local_C
        self.local_B = local_B
        self.local_f = local_f
        self.local_z = local_z
        self.local_p = local_p
        self.psend_max=psend_max
        self.edge_location = edge_location
        self.edge_f = edge_f
        self.edge_Nmax = edge_Nmax
        self.edge_wavelength = edge_wavelength
        self.local_g: List[int] = []
        self.tlimit: List[float] = []
        self.h_ijk: np.ndarray = np.array([])
        self.rho_j_prev: np.ndarray = np.zeros(self.N)
        self.edge_cave_prev: np.ndarray = np.zeros(self.N)
        self.step_cnt: int = 0

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.sigma2 = 10 ** (self.config['noise_power_dbm'] / 10) * 1e-3
        self.builder = StateBuilder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _distance(self, i, j) -> float:
        sx, sy = self.local_location[i]
        ex, ey = self.edge_location[j]
        return 1000 * np.linalg.norm(np.array([sx, sy]) - np.array([ex, ey]))

    def reset(self) -> torch.Tensor:
        self.step_cnt = 0
        self.local_g = get_local_g()
        self.tlimit = get_tlimit(self.local_g)
        self.h_ijk = generate_rician_h_batch()
        self.rho_j_prev = np.zeros(self.N)
        self.edge_cave_prev = np.zeros(self.N)

        self.builder.max_task_arrival = max(3, int(np.max(self.local_g) * 1.2))
        self.builder.max_time_limit = max(8, int(np.max(self.tlimit) * 1.2))

        init_state = self.builder.build_state(
            self.local_location, self.local_C, self.local_B,
            self.local_f, self.local_g, self.local_z, self.tlimit,self.psend_max,
            self.edge_location, self.edge_f, self.edge_Nmax,
            self.edge_wavelength,
            self.h_ijk,self.local_p,
            self.rho_j_prev, self.edge_cave_prev
        )
        return init_state


    def step(self, action: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[torch.Tensor, float, bool, dict]:
        self.step_cnt += 1
        total_delay, total_energy = 0.0, 0.0
        violated = False
        edge_g, edge_nexe, edge_cave = parse_action_to_edge_status(action, self.local_g, self.local_C, self.edge_Nmax)

        for i in range(self.M):
            node_id, chs = action[i]


            if node_id == 0:
                comp_time = self.local_g[i]*self.local_C[i] / self.local_f[i]
                if comp_time > self.tlimit[i]:
                    violated = True
                    break

                comp_energy = self.local_g[i]*self.local_z[i] * ((self.local_f[i]*1e6) ** 2) * (self.local_C[i]*1e6)
                total_delay += comp_time
                total_energy += comp_energy

            else:

                j = node_id - 1
                total_P = sum(self.local_p[i, j, ch - 1] for ch in chs)
                if total_P > self.psend_max[i]:
                    violated = True
                    break

                R_up = 0.0
                d_ij = self._distance(i, j)  # m

                for ch in chs:
                    k = ch - 1
                    lam, h2 = self.edge_wavelength[j, k], np.abs(self.h_ijk[i, j, k]) ** 2
                    P, beta = self.local_p[i, j, k], (lam / (4 * np.pi * d_ij)) ** 2
                    snr = P * beta * h2 / self.sigma2
                    R_up += self.config['bw'] * np.log2(1 + snr)

                T_upload = (self.local_B[i] * 8e6) / max(R_up, 1e-9)

                if edge_nexe[j] > 0:
                    mu_j = self.edge_f[j] / edge_cave[j]
                    rho = edge_g[j] / mu_j
                    if rho >= 1:
                        violated = True
                        break
                    T_wait = edge_g[j] / (mu_j * max(mu_j - edge_g[j], 1e-9))
                    T_exec = (self.local_C[i] * edge_nexe[j]) / self.edge_f[j]
                else:
                    T_wait, T_exec = 0.0, 0.0

                delay_i = self.local_g[i] * (T_upload + T_wait + T_exec)
                if delay_i > self.tlimit[i]:
                    violated = True
                    break

                energy_i = self.local_g[i] * total_P * T_upload
                total_delay += delay_i
                total_energy += energy_i

        done = (self.step_cnt >= self.config['max_steps'])

        if violated:
            reward = self.config['violation_penalty']
            #done = True
        else:

            T_min, T_max = self.config["norm_T_min"], self.config["norm_T_max"]
            E_min, E_max = self.config["norm_E_min"], self.config["norm_E_max"]
            clipped_delay = np.clip(total_delay, T_min, T_max)
            clipped_energy = np.clip(total_energy, E_min, E_max)

            eps = 1e-9
            T_norm = (clipped_delay - T_min) / (T_max - T_min + eps)
            E_norm = (clipped_energy - E_min) / (E_max - E_min + eps)

            cost = self.config['reward_alpha'] * T_norm + self.config['reward_beta'] * E_norm
            reward = -cost


        self.rho_j_prev = edge_g / (self.edge_f / (edge_cave + 1e-9) + 1e-9)
        self.rho_j_prev = np.clip(self.rho_j_prev, 0, 5)
        self.edge_cave_prev = edge_cave

        if self.config['task_dynamics']:
            self.local_g = get_local_g()
            self.tlimit = get_tlimit(self.local_g)

        if self.config['channel_dynamics']:
            self.h_ijk = generate_rician_h_batch()


        next_state = self.builder.build_state(
            self.local_location, self.local_C, self.local_B, self.local_f, self.local_g, self.local_z, self.tlimit,
            self.psend_max,
            self.edge_location, self.edge_f, self.edge_Nmax,
            self.edge_wavelength, self.h_ijk, self.local_p,
            self.rho_j_prev, self.edge_cave_prev
        )

        info = {
            'total_delay': total_delay if not violated else -1,
            'total_energy': total_energy if not violated else -1,
            'violated': violated
        }

        return next_state, reward, done, info