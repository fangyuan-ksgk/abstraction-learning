import torch 
import numpy 
from typing import Union 
from utils import HierTraj
from model import DAT
from constant import PLACE_HOLDER_STATE_TOK, PLACE_HOLDER_ACTION_TOK

# --------------------------------------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------------------------------------------


def _init_trajectory(obs: Union[numpy.array, torch.Tensor], K: int, L: int, device="cuda"):

    sample = ([[PLACE_HOLDER_STATE_TOK] if l == 0 else [] for l in range(L)], None)
    state = HierTraj.from_hierarchical_data([sample], K=K, L=L)

    trajectories = []
    states = []
    empty_tensor = torch.empty(0).to(device)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device) if isinstance(obs, numpy.ndarray) else obs
    states.append(obs_tensor) # initial state    
    trajectory = (torch.stack(states).to(device), None, None)
    trajectories.append(trajectory)

    return state, trajectories


# --------------------------------------------------------------------------------------------------------------------------
# Hiearchical Agent (DAT module based)
# --------------------------------------------------------------------------------------------------------------------------

class HiearchicalAgent: 

    def __init__(self, dat: DAT, init_obs: Union[numpy.array, torch.Tensor], device="cuda"): 
        self.dat = dat
        self.state, self.trajectory = _init_trajectory(init_obs, K=self.dat.K, L=self.dat.L, device=device)

    def act(self, epsilon: float): 
        # TBD: missing random action selection logic & epsilon-greedy logic
        pairs = self.dat.act(self.state, self.trajectory)
        assert len(pairs) == 1, "Only one sample is supported for now -- unless we figure out how to parallelize environment & load from state"
        _, action_idx = pairs[0]
        return action_idx
    
    def update(self, obs, action, reward): 

        ts = self.state.timestamps[self.state.sample_idx == 0][-1]
        self.dat.insert_next_token(sample_idx=0, next_token=PLACE_HOLDER_STATE_TOK, next_level=0, next_timestamp=ts)

        _, _action, _reward = self.trajectory[0].pop() # pop at the end
        assert _action == action, "Action mismatch"
        assert _reward is None, "Reward already exists"
        self.trajectory[0].append((obs, action, reward))