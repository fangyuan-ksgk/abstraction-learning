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
        self.device = device

    def act(self, epsilon: float): 
        # TBD: missing random action selection logic & epsilon-greedy logic
        pairs = self.dat.act(self.state, self.trajectory)
        assert len(pairs) == 1, "Only one sample is supported for now -- unless we figure out how to parallelize environment & load from state"
        _, action_idx = pairs[0]
        return action_idx.item() 
    
    def update(self, obs: Union[numpy.array, torch.Tensor], action: Union[int, torch.Tensor], reward: float): 
        
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device) if isinstance(obs, numpy.ndarray) else obs
        action = torch.tensor([action], dtype=torch.int64, device=self.device) if isinstance(action, int) else action
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device) if (isinstance(reward, float) or isinstance(reward, int)) else reward

        ts = self.state.timestamps[self.state.sample_idx == 0][-1]
        has_state = self.state.state_mask[self.state.timestamps == ts].any()
        has_action = self.state.action_mask[self.state.timestamps == ts].any()
        _obs, _action, _reward = self.trajectory.pop() # pop at the end

        assert has_action, "Last time-stamp misses action, illegal update"
        assert _action is not None and _action[-1] == action, "Provided action doesn't match last time-stamp action, update is illegal"
        assert _reward is None or _reward.size(0) < _action.size(0), "Reward already exists, update is illegal"

        if not has_state: 
            self.state.insert_next_token(sample_idx=0, next_token=PLACE_HOLDER_STATE_TOK, next_level=0, next_timestamp=ts)
            obs = torch.cat([_obs, obs.unsqueeze(0)]) if _obs is not None else obs
        else:
            assert _obs is not None, "No state in data, but it exists in trajectory"
            obs = torch.cat([_obs[:-1], obs.unsqueeze(0)])  # replace imaginary state with grounded one

        reward = torch.cat([_reward, reward]) if _reward is not None else reward

        self.trajectory.append((obs, _action, reward))



# Utility functions for game play visualization
# --------------------------------------------------------------------------------------------------------------------------
def collect_dat_game_play_frames(dat: DAT, env, n_rounds: int = 5):

    frames = {}
    for i in range(n_rounds):

        init_obs = env.reset()
        dat_bot = HiearchicalAgent(dat, init_obs, device="cpu")

        done = False
        frames[i] = []
        while not done:
            action = dat_bot.act(0.0) # select action & imagine next state 
            obs, reward, done, info = env.step(action) # environment step
            dat_bot.update(obs, action, reward)

            img = env.render("rgb_array")
            frames[i].append(img)

        del dat_bot

    return frames