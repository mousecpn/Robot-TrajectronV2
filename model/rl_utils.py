import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(Critic, self).__init__()
		fc = [nn.Linear(state_dim + action_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)

	def forward(self, state, action):
		x = torch.cat([state, action], -1)
		return self.fc(x)
	
class RewardModel(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):
		super(RewardModel, self).__init__()
		fc = [nn.Linear(state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)

	def forward(self, state):
		return self.fc(state)


class EnsembleCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], n_Q=2):
		super(EnsembleCritic, self).__init__()
		ensemble_Q = [Critic(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def forward(self, state, action):
		Q = [self.ensemble_Q[i](state, action) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q

class PathBuilder(dict):
    """
    Usage:
    ```
    path_builder = PathBuilder()
    path.add_sample(
        observations=1,
        actions=2,
        next_observations=3,
        ...
    )
    path.add_sample(
        observations=4,
        actions=5,
        next_observations=6,
        ...
    )
    path = path_builder.get_all_stacked()
    path['observations']
    # output: [1, 4]
    path['actions']
    # output: [2, 5]
    ```
    Note that the key should be "actions" and not "action" since the
    resulting dictionary will have those keys.
    """

    def __init__(self):
        super().__init__()
        self._path_length = 0
    
    def add_all(self, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        self._path_length += 1

    def get_all_stacked(self):
        output_dict = dict()
        for k, v in self.items():
            output_dict[k] = stack_list(v)
        return output_dict

    def __len__(self):
        return self._path_length



def stack_list(lst):
	if isinstance(lst[0], dict):
		return lst
	else:
		return np.array(lst)