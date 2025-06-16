import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from torch.utils.data._utils.collate import default_collate
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from dataset.map_generation import map2d_bilinear_generation2
def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x

def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt
    # dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=0.0) / dt

    return dx

def derivatives_of(x, dt=1, radian=False):
    timestep, dim = x.shape
    dxs = []
    for d in range(dim):
        dxs.append(derivative_of(x[:,d],dt))
    dxs = np.stack(dxs,axis=-1)
    return dxs

def load_data_cartesian(path, target_frequency, min_length, test_size=0.3, dim=2, viz=False, only_test=False):
    # trainData = {}
    trainData = []
    testData = []
    with open(path,'rb') as f:
        data = pickle.load(f)
    ee_log = data["data"][:100]
    frequency = data["frequency"]
    try:
        goals_list = data['goals'][:100]
    except:
        goals_list = None
    try:
        obstacles_list = data['obstacles'][:100]
    except:
        obstacles_list = None
    dt = 1./target_frequency
    scale = 1
    # for i in range(100):
    #     plt.plot(np.array(ee_log[i])[:,0], np.array(ee_log[i])[:,1])
    #     plt.scatter(np.array(goals[i])[:,0], np.array(goals[i])[:,1], c="red", s=20)
    #     plt.scatter(np.array(obstacles[i])[:,0], np.array(obstacles[i])[:,1], c="blue", s=20, marker="*")
    #     plt.show()
    # dt = 1./frequency

    stride = int(frequency//target_frequency)
    if goals_list is not None:
        train_set, test_set, goals_trainset, goals_testset, obs_trainset, obs_testset = train_test_split(ee_log, goals_list, obstacles_list, test_size=test_size, random_state=42)
    else:
        train_set, test_set  = train_test_split(ee_log, test_size=test_size, random_state=42)

    # traj2context_train = []
    # traj2context_test = []
    goals_train = []
    goals_test = []
    obs_train = []
    obs_test = []
    goals = None
    obs = None
    if not only_test:
        # training dataset
        for l in range(len(train_set)):
            if goals_list is not None:
                goals = (np.array(goals_trainset[l])*scale).reshape(-1,3)
                obs = (np.array(obs_trainset[l])*scale).reshape(-1,3)
                goals = goals[:,:dim]
                obs = obs[:,:dim]
            cur_sequence = (np.array(train_set[l])*scale)
            for s in range(stride):
                idx_list = np.array(range(s, cur_sequence.shape[0], stride))
                # term = cur_sequence[:,:3]
                term = cur_sequence[idx_list,:dim]
                
                if term.shape[0] < min_length:
                    term = cur_sequence[:,:dim]

                trainData.append(term)
                goals_train.append(goals)
                obs_train.append(obs)

    for l in range(len(test_set)):
        if goals_list is not None:
            goals = (np.array(goals_testset[l])*scale).reshape(-1,3)
            obs = (np.array(obs_testset[l])*scale).reshape(-1,3)
            goals = goals[:,:dim]
            obs = obs[:,:dim]
        cur_sequence = (np.array(test_set[l])*scale)
        # cur_sequence += (np.random.rand(cur_sequence.shape[0],cur_sequence.shape[1])-0.5)*0.1
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        term = cur_sequence[idx_list,:dim]
        if viz == True:
            testData.append(term)
            goals_test.append(goals)
            obs_test.append(obs)
        else:
            for j in range(term.shape[0]//min_length):
                term_j = term[j*min_length:(j+1)*min_length]
                if term_j.shape[0] < min_length:
                    continue

                testData.append(term_j)
                goals_test.append(goals)
                obs_test.append(obs)
                # traj2context_test.append(l)
        # if term.shape[0] < min_length:
        #     continue
    data_dict ={
        "trainData": trainData,
        "testData": testData,
        "target_frequency": target_frequency,
        
    }
    if goals_list is not None:
        data_dict["goals_train"] = goals_train
        data_dict["goals_test"] = goals_test
    if obstacles_list is not None:
        data_dict["obs_train"] = obs_train
        data_dict["obs_test"] = obs_test
    return data_dict


class TrajDataset(Dataset):
    def __init__(self, Data, max_history_length, min_future_timesteps, eval=False):
        self.data = Data
        # self.data = torch.FloatTensor(Data)
        self.max_ht = max_history_length
        self.min_ft = min_future_timesteps
        # self.inseq = self.data[:, :in_length, :]
        # self.outseq = self.data[:, in_length:, :]
        self.eval = eval

        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data[index]
        if self.eval:
            t = np.array(8)
        else:
            t = np.random.choice(np.arange(3, len(seq)-self.min_ft), replace=False)
            # t = np.array(8)
        timestep_range_x = np.array([t - self.max_ht, t])
        timestep_range_y = np.array([t, t + self.min_ft])
        first_history_index = (self.max_ht - t).clip(0)

        length = timestep_range_x[1] - timestep_range_x[0]

        data_array = seq[max(timestep_range_x[0],0):timestep_range_x[1]]
        x = np.full((length, data_array.shape[1]), fill_value=np.nan)
        x[first_history_index:length] = data_array.copy()

        y = seq[max(timestep_range_y[0],0):timestep_range_y[1]] # velocity

        # x = seq[max(timestep_range_x[0],0):timestep_range_x[1]+1]
        # y = seq[timestep_range_y[0]:timestep_range_y[1],3:6] # velocity
        dim = x.shape[1]
        if dim == 9:
            std = np.array([3,3,3,2,2,2,1,1,1])
        elif dim == 6:
            std = np.array([3,3,2,2,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])
        # std = np.array([2,2,2,2,2,2,1,1,1])


        rel_state = np.zeros_like(x[0])
        rel_state[0:dim//3] = np.array(x)[-1, 0:dim//3]

        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), (y - rel_state) / std)
        x_t = torch.tensor(x, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float)
        x_st_t = torch.tensor(x_st, dtype=torch.float)
        y_st_t = torch.tensor(y_st, dtype=torch.float)


        return first_history_index, x_t, y_t, x_st_t, y_st_t

class ContextTrajDataset(Dataset):
    def __init__(self, data, goals, obs, max_history_length, min_future_timesteps, eval=False, pad=False, random_drop_goal=False):
        self.data = data
        # self.data = torch.FloatTensor(Data)
        self.max_ht = max_history_length
        self.min_ft = min_future_timesteps
        # self.inseq = self.data[:, :in_length, :]
        # self.outseq = self.data[:, in_length:, :]
        self.eval = eval
        self.goals = goals
        self.obs = obs
        self.random_drop_goal = random_drop_goal
        self.random_drop_obs = random_drop_goal
        self.pad = pad
        self.rotation_aug = False

        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data[index]
        if self.eval:
            t = np.array(8)
        else:
            t = np.random.choice(np.arange(3, len(seq)-self.min_ft), replace=False)
            # t = np.array(8)
        timestep_range_x = np.array([t - self.max_ht, t])
        timestep_range_y = np.array([t, t + self.min_ft])
        first_history_index = (self.max_ht - t).clip(0)
        
        length = timestep_range_x[1] - timestep_range_x[0]

        data_array = seq[max(timestep_range_x[0],0):timestep_range_x[1]]
        x = np.full((length, data_array.shape[1]), fill_value=np.nan)
        x[first_history_index:length] = data_array.copy()

        y = seq[max(timestep_range_y[0],0):timestep_range_y[1]] # velocity

        # x = seq[max(timestep_range_x[0],0):timestep_range_x[1]+1]
        # y = seq[timestep_range_y[0]:timestep_range_y[1],3:6] # velocity
        dim = x.shape[1]
        if dim == 9:
            std = np.array([3,3,3,2,2,2,1,1,1])
        elif dim == 6:
            std = np.array([3,3,2,2,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])
        # std = np.array([2,2,2,2,2,2,1,1,1])

        goals = self.goals[index]
        goals = np.array(goals)[:, 0:dim//3]

        obs = self.obs[index]
        obs = np.array(obs)[:, 0:dim//3]

        if self.rotation_aug is True:
            angle = np.random.random() * 2
            rot = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, angle])
            rot_mat = rot.as_matrix()[:2,:2]
            goals = goals @ rot_mat
            obs = obs @ rot_mat
            x[first_history_index:length, :2] = x[first_history_index:length, :2] @ rot_mat
            x[first_history_index:length, 2:4] = x[first_history_index:length, 2:4] @ rot_mat
            x[first_history_index:length, 4:] = x[first_history_index:length, 4:] @ rot_mat
            y[:,:2] = y[:,:2] @ rot_mat
            y[:,2:4] = y[:,2:4] @ rot_mat
            y[:,4:] = y[:,4:] @ rot_mat
        
        # plt.plot(x[:,0], x[:,1], c='green')
        # plt.plot(y[:,0], y[:,1], c='red')
        # plt.show()

        rel_state = np.zeros_like(x[0])
        rel_state[0:dim//3] = np.array(x)[-1, 0:dim//3]

        goals = (goals - rel_state[0:dim//3])/std[:dim//3]
        obs = (obs - rel_state[0:dim//3])/std[:dim//3]

        if self.pad:
            x = np.where(np.isnan(x), x[None, first_history_index].repeat(8,0), x)
            first_history_index = 0
            # y = np.where(np.isnan(y), y[None, first_history_index].repeat(12,0), y)
        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), (y - rel_state) / std)

        x_t = torch.tensor(x, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float)
        x_st_t = torch.tensor(x_st, dtype=torch.float)
        y_st_t = torch.tensor(y_st, dtype=torch.float)

        goals = torch.tensor(goals, dtype=torch.float)
        ##### randomly add goals ####
        if np.random.random() < 0.5 and self.eval is False:
            goals = torch.cat((goals, y_st_t[-1:,:2]), dim=0)
        ##### randomly add goals ####
        obs = torch.tensor(obs, dtype=torch.float)

        ##### random drop goal #########
        if self.random_drop_goal and np.random.random() < 0.5 and self.eval is False:
            mask_idx = np.random.choice(goals.shape[0])
            mask = torch.ones(goals.shape[0]).bool()
            mask[mask_idx] = False
            goals = goals[mask]
        if self.random_drop_obs and np.random.random() < 0.5 and self.eval is False:
            mask_idx = np.random.choice(obs.shape[0])
            mask = torch.ones(obs.shape[0]).bool()
            mask[mask_idx] = False
            obs = obs[mask]

        return first_history_index, x_t, y_t, x_st_t, y_st_t, goals, obs
    
    def collate(self, batch):
        first_history_indices, x_ts, y_ts, x_st_ts, y_st_ts, goals, obs = [], [], [], [], [], [], []
        for b_idx in range(len(batch)):
            first_history_indices.append(batch[b_idx][0])
            x_ts.append(batch[b_idx][1])
            y_ts.append(batch[b_idx][2])
            x_st_ts.append(batch[b_idx][3])
            y_st_ts.append(batch[b_idx][4])
            goals.append(batch[b_idx][5])
            obs.append(batch[b_idx][6])
        first_history_indices = torch.tensor(first_history_indices)
        x_ts = torch.stack(x_ts)
        y_ts = torch.stack(y_ts)
        x_st_ts = torch.stack(x_st_ts)
        y_st_ts = torch.stack(y_st_ts)
        context = {
            "goals": goals,
            "obstacles": obs
        }
        return first_history_indices, x_ts, y_ts, x_st_ts, y_st_ts, context
    
    def pad_collate(self, batch):
        first_history_indices, x_ts, y_ts, x_st_ts, y_st_ts, goals = [], [], [], [], [], []
        for b_idx in range(len(batch)):
            first_history_indices.append(batch[b_idx][0])
            x_ts.append(batch[b_idx][1])
            y_ts.append(batch[b_idx][2])
            x_st_ts.append(batch[b_idx][3])
            y_st_ts.append(batch[b_idx][4])
            goals.append(batch[b_idx][5])
        first_history_indices = torch.tensor(first_history_indices)
        x_ts = torch.stack(x_ts)
        y_ts = torch.stack(y_ts)
        x_st_ts = torch.stack(x_st_ts)
        y_st_ts = torch.stack(y_st_ts)
        context = {
            "goals": goals
        }
        return first_history_indices, x_ts, y_ts, x_st_ts, y_st_ts, context


class ImageContextTrajDataset(Dataset):
    def __init__(self, data, goals, obs, max_history_length, min_future_timesteps, frequency, eval=False, pad=False, random_drop_goal=False):
        self.data = data
        # self.data = torch.FloatTensor(Data)
        self.max_ht = max_history_length
        self.min_ft = min_future_timesteps
        # self.inseq = self.data[:, :in_length, :]
        # self.outseq = self.data[:, in_length:, :]
        self.eval = eval
        self.goals = goals
        self.obs = obs
        self.random_drop_goal = random_drop_goal
        self.random_drop_obs = random_drop_goal
        self.pad = pad
        self.rotation_aug = True
        self.map_resolution = 25
        self.map_scale = 1
        self.dt=1/frequency

        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data[index]
        goals = self.goals[index]
        obs = self.obs[index]

        # if not self.eval:
        #     tr = np.random.choice([0,1,2,3])
        #     seq, goals, obs = transform_bmi(seq, tr, goals, obs)
        vel_seq = derivatives_of(seq, dt=self.dt)
        acc_seq = derivatives_of(vel_seq, dt=self.dt)
        seq = np.concatenate((seq,vel_seq,acc_seq),axis=-1)
        
        if self.eval:
            t = np.array(8)
        else:
            t = np.random.choice(np.arange(1, len(seq)-self.min_ft), replace=False)
            # t = np.array(8)
        timestep_range_x = np.array([t - self.max_ht, t])
        timestep_range_y = np.array([t, t + self.min_ft])
        first_history_index = (self.max_ht - t).clip(0)
        
        length = timestep_range_x[1] - timestep_range_x[0]

        data_array = seq[max(timestep_range_x[0],0):timestep_range_x[1]]
        x = np.full((length, data_array.shape[1]), fill_value=np.nan)
        x[first_history_index:length] = data_array.copy()

        y = seq[max(timestep_range_y[0],0):timestep_range_y[1]] # velocity

        # x = seq[max(timestep_range_x[0],0):timestep_range_x[1]+1]
        # y = seq[timestep_range_y[0]:timestep_range_y[1],3:6] # velocity
        dim = x.shape[1]
        if dim == 9:
            std = np.array([3,3,3,2,2,2,1,1,1])
        elif dim == 6:
            std = np.array([3,3,2,2,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])
        # std = np.array([2,2,2,2,2,2,1,1,1])

        
        goals = np.array(goals)[:, 0:dim//3]
        obs = np.array(obs)[:, 0:dim//3]
        
        rel_state = np.zeros_like(x[0])
        rel_state[0:dim//3] = np.array(x)[-1, 0:dim//3]

        goals = (goals - rel_state[0:dim//3])#/std[:dim//3]
        obs = (obs - rel_state[0:dim//3])#/std[:dim//3]


        if self.pad:
            x = np.where(np.isnan(x), x[None, first_history_index].repeat(8,0), x)
            first_history_index = 0
            # y = np.where(np.isnan(y), y[None, first_history_index].repeat(12,0), y)
        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), (y - rel_state) / std)


        ##### random drop goal #########
        # if self.random_drop_goal and np.random.random() < 0.5 and self.eval is False:
        #     mask_idx = np.random.choice(goals.shape[0])
        #     mask = np.ones(goals.shape[0]).bool()
        #     mask[mask_idx] = False
        #     goals = goals[mask]
        # if self.random_drop_obs and np.random.random() < 0.5 and self.eval is False:
        #     mask_idx = np.random.choice(obs.shape[0])
        #     mask = np.ones(obs.shape[0]).bool()
        #     mask[mask_idx] = False
        #     obs = obs[mask]
        # (np.cumsum(x[:,2:4],axis=-1)*0.1 +  x[:1,:2]) - x[:,:2]
        
        ##### HER ####
        # if np.random.random() < 0.3 and self.eval is False:
        #     y_ext = seq[max(timestep_range_y[0],0):]
        #     y_ext += (np.random.rand(y_ext.shape[0],y_ext.shape[1])-0.5)*0.5
        #     y_ext_rel = np.where(np.isnan(y_ext), np.array(np.nan), (y_ext - rel_state))
        #     her_idx = np.random.choice(np.arange(0, len(y_ext_rel)), replace=False)
        #     goals = np.concatenate((goals, y_ext_rel[her_idx:her_idx+1,:2]), axis=0)
        # elif np.random.random() < 0.3 and self.eval is False:
        #     her_obs_idx = np.random.choice(np.arange(3, len(y_st)), replace=False)
        #     obs = np.concatenate((obs, y_st[her_obs_idx:her_obs_idx+1,:2]*std[:dim//3]), axis=0)
        ##### HER ####

        # fig, ax = plt.subplots(1, 2, figsize=(15,7))

        # ax[0].set_xlim(-5, 5)
        # ax[0].set_ylim(-5, 5)

        # ax[1].set_xlim(-5, 5)
        # ax[1].set_ylim(-5, 5)


        # ax[0].plot(x_st[first_history_index:length,0], x_st[first_history_index:length,1], c='green')
        # ax[0].plot(y_st[:,0], y_st[:,1], c='red')
        # for g in goals:
        #     circle = plt.Circle((g[0], g[1]), radius=0.15, color='g', fill=True, linewidth=2)
        #     ax[0].add_patch(circle)
        # for o in obs:
        #     circle = plt.Circle((o[0], o[1]), radius=0.15, color='r', fill=True, linewidth=2)
        #     ax[0].add_patch(circle)

        if self.rotation_aug is True and self.eval is False:
            angle = np.random.random() * 2
            rot = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, angle])
            rot_mat = rot.as_matrix()[:2,:2]
            goals = goals @ rot_mat
            obs = obs @ rot_mat
            x_st[first_history_index:length, :2] = x_st[first_history_index:length, :2] @ rot_mat
            x_st[first_history_index:length, 2:4] = x_st[first_history_index:length, 2:4] @ rot_mat
            x_st[first_history_index:length, 4:] = x_st[first_history_index:length, 4:] @ rot_mat
            y_st[:,:2] = y_st[:,:2] @ rot_mat
            y_st[:,2:4] = y_st[:,2:4] @ rot_mat
            y_st[:,4:] = y_st[:,4:] @ rot_mat
            # if np.random.random() < 0.1:
            #     x_st[first_history_index:length, :] *= 0.0
            x = np.where(np.isnan(x_st), np.array(np.nan), x_st * std + rel_state)
            y = np.where(np.isnan(y_st), np.array(np.nan), y_st * std + rel_state)
        
        # np.cumsum(x[1:,2:4],axis=0)*0.1 + x[:1,:2] - x[1:,:2]
        # np.cumsum(x[1:,4:6],axis=0)*0.1 + x[:1,2:4] - x[1:,2:4]
        
        
        # ax[1].plot(x_st[first_history_index:length,0], x_st[first_history_index:length,1], c='green')
        # ax[1].plot(y_st[:,0], y_st[:,1], c='red')
        # for g in goals:
        #     circle = plt.Circle((g[0], g[1]), radius=0.15, color='g', fill=True, linewidth=2)
        #     ax[1].add_patch(circle)
        # for o in obs:
        #     circle = plt.Circle((o[0], o[1]), radius=0.15, color='r', fill=True, linewidth=2)
        #     ax[1].add_patch(circle)
        # plt.show()

        map_tensor = map2d_bilinear_generation2(goals.tolist(), obs.tolist(), self.map_scale, self.map_resolution)
        map_tensor = torch.tensor(map_tensor, dtype=torch.float)

        goals_padded = np.full((10, 2), fill_value=np.nan)
        obs_padded = np.full((10, 2), fill_value=np.nan)
        goals_padded[0:goals.shape[0]] = goals
        obs_padded[0:obs.shape[0]] = obs
        goals = goals_padded
        obs = obs_padded

    
        x_t = torch.tensor(x, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float)
        x_st_t = torch.tensor(x_st, dtype=torch.float)
        y_st_t = torch.tensor(y_st, dtype=torch.float)
        goals = torch.tensor(goals, dtype=torch.float)
        obs = torch.tensor(obs, dtype=torch.float)

        return first_history_index, x_t, y_t, x_st_t, y_st_t, goals, obs, map_tensor
    
    def collate(self, batch):
        first_history_indices, x_ts, y_ts, x_st_ts, y_st_ts, goals, obs, maps = zip(*batch)
        # for b_idx in range(len(batch)):
        #     first_history_indices.append(batch[b_idx][0])
        #     x_ts.append(batch[b_idx][1])
        #     y_ts.append(batch[b_idx][2])
        #     x_st_ts.append(batch[b_idx][3])
        #     y_st_ts.append(batch[b_idx][4])
        #     goals.append(batch[b_idx][5])
        #     obs.append(batch[b_idx][6])
        #     maps.append(batch[b_idx][7])
        first_history_indices = torch.tensor(first_history_indices)
        x_ts = torch.stack(x_ts)
        y_ts = torch.stack(y_ts)
        x_st_ts = torch.stack(x_st_ts)
        y_st_ts = torch.stack(y_st_ts)
        maps = torch.stack(maps)
        goals = torch.stack(goals)
        obs = torch.stack(obs)
        context = {
            "goals": goals,
            "obstacles": obs,
            "map":maps
        }
        return first_history_indices, x_ts, y_ts, x_st_ts, y_st_ts, context
 

