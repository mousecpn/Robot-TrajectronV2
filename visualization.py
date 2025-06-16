import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import torch
from dataset.preprocessing import load_data_cartesian,derivatives_of
from model.trajectron import Trajectron
from argument_parser import args
import json
from dataset.map_generation import map2d_bilinear_generation
import imageio
import pickle

def main():
    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    torch.cuda.set_device(args.device)

    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['map_encoding'] = args.map_encoding

    data_dict = load_data_cartesian(args.data_path, 10, 20, test_size=0.1, viz=True, only_test=True)
    
    
    testData, goals_test, obs_test = data_dict["testData"], data_dict["goals_test"], data_dict["obs_test"]
    pred_dim = testData[0].shape[-1]
    hyperparams["frequency"] = data_dict['target_frequency']
    dt = 1/hyperparams["frequency"]

    trajectron = Trajectron(hyperparams, args.device )
    model = torch.load(args.checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    count = 1
    os.makedirs('gif_images', exist_ok=True)
    filenames = []
    for l in range(len(testData)):
        data = np.array(testData[l])
        vel_seq = derivatives_of(data, dt=1/dt)
        acc_seq = derivatives_of(vel_seq, dt=dt)
        data = np.concatenate((data,vel_seq,acc_seq), axis=-1)
        
        fig = plt.figure()
        ax = plt.axes()
        # ax = plt.axes(projection='3d')
        # plt.xlabel('Y-axis', fontsize=12) 
        # plt.ylabel('X-axis', fontsize=12)
        # ax.set_axis_off()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.grid()
        # ax.axes.zaxis.set_ticklabels([])
        # ax.axes.get_zaxis().set_visible(False)
        plt.ion()
        # data = data[::2,:]
        steps = data.shape[0]-8
        x_range = (-9, 9)
        y_range = (-3, 13)
        ax.set_xlim([x_range[0], x_range[1]])
        ax.set_ylim([y_range[0], y_range[1]])
        plt.tick_params(axis='both', labelsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        curve = None
        agent = None


        for j in range(steps):
            first_history_index = torch.LongTensor(np.array([0])).cuda()
            x = data[j:j+8,:9]
            y = data[j+8:j+12,:9]
            # ph = data.shape[0]-(j+8)
            ph = ph
            dim = x.shape[1]
            if dim == 9:
                std = np.array([3,3,3,2,2,2,1,1,1])
            else:
                std = np.array([3,3,2,2,1,1])

            goals = goals_test[l]
            goals = np.array(goals)[:, 0:dim//3]
            obs = obs_test[l]
            obs = np.array(obs)[:, 0:dim//3]

            if j == 0:
                width = 1.3
                goals = goals - width/2
                obs = obs - width/2
                for g in goals:
                    circle = plt.Rectangle((g[1], g[0]), width=width, height=width, facecolor='#4DAA59', fill=True, edgecolor='black', linewidth=0.5)
                    ax.add_patch(circle)
                for o in obs:
                    circle = plt.Rectangle((o[1],o[0]), width=width, height=width, facecolor='#E76D7E', fill=True, edgecolor='black', linewidth=0.5)
                    ax.add_patch(circle)

            rel_state = np.zeros_like(x[0])
            rel_state[0:dim//3] = np.array(x)[-1, 0:dim//3]
            goals = (goals - rel_state[0:dim//3])/std[:dim//3]

            obs = (obs - rel_state[0:dim//3])/std[:dim//3]


            map_tensor = map2d_bilinear_generation(goals.tolist(), obs.tolist(), 3, 25)

            x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
            y_st = np.where(np.isnan(y), np.array(np.nan), y / std)
            x_t = torch.tensor(x, dtype=torch.float).unsqueeze(0).cuda()
            y_t = torch.tensor(y, dtype=torch.float).unsqueeze(0).cuda()
            x_st_t = torch.tensor(x_st, dtype=torch.float).unsqueeze(0).cuda()
            y_st_t = torch.tensor(y_st, dtype=torch.float).unsqueeze(0).cuda()
            obs = torch.tensor(obs, dtype=torch.float)
            goals = torch.tensor(goals, dtype=torch.float)
            context = {
                "goals": [goals],
                "obstacles": [obs],
            }
            
            batch = (first_history_index, x_t, y_t[...,dim//3:2*dim//3], x_st_t, y_st_t[...,dim//3:2*dim//3], {'map':map_tensor.unsqueeze(0).cuda()})

            
            # try:
            with torch.no_grad():
                ################# most likely ##############################
                y_dist, _, predictions = trajectron.predict(batch,
                                        ph=ph,
                                        num_samples=1, # doesn't matter when all_z_sep is true
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=True,
                                        full_dist=False,
                                        dist=True)
            # except:
            #     pass

            mode_score = np.exp(trajectron.model.latent.p_dist.logits.detach().cpu().numpy()[0,0])
            vis_data = data[:j+8,:9]
            ax.plot(vis_data[:,1], vis_data[ :,0], '#34638D')
            ax.scatter(vis_data[::2,1], vis_data[::2,0], s=5, c='#34638D')
            if agent is not None:
                agent.remove()
            agent = plt.Circle((vis_data[-1,1],vis_data[-1,0]), 0.5, facecolor='#DEDEDE', fill=True, edgecolor='black',linestyle='-.', linewidth=0.5)
            ax.add_patch(agent)
            # plt.pause(0.01)
            # plt.ioff()
            # vis_pred = predictions[:, 0]#.detach().cpu().numpy()
            # vis_pred = np.concatenate((data[j+7:j+8,:pred_dim].reshape(1,1,pred_dim), vis_pred),axis=1)
            if curve is not None:
                if type(curve) == list:
                    for c in curve:
                        c.pop(0).remove()
                else:
                    curve.pop(0).remove()
            # if curve is not None:
            #     curve.remove()
                # dist_print.remove()
            # last_prod, dist_print = visualize_distribution2d_running(ax, y_dist, x_range, y_range, z, None, print=True)
            # curve, = ax.plot(vis_pred[0, :,0], vis_pred[0, :,1], 'red')
            curve = []
            for s in range(predictions.shape[0]):
                vis_pred = predictions[s]#.detach().cpu().numpy()
                vis_pred = np.concatenate((data[j+7:j+8,:pred_dim].reshape(1,1,pred_dim), vis_pred),axis=1)
                # curve.append(ax.plot(vis_pred[0, :,1], vis_pred[0, :,0], 'red', alpha=mode_score[s] ))  
                curve.append(ax.plot(vis_pred[0, :,1], vis_pred[0, :,0], '#34638D', linestyle='--', alpha=mode_score[s] ))  
                
                # curve.append(ax.plot(vis_pred[0, :,0], vis_pred[0, :,1], 'red')) 
            img_file_name = 'gif_images/traj_step{}_index{}.pdf'.format(j,l)
            filenames.append(img_file_name)
            # plt.savefig(img_file_name, bbox_inches='tight', pad_inches=0.1)

            plt.pause(0.001)
            plt.ioff()
        # data = data
        ax.plot(data[:,1], data[ :,0], 'blue')
        ax.scatter(data[::2,1], data[::2,0], s=5, c='green')
        # plt.show()
        plt.pause(0.02)
        plt.close(fig)

        # with imageio.get_writer("traj{}.mp4".format(l),fps=10) as writer:
        #     for filename in filenames:
        #         writer.append_data(imageio.imread(filename))

        # ax.set_title('3D line plot')
        # plt.show()
    return


if __name__=="__main__":
    main()