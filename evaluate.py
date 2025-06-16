import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import evaluation
from model.trajectron import Trajectron
from dataset.preprocessing import load_data_cartesian,ContextTrajDataset
from torch.utils.data import DataLoader

def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=1,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    # for node in histories_dict:
    history = histories_dict
    future = futures_dict
    predictions = prediction_dict
    future = np.concatenate((history[-1:,:],future),axis=0)
    node_circle_size = np.sqrt(np.mean((history[1:] - history[:-1,:])**2)) * node_circle_size

    ax.plot3D(history[:, 0], history[:, 1], history[:, 2], 'k--')

    for sample_num in range(prediction_dict.shape[0]):

        if kde and predictions.shape[0] >= 50:
            line_alpha = 0.2
            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color=np.random.choice(cmap), alpha=0.8)

        ax.plot3D(predictions[sample_num, batch_num, :, 0], predictions[sample_num, batch_num, :, 1],predictions[sample_num, batch_num, :, 2],
                color=cmap[1],
                linewidth=line_width, alpha=line_alpha)

        ax.plot3D(future[:, 0],
                future[:, 1],
                future[:, 2],
                'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        # Current Node Position
        # circle = plt.Circle((history[-1, 0],
        #                      history[-1, 1]),
        #                     node_circle_size,
        #                     facecolor='g',
        #                     edgecolor='k',
        #                     lw=circle_edge_width,
        #                     zorder=3)
        # ax.add_artist(circle)

    # ax.axis('equal')
    




if __name__=='__main__':
    from argument_parser import args

    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)


    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['map_encoding'] = True
    best_ade = 1000

    data_dict = load_data_cartesian(args.data_path, 20, 20, test_size=0.1, aug=True)
    testdataset = ContextTrajDataset(data_dict["testData"], data_dict["goals_test"], max_history_length=8, min_future_timesteps=12, eval=True)
    
    eval_dataloader = DataLoader(testdataset,
                                    pin_memory=True,
                                    batch_size=256,
                                    shuffle=True,
                                    num_workers=0,
                                    collate_fn=testdataset.collate)
    
    hyperparams["frequency"] = data_dict['target_frequency']


    model_dir = None

    trajectron = Trajectron(hyperparams,
                            args.device)
    model = torch.load(args.checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()

    #################################
    #           EVALUATION          #
    #################################
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    with torch.no_grad():
        # Calculate evaluation loss
        eval_loss_list = []
        pbar = tqdm(eval_dataloader, ncols=80)
        ml_ade = []
        ml_fde = []
        bo20_ade = []
        bo20_fde = []
        for batch in pbar:
            # fig = plt.figure()
            (first_history_index, x_t, y_t, x_st_t, y_st_t, context) = batch
            batch = (first_history_index, x_t, y_t[...,2:4], x_st_t, y_st_t[...,2:4], context)
            eval_loss = trajectron.eval_loss(batch)
            
            pbar.set_description(f"L: {eval_loss.item():.2f}")
            eval_loss_list.append({'nll': [eval_loss]})

            ### best of 20
            predictions = trajectron.predict(batch,
                                    ph,
                                    num_samples=20,
                                    z_mode=False,
                                    gmm_mode=False,
                                    all_z_sep=False,
                                    full_dist=True,
                                    dist=True)
            
            batch_ade = np.min(evaluation.compute_ade(predictions, y_t[...,0:2].detach().cpu().numpy()),axis=0)
            batch_fde = np.min(evaluation.compute_fde(predictions, y_t[...,0:2].detach().cpu().numpy()),axis=0)


            bo20_ade.append(batch_ade)
            bo20_fde.append(batch_fde)

            ### most likely
            predictions = trajectron.predict(batch,
                    ph,
                    num_samples=20,
                    z_mode=True,
                    gmm_mode=True,
                    all_z_sep=True,
                    full_dist=False)

            batch_ade = np.min(evaluation.compute_ade(predictions, y_t[...,0:2].detach().cpu().numpy()),axis=0)
            batch_fde = np.min(evaluation.compute_fde(predictions, y_t[...,0:2].detach().cpu().numpy()),axis=0)


            ml_ade.append(batch_ade)
            ml_fde.append(batch_fde)
            
            
        ml_ade = np.mean(np.concatenate(ml_ade,axis=0))*1000
        ml_fde = np.mean(np.concatenate(ml_fde,axis=0))*1000

        bo20_ade = np.mean(np.concatenate(bo20_ade,axis=0))*1000
        bo20_fde = np.mean(np.concatenate(bo20_fde,axis=0))*1000

        print("most likely ade:", ml_ade)
        print("most likely fde:", ml_fde)

        print("best of 20 ade:", bo20_ade)
        print("best of 20 fde:", bo20_fde)

