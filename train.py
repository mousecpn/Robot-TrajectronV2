import torch
import numpy as np
from torch.utils.data import DataLoader
from model.trajectron import Trajectron
from tqdm import tqdm
import json
from argument_parser import args
import os
from dataset.preprocessing import ImageContextTrajDataset, load_data_cartesian
import torch.nn as nn
import torch.optim as optim
import evaluation
import matplotlib.pyplot as plt
# from train_rl import navigation_evaluate
import warnings
 
warnings.filterwarnings('ignore')

def main():
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
    dim = 2

    data_dict = load_data_cartesian(args.data_path, 20, 20, test_size=0.1)

    traindataset = ImageContextTrajDataset(data_dict["trainData"], data_dict["goals_train"], data_dict["obs_train"], max_history_length=8, min_future_timesteps=12, frequency=data_dict["target_frequency"], eval=False, pad=False, random_drop_goal=False)
    testdataset = ImageContextTrajDataset(data_dict["testData"], data_dict["goals_test"], data_dict["obs_test"], max_history_length=8, min_future_timesteps=12, frequency=data_dict["target_frequency"], eval=True, pad=False)
    train_dataloader = DataLoader(traindataset,
                                    collate_fn=traindataset.collate,
                                    pin_memory=True,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.preprocess_workers)
    
    eval_dataloader = DataLoader(testdataset,
                                    collate_fn=testdataset.collate,
                                    pin_memory=True,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.preprocess_workers)
    
    hyperparams["frequency"] = data_dict["target_frequency"]

    trajectron = Trajectron(hyperparams, args.device)
    
    
    trajectron.set_annealing_params()
    trajectron.model.train()

    optimizer = optim.Adam([
                    {'params': trajectron.model.node_modules.parameters(), "lr":hyperparams['learning_rate']},
                    {'params': trajectron.model.critic.parameters(), "lr":1e-4},
                    {'params': trajectron.model.log_alpha, "lr":1e-4}
                        ]
                     )
    # Set Learning Rate
    if hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=hyperparams['learning_decay_rate'])


    curr_iter = 0
    for epoch in range(1, args.train_epochs + 1):
        trajectron.model.to(args.device)
        pbar = tqdm(train_dataloader, ncols=80)
        for batch in pbar:
            (first_history_index, x_t, y_t, x_st_t, y_st_t, context) = batch

            batch = (first_history_index, x_t, y_t[...,dim:dim*2], x_st_t, y_st_t[...,dim:dim*2], context)
            trajectron.set_curr_iter(curr_iter)
            trajectron.step_annealers()
            optimizer.zero_grad()
            train_loss = trajectron.train_loss(batch)
            pbar.set_description(f"Epoch {epoch},  L: {train_loss.item():.2f}")
            train_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model.parameters(), hyperparams['grad_clip'])
            optimizer.step()


            # Stepping forward the learning rate scheduler and annealers.
            if optimizer.param_groups[0]['lr'] > hyperparams['min_learning_rate']:
                lr_scheduler.step()

            curr_iter += 1
            
        print("learning_rate:",lr_scheduler.get_last_lr()[0])

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            trajectron.model.to(args.device)
            trajectron.model.eval()
            with torch.no_grad():
                # Calculate evaluation loss
                eval_loss_list = []
                print(f"Starting Evaluation @ epoch {epoch}")
                pbar = tqdm(eval_dataloader, ncols=80)
                ade = []
                fde = []
                for batch in pbar:
                    (first_history_index, x_t, y_t, x_st_t, y_st_t, context) = batch
                    batch = (first_history_index, x_t, y_t[...,dim:dim*2], x_st_t, y_st_t[...,dim:dim*2], context)
                    eval_loss = trajectron.eval_loss(batch)
                    
                    pbar.set_description(f"Epoch {epoch}, L: {eval_loss.item():.2f}")
                    eval_loss_list.append({'nll': [eval_loss]})
                    # predictions = trajectron.predict(batch,
                    #                         ph,
                    #                         num_samples=20,
                    #                         z_mode=False,
                    #                         gmm_mode=False,
                    #                         full_dist=False)
                    predictions = trajectron.predict(batch,
                                        ph,
                                        num_samples=20,
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=True,
                                        full_dist=False)
                    
                    batch_ade = np.min(evaluation.compute_ade(predictions, y_t[...,0:dim].detach().cpu().numpy()),axis=0)
                    batch_fde = np.min(evaluation.compute_fde(predictions, y_t[...,0:dim].detach().cpu().numpy()),axis=0)
                    # ax = plt.axes()
                    # visualization.plot_trajectories2d(ax, predictions, x_t[0,:,0:2].detach().cpu().numpy() ,y_t[0,:,0:2].detach().cpu().numpy())


                    ade.append(batch_ade) 
                    fde.append(batch_fde)
                ade = np.mean(np.concatenate(ade,axis=0))*1000
                fde = np.mean(np.concatenate(fde,axis=0))*1000
            if ade < best_ade: 
                best_ade = ade
                # model_registrar.save_models(epoch)
            model_save = trajectron.model
            torch.save(model_save.node_modules, "checkpoints/epoch{}|{}Hz|ade{:.2f}.pth".format(epoch,data_dict["target_frequency"],ade))
            torch.save(model_save.critic, "checkpoints/epoch{}|{}Hz|ade{:.2f}|critic.pth".format(epoch,data_dict["target_frequency"],ade))
            print("ade:", ade)
            print("fde:", fde)
            # navigation_evaluate(trajectron,100)

            trajectron.model.train()
    return


if __name__=="__main__":
    main()

# python main.py --eval_every 10 --vis_every 10 --train_data_dict eth_train.pkl --eval_data_dict eth_val.pkl --offline_scene_graph yes --preprocess_workers 0 --batch_size 256 --log_dir experiments/pedestrians/models --log_tag _eth_vel_ar3 --train_epochs 100 --augment --conf config/config1.json --no_edge_encoding 