import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.model_utils import ModeKeys,rgetattr,rsetattr,CustomLR,exp_anneal,sigmoid_anneal,unpack_RNN_state,run_lstm_on_variable_length_seqs,mutual_inf_mc
from model.dynamics import SingleIntegrator
from model.discrete_latent import DiscreteLatent
from model.gmm3d import GMM3D
from model.gmm2d import GMM2D
from model.attention import AdditiveAttention, MapEncoder, SimpleMapEncoder
from model.rl_utils import EnsembleCritic,RewardModel
import time
import matplotlib.pyplot as plt
import torch.distributions as td

class MultimodalGenerativeCVAE(nn.Module):
    def __init__(self,
                 hyperparams,
                 device):
        super(MultimodalGenerativeCVAE,self).__init__()
        self.hyperparams = hyperparams
        self.device = device
        self.curr_iter = 0
        # self.arch = self.hyperparams['arch']

        self.node_modules = nn.ModuleDict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state']
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state.values()]))
        self.context_types = ['goals', 'obstacles']

        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))
        self.create_graphical_model()

        dyn_limits = hyperparams['dynamic']['limits']
        self.dynamic = SingleIntegrator(1./self.hyperparams['frequency'], dyn_limits, device, self.x_size)


        # RL setting
        # self.col_threshold = 1.3
        self.col_threshold = 0.4
        self.goal_threshold = 0.8
        self.col_reward = -1
        self.goal_reward = 1
        self.action_reward = -0.01
        
        
        self.pg_alpha = 1.0
        self.discount_factor = 0.9
        self.critic_w = 0.5
        self.total_it = 0.0
        self.target_update_interval = 1
        self.tau = 0.005

        self.critic = EnsembleCritic(self.hyperparams['dec_rnn_dim'], self.pred_state_length).to(device)
        self.critic_target = EnsembleCritic(self.hyperparams['dec_rnn_dim'], self.pred_state_length).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # self.reward_model = RewardModel(self.hyperparams['dec_rnn_dim']).to(device)
        self.bce_loss = nn.BCELoss(reduction='none')

        self.state_cache = []
        self.img_enc_cache = None
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = - torch.prod(torch.Tensor((2,)).to(self.device)).item()


    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model):
        self.node_modules[name] = model.to(self.device)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule('/node_history_encoder',
                        model=nn.LSTM(input_size=self.state_length,
                                                hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                batch_first=True))
        
        
        if self.hyperparams['map_encoding']:
            self.add_submodule('/map_encoder', 
                               model=MapEncoder(self.hyperparams['enc_map_dim']))
            self.eie_output_dims = self.hyperparams['enc_map_dim']

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule('/node_future_encoder',
                           model=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule('/node_future_encoder/initial_h',
                           model=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule('/node_future_encoder/initial_c',
                           model=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))


        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        
        if self.hyperparams['map_encoding']:
            x_size += self.eie_output_dims

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule('/p_z_x',
                               model=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule('/hx_to_z',
                           model=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule('/q_z_xy',
                               #                                           Node Future Encoder
                               model=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule('/hxy_to_z',
                           model=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule('/decoder/state_action',
                           model=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule( '/decoder/rnn_cell',
                           model=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule('/decoder/initial_h',
                           model=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule('/decoder/proj_to_GMM_log_pis',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule('/decoder/proj_to_GMM_mus',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule('/decoder/proj_to_GMM_log_sigmas',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule('/decoder/proj_to_GMM_corrs',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']*3))

        self.x_size = x_size
        self.z_size = z_size


    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    
    def create_graphical_model(self):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  })

        self.create_new_scheduler(name='latent.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])


    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               labels,
                               labels_st,
                               first_history_indices,
                               context) -> torch.Tensor:
        """
        Encodes input and output tensors for node.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: tuple(x, y_e, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - y_e: Encoded label / future of the node.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        x, y_e, y = None, None, None
        initial_dynamics = dict()

        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:self.pred_state_length]
        node_vel = inputs[:, -1, self.pred_state_length:2*self.pred_state_length]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        node_pos_st = inputs_st[:, -1, 0:self.pred_state_length]
        node_vel_st = inputs_st[:, -1, self.pred_state_length:2*self.pred_state_length]

        n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel

        self.dynamic.set_initial_condition(initial_dynamics)

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)


        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = labels_st
        

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams['map_encoding']:
            map_feature = self.node_modules['/map_encoder'](context['map'])
            self.img_enc_cache = map_feature

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()
        
        if self.hyperparams['map_encoding']:
            x_concat_list.append(map_feature)

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)
        
        if torch.isnan(x).sum():
            print()

        return x, y_e, y, n_s_t0

    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        # if self.arch == "mamba":
        #     node_hist = self.node_modules['/node_history_projector'](node_hist)
        #     _, outputs = self.node_modules['/node_history_encoder'](node_hist)[:,-1,:]
        # else:
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules['/node_history_encoder'],
                                                    original_seqs=node_hist,
                                                    lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules['/node_future_encoder/initial_h']
        initial_c_model = self.node_modules['/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state


    def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules['/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules['/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules['/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules['/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(self, tensor) -> torch.Tensor:
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules['/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules['/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules['/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules['/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(self, mode, x, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False, measure=None):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM3D. If mode is Predict, also samples from the GMM.
        """
        self.state_cache = []
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules['/decoder/rnn_cell']
        initial_h_model = self.node_modules['/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules['/decoder/state_action'](n_s_t0)
        self.a_0 =  a_0.repeat(num_samples * num_components, 1)
        state = initial_state

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
        outputs = []
        
        self.state_cache.append(state)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            self.state_cache.append(state)
            
            if self.pred_state_length == 2:
                gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t[...,:1])
            else:
                gmm = GMM3D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]
            
            if j == 0 and measure is not None:
                pred_cov = gmm.cov.reshape(num_components, self.pred_state_length, self.pred_state_length)
                user_cov = measure.covariance_matrix
                K = torch.bmm(pred_cov, torch.linalg.inv(pred_cov + user_cov))
                new_mus = gmm.mus.reshape(num_components, self.pred_state_length, 1) + torch.bmm(K, (measure.mean - gmm.mus.reshape(num_components, self.pred_state_length)).reshape(num_components, self.pred_state_length, 1))
                new_cov = pred_cov - torch.bmm(K, pred_cov)

                v_u = measure.mean[0,:]
                # pis = gmm.log_pis.exp().reshape(-1)
                pis = self.latent.p_dist.logits.exp().reshape(-1)
                v_mode = gmm.mus[:,0,:]
                p_r_xcvu = td.MultivariateNormal(v_mode.reshape(-1, self.pred_state_length), gmm.cov[:,0])
                pis_lh = p_r_xcvu.log_prob(v_u.reshape(-1,2).repeat(num_components,1)).exp()

                pis_posterior = pis*(pis_lh + 1e-5)
                pis_posterior = pis_posterior/pis_posterior.sum()

                self.latent.p_dist = td.Categorical(logits=pis_posterior.log())
                # gmm.pis_cat_dist = td.Categorical(logits=pis_posterior.log())
                # gmm.log_pis = pis_posterior.log().reshape(1,1,1,-1)

                gmm = gmm.from_log_pis_mus_cov_mats(gmm.log_pis, new_mus.reshape(num_components, self.pred_state_length), new_cov, kf=True)
                mu_t = gmm.mus.squeeze(0)
                log_sigma_t = gmm.log_sigmas.squeeze(0)
                corr_t = gmm.corrs

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t[...,0].reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, self.pred_state_length
                ).permute(0, 2, 1, 3).reshape(-1, self.pred_state_length * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, self.pred_state_length
                ).permute(0, 2, 1, 3).reshape(-1, self.pred_state_length * num_components))
            if self.pred_state_length == 2:
                corrs.append(
                    corr_t[...,:1].reshape(
                        num_samples, num_components, -1
                    ).permute(0, 2, 1).reshape(-1, num_components))
            else:
                corrs.append(
                    corr_t.reshape(
                        num_samples, num_components, -1
                    ).permute(0, 2, 1).reshape(-1, 3* num_components))


            # dec_inputs = [zx, mu_t]
            dec_inputs = [zx, a_t]
            outputs.append(a_t)
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        outputs = torch.stack(outputs, dim=1) # (n_samples*n_components*batch_size, ph, dim)

        if self.pred_state_length == 3:
            a_dist = GMM3D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                        torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                        torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                        torch.reshape(corrs, [num_samples, -1, ph, num_components, 3]))
        else:
            a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                        torch.reshape(mus, [num_samples, -1, ph, num_components, pred_dim]),
                        torch.reshape(log_sigmas, [num_samples, -1, ph, num_components, pred_dim]),
                        torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        if self.hyperparams['dynamic']['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()            
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, a_dist, sampled_future
        else:
            # a_sample = a_dist.rsample()
            # return y_dist, a_sample
            return y_dist, outputs.reshape(num_samples*num_components,-1,ph,pred_dim)
    
    def p_y_xz2z(self, mode, x, n_s_t0, z_stacked, z_T,
               num_samples, ph_limit=100, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param z_T: stop predicting at z_T
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM3D. If mode is Predict, also samples from the GMM.
        """
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules['/decoder/rnn_cell']
        initial_h_model = self.node_modules['/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules['/decoder/state_action'](n_s_t0)

        state = initial_state

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
        outputs = []

        T = 1

        ph = 0
        while True:
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM3D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t[...,0].reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 3
                ).permute(0, 2, 1, 3).reshape(-1, 3 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 3
                ).permute(0, 2, 1, 3).reshape(-1, 3 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, 3* num_components))


            # dec_inputs = [zx, mu_t]
            dec_inputs = [zx, a_t]
            outputs.append(a_t)
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
            ph += 1
            if ph > ph_limit:
                print("out ot ph_limit")
                break
            if ph == 1:
                pos_mus = self.dynamic.initial_conditions['pos'].unsqueeze(1)[:, None].repeat(num_samples, 1, num_components, 1)
            else:
                pos_mus += mus[-1].reshape(num_samples, -1, num_components, pred_dim) * self.dynamic.dt
            if pos_mus[...,-1].mean() < z_T:
                break

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        outputs = torch.stack(outputs,dim=1)

        a_dist = GMM3D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components, 3]))

        if self.hyperparams['dynamic']['distribution']:
            y_dist = self.dynamic.integrate_distribution2zT(a_dist, z_T)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()            
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            a_sample = a_dist.rsample()
            return y_dist, a_sample

    def encoder(self, mode, x, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, x, y_e)
        self.latent.p_dist = self.p_z_x(mode, x)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p()
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(self, mode, x, y, n_s_t0, z, labels, prediction_horizon, num_samples, context={}):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        y_dist, outputs = self.p_y_xz(mode, x, n_s_t0, z,
                             prediction_horizon, num_samples, num_components=num_components)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        # prob = (y_dist.log_pis/torch.sum(y_dist.log_pis,-1,keepdim=True)).unsqueeze(-1)
        # pred = torch.sum(prob*y_dist.mus,dim=-2).squeeze(0)
        # mse = torch.sqrt(torch.mean((labels - pred)**2,dim=-2)).mean(0).sum(0)
        
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        # auxi_loss = mseloss# + col_loss
        # return log_p_y_xz, auxi_loss
        return log_p_y_xz, outputs, y_dist
    
    def collision_regularization(self, trajectories, obstacles):
        """
        trajectories: torch.tensor(batch_size, N_samples, horizon, dim)
        obstacles: [torch.tensor(N_obs, dim)]
        """
        col_loss = 0.
        batch_size = len(obstacles)
        count = 0
        for i in range(batch_size):
            if obstacles[i].shape[0] != 0:
                pairwise_distance = torch.norm(obstacles[i][None, None].to(self.device)-trajectories[i][:,:, None], p=2, dim=-1) #(N_samples, horizon, N_obs)
                col_loss -= pairwise_distance.mean()
                count += 1
        return col_loss/count
    
    # def reward_function(self, samples_trajectories, context):
    #     """
    #     trajectories: torch.tensor(batch_size, N_samples, horizon, dim)
    #     context: {
    #         obstacles: [torch.tensor(N_obs, dim)],
    #         goals: [torch.tensor(N_goals, dim)]
    #     }
    #     """
    #     batch_size, N_samples, horizon, dim = samples_trajectories.shape
    #     reward = torch.zeros(batch_size, N_samples).to(self.device)
    #     terminal_step = torch.ones(batch_size, N_samples).long().to(self.device) * self.ph

    #     obstacles = context['obstacles']
    #     goals = context['goals']
    #     for i in range(batch_size):
    #         goal_i = goals[i][~torch.isnan(goals[i])].reshape(-1,self.pred_state_length)
    #         if goal_i.shape[0] != 0:
    #             pairwise_goals_distance = torch.norm(goal_i[None, None].to(self.device)-samples_trajectories[i][:,:, None], p=2, dim=-1) #(N_samples, horizon, N_obs)
    #             mingoal_distance = pairwise_goals_distance.min(2)[0] 
    #             min_distance, arrived_step = mingoal_distance.min(1) #(N_samples, )
    #             arrived_mask = min_distance < self.goal_threshold
    #             reward[i][arrived_mask] = self.goal_reward
    #             terminal_step[i][arrived_mask] = arrived_step[arrived_mask]

    #         obs_i = obstacles[i][~torch.isnan(obstacles[i])].reshape(-1,self.pred_state_length)
    #         if obs_i.shape[0] != 0:
    #             pairwise_obs_distance = torch.norm(obs_i[None, None].to(self.device)-samples_trajectories[i][:,:, None], p=2, dim=-1) #(N_samples, horizon, N_obs)
    #             minobs_distance = pairwise_obs_distance.min(2)[0] 
    #             min_distance, crash_step = minobs_distance.min(1) #(N_samples, )
    #             crash_mask = min_distance < self.col_threshold
    #             reward[i][crash_mask] = self.col_reward
    #             terminal_step[i][crash_mask] = crash_step[crash_mask]            

    #     return reward, terminal_step

    def reward_function(self, samples_trajectories, context):
        """
        trajectories: torch.tensor(batch_size, N_samples, horizon, dim)
        context: {
            obstacles: [torch.tensor(N_obs, dim)],
            goals: [torch.tensor(N_goals, dim)]
        }
        """
        batch_size, N_samples, horizon, dim = samples_trajectories.shape
        reward = torch.zeros(batch_size, N_samples).to(self.device)
        terminal_step = torch.ones(batch_size, N_samples).long().to(self.device) * self.ph

        obstacles = context['obstacles'] # (batch_size, N_obs, 2)
        goals = context['goals'] # (batch_size, N_goals, 2)

        pairwise_goals_distance = torch.norm(goals[:, None, None].to(self.device)-samples_trajectories[:, :,:, None], p=2, dim=-1) #(batch_size, N_samples, horizon, N_obs)
        pairwise_goals_distance[torch.isnan(pairwise_goals_distance)] = self.goal_threshold + 1
        mingoal_distance = pairwise_goals_distance.min(-1)[0] #(batch_size, N_samples, horizon)
        min_distance, arrived_step = mingoal_distance.min(-1) #(batch_size, N_samples, )
        arrived_mask = min_distance < self.goal_threshold
        reward[arrived_mask] = self.goal_reward
        terminal_step[arrived_mask] = arrived_step[arrived_mask]

        pairwise_obs_distance = torch.norm(obstacles[:, None, None].to(self.device)-samples_trajectories[:,:, :, None], p=2, dim=-1) #(N_samples, horizon, N_obs)
        pairwise_obs_distance[torch.isnan(pairwise_obs_distance)] = self.col_threshold + 1
        minobs_distance = pairwise_obs_distance.min(-1)[0] 
        min_distance, crash_step = minobs_distance.min(-1) #(N_samples, )
        crash_mask = min_distance < self.col_threshold
        reward[crash_mask] = self.col_reward
        terminal_step[crash_mask] = crash_step[crash_mask]

        #################### viz #######################
        # reward_idx = torch.where(reward>0)[0][0]
        # reward_idx = 0
        # for i in range(samples_trajectories.shape[1]):
        #     if reward[reward_idx,i] == 1:
        #         color = 'orange'
        #     elif reward[reward_idx,i] <0:
        #         color = 'red'
        #     else:
        #         color = 'blue'
        #     plt.plot(samples_trajectories[reward_idx,i,:,0].detach().cpu().numpy(), samples_trajectories[reward_idx,i,:,1].detach().cpu().numpy(), c=color)
        # plt.scatter(goals[reward_idx][~torch.isnan(goals[reward_idx])].reshape(-1,2)[:,0].detach().cpu().numpy(), goals[reward_idx][~torch.isnan(goals[reward_idx])].reshape(-1,2)[:,1].detach().cpu().numpy(), c='green', s=30)
        # plt.scatter(obstacles[reward_idx][~torch.isnan(obstacles[reward_idx])].reshape(-1,2)[:,0].detach().cpu().numpy(), obstacles[reward_idx][~torch.isnan(obstacles[reward_idx])].reshape(-1,2)[:,1].detach().cpu().numpy(), c='red', s=30)
        # # plt.show() 
        # plt.savefig('reward.png')
        # plt.close()
        #################### viz #######################

        return reward, terminal_step
    
    def apf_reward_function(self, samples_trajectories, context):
        """
        trajectories: torch.tensor(batch_size, N_samples, horizon, dim)
        context: {
            obstacles: [torch.tensor(N_obs, dim)],
            goals: [torch.tensor(N_goals, dim)]
        }
        return: reward: torch.tensor(batch_size, N_samples, horizon,
        """
        batch_size, N_samples, horizon, dim = samples_trajectories.shape
        samples_trajectories = torch.cat((samples_trajectories, torch.zeros_like(samples_trajectories[:,:,:1,:])), dim=-2)
        returns = torch.zeros(batch_size, N_samples, horizon+1).to(self.device)
        reward_t = torch.zeros(batch_size, N_samples, horizon).to(self.device)
        terminal_step = torch.ones(batch_size, N_samples).long().to(self.device) * self.ph

        obstacles = context['obstacles']
        goals = context['goals']
        alpha_goal = 100
        maximun_obs_range = 3
        constant_k = 0.04
        for i in range(batch_size):
            if goals[i].shape[0] != 0:
                pairwise_goals_distance = torch.norm(goals[i][None, None].to(self.device)-samples_trajectories[i][:,:, None], p=2, dim=-1) #(N_samples, horizon+1, N_obs)
                mingoal_distance = pairwise_goals_distance.min(2)[0]  #(N_samples, horizon+1)
                min_distance, arrived_step = mingoal_distance.min(1) #(N_samples, )
                returns[i] -= alpha_goal * mingoal_distance
                arrived_mask = min_distance < self.goal_threshold
                # reward[i][arrived_mask] = self.goal_reward
                terminal_step[i][arrived_mask] = arrived_step[arrived_mask] -1

            if obstacles[i].shape[0] != 0:
                pairwise_obs_distance = torch.norm(obstacles[i][None, None].to(self.device)-samples_trajectories[i][:,:, None], p=2, dim=-1) #(N_samples, horizon+1, N_obs)
                minobs_distance = pairwise_obs_distance.min(2)[0]  #(N_samples, horizon+1)
                min_distance, crash_step = minobs_distance.min(1) #(N_samples, )

                repulsive_field = (1/(pairwise_obs_distance+constant_k) - 1/(maximun_obs_range+constant_k)) #(N_samples, horizon+1, N_obs)
                repulsive_field[repulsive_field<0] = 0.0
                repulsive_field = repulsive_field.sum(2)[0] #(N_samples, horizon+1)
                beta = torch.ones(N_samples, horizon+1).to(self.device) * 2
                beta[mingoal_distance<= maximun_obs_range*3/4] /= torch.exp(4*(maximun_obs_range*3/4-mingoal_distance))[mingoal_distance<= maximun_obs_range*3/4]
                returns[i] -= beta * repulsive_field

                crash_mask = min_distance < self.col_threshold
                # reward[i][crash_mask] = self.col_reward
                terminal_step[i][crash_mask] = crash_step[crash_mask] - 1
            reward_t = returns[...,1:] - returns[...,:-1]      

        return reward_t, returns, terminal_step
    
    def GAE(self, Qs, rewards, gamma_, lambda_, terminal_steps):
        """
        Qs: torch.tensor(N_samples*batch_size, horizon+1)
        rewards: torch.tensor(N_samples*batch_size, horizon+1)
        """
        delta = torch.zeros_like(Qs[...,:-1]) # (N_samples, batch_size, horizon)
        delta = rewards[...,:-1] + gamma_*Qs[...,1:] - gamma_*Qs[...,:-1]
        N_steps = rewards.shape[-1]
        advantage = torch.zeros_like(Qs)
        for i in reversed(range(0, advantage.shape[-1]-1)):
            advantage[...,i] = (delta[...,i] + (gamma_*lambda_)*advantage[...,i+1]) * (terminal_steps>=i).float()
        return advantage[...,:N_steps]
    
    def rl_train(self,
                 first_history_index, 
                 x_t, 
                 x_st_t, 
                 a_t, 
                 r_t, 
                 context):
        mode = ModeKeys.PREDICT

        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                            inputs=x_t,
                                                            inputs_st=x_st_t,
                                                            labels=None,
                                                            labels_st=None,
                                                            first_history_indices=first_history_index,
                                                            context=context)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=False,
                                                              full_dist=True,
                                                              all_z_sep=False)

        y_dist, a_dist, our_sampled_future = self.p_y_xz(mode, x, n_s_t0, z,
                                            1,
                                            num_samples,
                                            num_components,
                                            False)
        
        

        
        return

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   prediction_horizon,
                   context={}
                   ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN
        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=labels,
                                                        labels_st=labels_st,
                                                        first_history_indices=first_history_indices,
                                                        context=context)
        
        t1 = time.time()
        

        z, kl = self.encoder(mode, x, y_e)
        log_p_y_xz, samples_actions, y_dist = self.decoder(mode, x, y, n_s_t0, z,
                                  labels,  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'],
                                  context=context)
        
        
        #############  gt collision detection ###############
        # if 'obstacles' in context.keys() and 'goals' in context.keys():
        #     gt_traj = torch.cumsum(labels.reshape(-1,self.ph, self.pred_state_length), dim=1) * self.dynamic.dt
        #     rewards, terminal_steps = self.reward_function(gt_traj.unsqueeze(1), context)
        #     mask_gt = rewards.reshape(-1)>=0
        # else:
        mask_gt = torch.ones_like(labels[:,0,0]).bool()
        #############  gt collision detection ###############

        
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)[mask_gt]  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist, mask_gt)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist, mask_gt)

        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO #+ mseloss
        
        ############ mseloss ############
        num_components = self.hyperparams['N'] * self.hyperparams['K']
        # mseloss,_ = torch.min(torch.mean((outputs.reshape((-1,)+labels.shape) -labels.unsqueeze(0))**2,dim=(2,3)),dim=0)
        # mseloss = mseloss.mean()
        ############ mseloss ############

        ############# dynamics integral ###############
        expand_n_s_t0 = n_s_t0[:,:2].repeat(num_components,1).unsqueeze(1)
        samples_trajectories = torch.cumsum(samples_actions.reshape(-1,self.ph, self.pred_state_length), dim=1) * self.dynamic.dt + expand_n_s_t0
        samples_trajectories = samples_trajectories.reshape(num_components, -1, samples_trajectories.shape[-2], samples_trajectories.shape[-1]).permute(1,0,2,3)
        ############# dynamics intrgral ###############

        ############# obstacle avoidance loss ###############
        # if 'obstacles' in context.keys():
        #     obs = context['obstacles']
        #     col_loss = self.collision_regularization(samples_trajectories, obs)
        ############# obstacle avoidance loss ###############
        RL = False
        if RL == True:
            if self.total_it % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            
            if 'obstacles' in context.keys() and 'goals' in context.keys() and self.hyperparams['map_encoding'] is True:
                rewards, terminal_steps = self.reward_function(samples_trajectories, context) # (batch_size, sample_N)
                # returns = rewards * self.discount_factor**terminal_steps
                # returns[terminal_steps>=self.ph] = 0.0
                # done_mask = self.done_mask(terminal_steps.reshape(-1),self.ph)


                
            ############# Critic Optimization #########
                
                state_cache = torch.stack(self.state_cache, dim=1)
                samples_actions_ = torch.cat((self.a_0.reshape(-1, 1, self.pred_state_length), samples_actions.reshape(-1, self.ph,  self.pred_state_length)), dim=1).reshape(-1, self.pred_state_length)
                with torch.no_grad():
                    target_Q = self.critic_target(state_cache.reshape(-1,self.hyperparams['dec_rnn_dim']), samples_actions_) # (sample_N*batch_size*(ph+1), N_Q)
                    reward_t = torch.zeros_like(state_cache[...,0]) # (sample_N*batch_size, ph+1)
                    # reward_t = torch.ones_like(state_cache[...,0]) * self.action_reward
                    rewards[terminal_steps>=self.ph] = 0.0
                    terminal_steps = terminal_steps.permute(1,0).reshape(-1,1).contiguous()
                    done_mask = self.done_mask(terminal_steps[...,0],self.ph)
                    reward_t.scatter_(1, terminal_steps, rewards.permute(1,0).reshape(-1,1).contiguous()) # (sample_N*batch_size, ph+1)
                    target_Q = torch.min(target_Q, -1)[0].reshape(-1, self.ph+1) #  (sample_N*batch_size, ph+1)

                    ######### GAE ###################
                    advantage = self.GAE(target_Q, reward_t, gamma_=0.95, lambda_=0.98, terminal_steps=terminal_steps[...,0])
                    target_Q[:,:-1] = advantage[:,:-1] + target_Q[:,:-1]
                    target_Q.scatter_(1, terminal_steps, rewards.permute(1,0).reshape(-1,1).contiguous())
                    target_Q = target_Q[:,:-1]
                    ######### GAE ###################
                    
                    #######################
                    # target_Q[:,:-1] = reward_t[:,:-1] + self.discount_factor*target_Q[:,1:] # (sample_N*batch_size, ph+1)
                    # target_Q.scatter_(1, terminal_steps, rewards.permute(1,0).reshape(-1,1).contiguous())
                    # target_Q = target_Q[:,:-1]
                    ############################
                    target_Q.clip_(-1,1)
                
                Q = self.critic(state_cache.reshape(-1,self.hyperparams['dec_rnn_dim']), samples_actions_) # (sample_N*batch_size*(ph+1), N_Q)
                Q = Q.reshape(-1, self.ph+1, Q.shape[-1])
                
                critic_loss = self.critic_w * (Q[:,:-1] - target_Q.unsqueeze(-1)).pow(2).sum(-1)
                critic_loss = critic_loss[done_mask].mean()
                
                loss += critic_loss
                
                
            ############# Critic Optimization #########
                
            ############# reward model Optimization #########
                # R = self.reward_model(state_cache.reshape(-1,self.hyperparams['dec_rnn_dim'])).reshape(-1, self.ph+1)
                # reward_loss = self.bce_loss(R[...,1:].sigmoid(), ((reward_t[...,:-1]+1)/2))
                # loss += reward_loss[done_mask].mean()

            ############# reward model Optimization #########
            
            ############# Analytical MaxEnt #########
                # self.alpha = self.log_alpha.exp()
                # cov = y_dist.get_covariance_matrix()
                # det_cov = torch.det(cov)
                # h_i = 0.5*torch.log(((2*torch.e*torch.pi)**2*det_cov))
                # gaussian_alpha = y_dist.log_pis.exp()
                # entropy_lb = (h_i*gaussian_alpha).sum(-1).mean()
            ############# Analytical MaxEnt #########

            ############# Analytical MaxEnt2 #########
                self.alpha = self.log_alpha.exp()
                cov = y_dist.get_covariance_matrix()
                mus = y_dist.mus
                n_s, bs, ts, n_c, dim, _ = cov.shape
                cross_cov = cov[:,:,:,None] + cov[:,:,:,:, None] # (1, bs, ts, n_c, n_c, dim, dim)
                cross_mus_diff = (mus[:,:,:,None] - mus[:,:,:,:,None]) # (1, bs, ts, n_c, n_c, dim)
                cross_cov_inv_flatten = torch.inverse(cross_cov.reshape(-1,dim,dim))
                term = torch.bmm(cross_mus_diff.reshape(-1,1,dim),cross_cov_inv_flatten)
                score = torch.bmm(term, cross_mus_diff.reshape(-1,dim,1)).reshape(n_s, bs, ts, n_c, n_c)
                cross_cov_det = torch.det(cross_cov) # (1, bs, ts, n_c, n_c)
                Z = ((2*torch.pi)**2*cross_cov_det)**(1/2)
                cross_prob = (-0.5*score).exp()*Z.pow(-1)
                gaussian_alpha = y_dist.log_pis.exp()
                entropy = (cross_prob * gaussian_alpha.unsqueeze(-2)).sum(-1).log()
                entropy_lb = - (entropy*gaussian_alpha).sum(-1).mean()
            ############# Analytical MaxEnt2 #########
            
            ############# policy gradient #############
                # log_policy = torch.clamp(y_dist.log_prob(samples_actions), max=self.hyperparams['log_p_yt_xz_max']) # (sample_N, batch_size, ph)
                # done_mask = done_mask.reshape(log_policy.permute(1,0,2).shape)
                # actor_loss = - self.pg_alpha*(returns.unsqueeze(-1)*log_policy.permute(1,0,2))[done_mask].mean()
                # loss += actor_loss
            ############# policy gradient #############
                
            ############# Actor optimization gradient #############           
                # advantage = target_Q - Q[:,:-1].min(-1)[0]
                advantage = advantage[:, :-1]
                ##### normalize advantage ####
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # gaussian normalization
                # advantage = ((advantage - advantage.min()) / (advantage.max() - advantage.min() + 1e-8))*2 - 1.0 # gaussian normalization
                ##### normalize advantage ####
                log_policy = torch.clamp(y_dist.log_prob(samples_actions), max=self.hyperparams['log_p_yt_xz_max']) # (sample_N, batch_size, ph)
                actor_loss = - self.pg_alpha*(advantage.detach()*log_policy.reshape(-1,self.ph))[done_mask].mean()
                # neg_entropy_ref = log_policy.mean()
                
                neg_entropy = - entropy_lb
                alpha_loss = - (self.log_alpha * (log_policy + self.target_entropy).detach()).mean()

                loss += actor_loss + self.alpha[0].detach() * neg_entropy + alpha_loss
                
            ############# Actor optimization #############
            

        self.total_it += 1

        return loss
    
    # def done_mask2(self, terminal_steps, T):
    #     done_mask = torch.ones_like(terminal_steps.unsqueeze(-1).repeat(1,T))
    #     for t in range(T):
    #         mask_t = torch.zeros_like(terminal_steps)
    #         mask_t[terminal_steps>=t] = 1
    #         done_mask[:, t] = mask_t
    #     return done_mask 

    def done_mask(self, terminal_steps, T):
        done_mask = torch.ones_like(terminal_steps.unsqueeze(-1).repeat(1,T))
        steps = torch.cumsum(torch.ones_like(terminal_steps.unsqueeze(-1).repeat(1,T)),dim=-1) - 1
        done_mask[steps > terminal_steps.unsqueeze(-1).repeat(1,T)] = 0
        return done_mask 


    def eval_loss(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  prediction_horizon,
                  context={}) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                            inputs=inputs,
                                                            inputs_st=inputs_st,
                                                            labels=labels,
                                                            labels_st=labels_st,
                                                            first_history_indices=first_history_indices,
                                                            context=context)

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _, _ = self.p_y_xz(ModeKeys.PREDICT, x, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood

        return nll

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                prediction_horizon,
                num_samples,
                context={},
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                measure=None):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=None,
                                                        labels_st=None,
                                                        first_history_indices=first_history_indices,
                                                        context=context)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, a_dist, our_sampled_future = self.p_y_xz(mode, x, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode,
                                            measure)
        if dist == True:
            return y_dist, a_dist, our_sampled_future
        return our_sampled_future

    def predict2(self,
                inputs,
                inputs_st,
                first_history_indices,
                z_T,
                num_samples,
                context={},
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                ph_limit=100,):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=None,
                                                        labels_st=None,
                                                        first_history_indices=first_history_indices,
                                                        context=context)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, our_sampled_future = self.p_y_xz2z(mode, x, n_s_t0, z,
                                            z_T,
                                            num_samples,
                                            ph_limit,
                                            num_components,
                                            gmm_mode)
        if dist == True:
            return y_dist, our_sampled_future
        return our_sampled_future
    
    def get_latent(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   prediction_horizon,
                   context
                   ) -> torch.Tensor:

        mode = ModeKeys.TRAIN

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=labels,
                                                        labels_st=labels_st,
                                                        first_history_indices=first_history_indices,
                                                        context=context)
        return x


    def encode_edge(self,
                    mode,
                    node_history,
                    node_history_st,  # (bs, steps, state)
                    node_history_encoded, # (bs, dim)
                    contexts,  # [(obj, 3)]
                    context_type):

        max_hl = self.hyperparams['maximum_history_length']

        context_state = list()
        for c_idx in range(len(contexts)): #batch size
            context = contexts[c_idx].to(self.device)
            if context.shape[0] == 0:
                context = torch.zeros((1, context.shape[-1]), device=self.device)
            context_with_history = torch.cat((context, node_history_encoded[c_idx:c_idx+1].repeat(context.shape[0],1)), dim=-1) # (obj, dim+2)

            output = self.node_modules['/{}_edge_encoder'.format(context_type)](context_with_history)

            # output = F.dropout(output,
            #                     p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
            #                     training=(mode == ModeKeys.TRAIN))  # [obj, enc_edge_dim]
            combined_edges, _ = self.node_modules['/{}_edge_influence_encoder'.format(context_type)](output.unsqueeze(0),
                                                                            node_history_encoded[c_idx:c_idx+1])
            # combined_edges = F.dropout(combined_edges,
            #                             p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
            #                             training=(mode == ModeKeys.TRAIN))
            context_state.append(combined_edges)
        return torch.stack(context_state, dim=0)

    def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoded, batch_size):
        if len(encoded_edges) == 0:
            combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

        else:
            encoded_edges = torch.stack(encoded_edges, dim=1)
            combined_edges, _ = self.node_modules['/edge_influence_encoder'](encoded_edges,
                                                                            node_history_encoded)
            combined_edges = F.dropout(combined_edges,
                                        p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                        training=(mode == ModeKeys.TRAIN))

        return combined_edges

    def p_y_xz_bounding(self, mode, x, n_s_t0, z_stacked, bounding,
               num_samples, ph_limit=100, num_components=1, gmm_mode=False, measure=None):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param bounding: stop predicting when outside the bounding np.array(3,2)
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM3D. If mode is Predict, also samples from the GMM.
        """
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules['/decoder/rnn_cell']
        initial_h_model = self.node_modules['/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules['/decoder/state_action'](n_s_t0)

        state = initial_state

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
        outputs = []

        T = 1

        ph = 0
        while True:
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            if self.pred_state_length == 2:
                gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t[...,:1])
            else:
                gmm = GMM3D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]
            
            if ph == 0 and measure is not None:
                pred_cov = gmm.cov.reshape(num_components, self.pred_state_length, self.pred_state_length)
                user_cov = measure.covariance_matrix
                K = torch.bmm(pred_cov, torch.linalg.inv(pred_cov + user_cov))
                new_mus = gmm.mus.reshape(num_components, self.pred_state_length, 1) + torch.bmm(K, (measure.mean - gmm.mus.reshape(num_components, self.pred_state_length)).reshape(num_components, self.pred_state_length, 1))
                new_cov = pred_cov - torch.bmm(K, pred_cov)

                v_u = measure.mean[0,:]
                # pis = gmm.log_pis.exp().reshape(-1)
                pis = self.latent.p_dist.logits.exp().reshape(-1)
                v_mode = gmm.mus[:,0,:]
                p_r_xcvu = td.MultivariateNormal(v_mode.reshape(-1, self.pred_state_length), gmm.cov[:,0])
                pis_lh = p_r_xcvu.log_prob(v_u.reshape(-1,2).repeat(num_components,1)).exp()

                pis_posterior = pis*(pis_lh + 1e-5)
                pis_posterior = pis_posterior/pis_posterior.sum()

                self.latent.p_dist = td.Categorical(logits=pis_posterior.log())
                # gmm.pis_cat_dist = td.Categorical(logits=pis_posterior.log())
                # gmm.log_pis = pis_posterior.log().reshape(1,1,1,-1)

                gmm = gmm.from_log_pis_mus_cov_mats(gmm.log_pis, new_mus.reshape(num_components, self.pred_state_length), new_cov, kf=True)
                mu_t = gmm.mus.squeeze(0)
                log_sigma_t = gmm.log_sigmas.squeeze(0)
                corr_t = gmm.corrs

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t[...,0].reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, pred_dim
                ).permute(0, 2, 1, 3).reshape(-1, pred_dim * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, pred_dim
                ).permute(0, 2, 1, 3).reshape(-1, pred_dim * num_components))
            if self.pred_state_length == 2:
                corrs.append(
                    corr_t[...,:1].reshape(
                        num_samples, num_components, -1
                    ).permute(0, 2, 1).reshape(-1, num_components))
            else:
                corrs.append(
                    corr_t.reshape(
                        num_samples, num_components, -1
                    ).permute(0, 2, 1).reshape(-1, 3* num_components))


            # dec_inputs = [zx, mu_t]
            dec_inputs = [zx, a_t]
            outputs.append(a_t)
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
            ph += 1
            if ph > ph_limit:
                # print("out of ph_limit")
                break
            if ph == 1:
                pos_mus = self.dynamic.initial_conditions['pos'].unsqueeze(1)[:, None].repeat(num_samples, 1, num_components, 1)
            else:
                pos_mus += mus[-1].reshape(num_samples, -1, num_components, pred_dim) * self.dynamic.dt
            if pos_mus[...,0].mean() < bounding[0,0] or pos_mus[...,0].mean() > bounding[0,1] or pos_mus[...,1].mean() < bounding[1,0] or pos_mus[...,1].mean() > bounding[1,1]:
                break

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        outputs = torch.stack(outputs,dim=1)

        if self.pred_state_length == 3:
            a_dist = GMM3D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                        torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                        torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                        torch.reshape(corrs, [num_samples, -1, ph, num_components, 3]))
        else:
            a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                        torch.reshape(mus, [num_samples, -1, ph, num_components, pred_dim]),
                        torch.reshape(log_sigmas, [num_samples, -1, ph, num_components, pred_dim]),
                        torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        if self.hyperparams['dynamic']['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()            
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, a_dist, sampled_future
        else:
            return y_dist, a_dist, outputs
    
    def predict3(self,
                inputs,
                inputs_st,
                first_history_indices,
                bounding,
                num_samples,
                context={},
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                ph_limit=100,
                measure=None):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=None,
                                                        labels_st=None,
                                                        first_history_indices=first_history_indices,
                                                        context=context)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)


        # bounding = np.array([[-100,100], [-100,100], [100, z_T]])
        y_dist, v_dist, our_sampled_future = self.p_y_xz_bounding(mode, x, n_s_t0, z,
                                                bounding,
                                                num_samples,
                                                ph_limit,
                                                num_components,
                                                gmm_mode,
                                                measure=measure)
        if dist == True:
            return y_dist, v_dist, our_sampled_future
        return our_sampled_future