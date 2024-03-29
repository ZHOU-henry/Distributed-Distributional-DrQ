import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, dist_info):
        super().__init__()

        self.dist_info = dist_info

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True))
            #, nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True))
            #, nn.Linear(hidden_dim, 1))

##########################################################################################     
        self.fc_q1 = nn.Linear(hidden_dim, 1)
        self.fc_q2 = nn.Linear(hidden_dim, 1)

        self.fc_dist_q1 = nn.Linear(hidden_dim, self.dist_info['n_atoms'])
        self.fc_dist_q2 = nn.Linear(hidden_dim, self.dist_info['n_atoms'])
##########################################################################################

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

##########################################################################################
        if self.dist_info['type'] == 'categorical':
            q1 = F.softmax(self.fc_dist_q1(q1), dim=-1)
            q2 = F.softmax(self.fc_dist_q2(q2), dim=-1)  
            # Probability distribution over n_atom q_values
        else:
            q1 = self.fc_q1(q1)
            q2 = self.fc_q2(q2)
##########################################################################################
        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, actor_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, task_name,
                 init_temperature, action_dim, learnable_temperature, dist_info, batch_size):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.actor_target_tau = actor_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.task_name = task_name

############################################################################################
        #self.discount = discount
        self.batch_size = batch_size
        self.dist_info = dist_info

        self.critic_loss = nn.CrossEntropyLoss()

        if self.dist_info['type'] == 'categorical':
            self.dist_type = self.dist_info['type']
            self.v_min = self.dist_info['v_min']
            self.v_max = self.dist_info['v_max']
            self.n_atoms = self.dist_info['n_atoms']
            self.delta = (self.v_max-self.v_min)/float(self.n_atoms-1)
            self.bin_centers = torch.tensor(np.array([self.v_min+i*self.delta for i in range(self.n_atoms)]).reshape(-1,1), dtype=torch.float32, device=self.device)
        else:
            pass

############################################################################################
        #self.init_temperature = init_temperature
        #self.action_dim = action_dim
        #self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        #self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        #self.target_entropy = -self.action_dim
        #self.learnable_temperature = learnable_temperature
############################################################################################


        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        # self.actor_target = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        # self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.dist_info).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, self.dist_info).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)


        #self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)      ##zzh

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        # self.actor_target.train()

############################################################################################

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def project(self, next_dist, reward, discount): #,terminates):
        #try:
        #next_distr = next_distr_v.data.cpu().numpy()
        target_z_dist = next_dist.data.cpu().numpy()
        rewards = reward.data.cpu().numpy()
        discount = discount.mean().cpu().numpy()
        #terminates = terminates.reshape(-1).astype(bool)
        #dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        #batch_size = len(rewards)
        proj_distr = np.zeros((self.batch_size, self.n_atoms), dtype=np.float32)
    

        #pdb.set_trace()

        for atom in range(self.n_atoms):
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * discount))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = (u == l).reshape(-1)
            #a = l[eq_mask]
            #b = target_z_dist[eq_mask, atom]
            proj_distr[eq_mask, l[eq_mask]] += target_z_dist[eq_mask, atom]
            ne_mask = (u != l).reshape(-1)
            proj_distr[ne_mask, l[ne_mask]] += target_z_dist[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += target_z_dist[ne_mask, atom] * (b_j - l)[ne_mask]
            '''
            if terminates.any():
                proj_distr[terminates] = 0.0
                tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[terminates]))
                b_j = (tz_j - self.v_min) / self.delta
                l = np.floor(b_j).astype(np.int64)
                u = np.ceil(b_j).astype(np.int64)
                eq_mask = (u == l).astype(bool)
                eq_dones = terminates.copy()
                eq_dones[terminates] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l] = 1.0
                ne_mask = (u != l).astype(bool)
                ne_dones = terminates.copy()
                ne_dones[terminates] = ne_mask.astype(bool)
                if ne_dones.any():
                    proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
                '''
        #except Exception as e:
            #print(e)
        
        return proj_distr

############################################################################################

    def load_model(self, encoder = True, actor = True, critic = False, critic_target = False):

        epochs = 8
        task_name = 'walker_walk'
        root_dir='/home/henry/Desktop/model_parameters/'+f'{task_name}'

        if encoder:
            self.encoder = torch.load(f'{root_dir}/model_encoder/_model_encoder'+f'_Epochs_{epochs}_.pt')

        if actor:
           self.actor  = torch.load(f'{root_dir}/model_action/_model_action'+f'_Epochs_{epochs}_.pt')

        if critic:
            self.critic = torch.load(f'{root_dir}/model_critic/_model_critic'+f'_Epochs_{epochs}_.pt')

        if critic_target:
            self.critic_target = torch.load(f'{root_dir}/model_critic_target/_model_critic_target'+f'_Epochs_{epochs}_.pt')
        
############################################################################################
        

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        if self.dist_type == 'categorical':
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)

                project_Q1 = self.project(target_Q1, reward, discount)
                project_Q2 = self.project(target_Q2, reward, discount)
            
            Q1, Q2 = self.critic(obs, action)
            Q1_dist_loss = -(torch.tensor(project_Q1, requires_grad=False).to(self.device)
                            *torch.log(Q1+1e-010)).sum(dim=1).mean()
            Q2_dist_loss = -(torch.tensor(project_Q2, requires_grad=False).to(self.device)
                            *torch.log(Q2+1e-010)).sum(dim=1).mean()

            critic_loss = Q1_dist_loss + Q2_dist_loss

        else:
            with torch.no_grad():
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)

            
                target_V = torch.min(target_Q1, target_Q2)
                target_Q = reward + (discount * target_V)

            Q1, Q2 = self.critic(obs, action)
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

############################################################################################
            '''
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)        
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob   ##zzh
            target_Q = reward + (discount * target_V)
            target_Q = target_Q.detach()
            '''
############################################################################################

        if self.use_tb:

            if self.dist_type == 'categorical':
                metrics['Q1_dist_loss'] =  Q1_dist_loss.item()
                metrics['Q2_dist_loss'] =  Q2_dist_loss.item()
            else:
                metrics['critic_target_q'] = target_Q.mean().item()

            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)

        if self.dist_type == 'categorical':
            Q1_actor_loss = -Q1.matmul(self.bin_centers).mean()
            Q2_actor_loss = -Q2.matmul(self.bin_centers).mean()
            
            actor_loss = torch.max(Q1_actor_loss, Q2_actor_loss)
        else:
            Q = torch.min(Q1, Q2)
            actor_loss = -Q.mean()
        

############################################################################################
        #actor_loss = (self.alpha.detach() * log_prob - Q).mean()   ## SAC
############################################################################################

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

############################################################################################
        '''
        if self.learnable_temperature:
            self.log_alpha_opt.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            #logger.log('train_alpha/loss', alpha_loss, step)
            #logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_opt.step()
        '''
############################################################################################

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

############################################################################################

        ## update actor target
        # utils.soft_update_params(self.actor, self.actor_target, self.actor_target_tau)

############################################################################################


        #########################################################################################
        ########store model, load model in visual.py###################
        root_dir='/home/henry/Desktop/model_parameters/'+f'{self.task_name}'  #+'_transfer'
        if step % 50000 == 0:
            epochs = int(step/50000)
            torch.save(self.actor,   f'{root_dir}/model_action/_model_action_Epochs_{epochs}_.pt')
            torch.save(self.encoder, f'{root_dir}/model_encoder/_model_encoder_Epochs_{epochs}_.pt')
            torch.save(self.critic,   f'{root_dir}/model_critic/_model_critic_Epochs_{epochs}_.pt')
            torch.save(self.critic_target, f'{root_dir}/model_critic_target/_model_critic_target_Epochs_{epochs}_.pt')
        #########################################################################################

        return metrics
