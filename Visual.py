import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_control import suite
from dm_control import viewer


from collections import deque
from typing import Any, NamedTuple

import dm_env
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from logger import  AverageMeter


#########################################################################################################
import sys
import pathlib
__file__='/home/henry/Desktop/drqv2-main/train.py'
sys.path.append(str(pathlib.Path(__file__)))
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

#########################################################################################################
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        c, h, w = x.size()
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
        base_grid = base_grid.unsqueeze(0).repeat(1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


#########################################################################################################
"""
walker_walk: 20
walker_run: 50
cheetah_run: 47
#hopper_hop: 51
cartpole_balance: 18
cartpole_swingup: 18
#quadruped_run: 51
humanoid_walk: 501
humanoid_run: 501
"""
epochs = 10
task_name = 'walker_walk'

root_dir='/home/henry/Desktop/model_parameters/'+f'walker_walk'

#root_dir='/home/henry/Desktop/model_parameters/'+f'{task_name}'

if 1:
    model_encoder = torch.load(f'{root_dir}/model_encoder/_model_encoder'+f'_Epochs_{epochs}_.pt')
    model_action  = torch.load(f'{root_dir}/model_action/_model_action'+f'_Epochs_{epochs}_.pt')

if 0:
    _reward = '0.9250'
    model_encoder = torch.load(f'{root_dir}/model_encoder/_model_encoder_Epochs_1_.pt')
    model_action  = torch.load(f'{root_dir}/model_action/_model_action_reward_{_reward}_.pt')
    # model_action  = torch.load(f'{root_dir}/model_action/_model_action_reward_{_reward}_.pt')


#########################################################################################################
domain, task = task_name.split('_', 1)
env1 = suite.load(domain,task)
#########################################################################################################
def policy(time_step, sample = False, augmentation=0):

    ##print(time_step.step_type)
    #print(time_step.reward)
    #print(time_step.reward)
    if time_step.reward == None:
        rrr
    else:
        rrr.update(value=time_step.reward)
    print(rrr.value())
    # image
    _size=(84, 84)
    _image=env1.physics.render(*_size, camera_id=0)

    # obs
    if time_step.step_type == StepType.FIRST:
        _frames = deque([], maxlen=3)
        pixels = _image.transpose(2, 0, 1).copy()
        for _ in range( 3 ):
            _frames.append(pixels)
        assert len(_frames) == 3
        obs = np.concatenate(list(_frames), axis=0)
        time_step.observation['_frames'] = _frames
        #time_step._replace(observation=obs)
    else:
        pixels = _image.transpose(2, 0, 1).copy()
        _frames = time_step.observation['_frames']
        _frames.append(pixels)
        assert len(_frames) == 3
        obs = np.concatenate(list(_frames), axis=0)

    obs = torch.as_tensor(obs, device='cuda')
    if augmentation == 1:
        aug = RandomShiftsAug(pad=4)
        obs = aug(obs.float())
    else:
        pass
    obs = model_encoder(obs.unsqueeze(0)).to('cuda')
    dist = model_action(obs, 0.1)
    if sample:
        action = dist.sample(clip=0.3)
    else:
        action = action = dist.mean
    action_cpu=action.cpu().detach().numpy()[0]

    return action_cpu


class reward:
    def __init__(self):
        self.reward_total = 0

    def __call__(self, reward):
        self.reward_total += reward
        return self.reward_total


'''
    #obs1 = time_step.observation
    #obs1['image']=_image
    obs2 = {
                k: v.astype(np.float32) if (
                    isinstance(v, np.ndarray) and v.dtype == np.float64) else v
                for k, v in obs1.items()}
    obs3 = {
                k: torch.as_tensor(np.array(v), device='cuda')[None]
                for k, v in obs2.items()}
  
    if augmentation == 1:
        aug = RandomShiftsAug(pad=4)
        image = obs3['image'].float()
        image = image.permute((0, 3, 1, 2))
        image = aug(image)
        image = image.permute((0, 2, 3, 1))
        obs3['image'] = image
    else:
        pass
  
    # action
    h = model_encoder(obs3).to('cuda') 
    dis = model_action(h, 0.1)
    action = dis.sample(clip = 0.3) 

    # return
    action_cpu=action.cpu().detach().numpy()[0]
    ##print(action_cpu)
    return action_cpu
    '''

#########################################################################################################
rrr= AverageMeter()
viewer.launch(env1, policy)

#########################################################################################################
