import numpy as np
from torch import distributions

from dm_control import suite
from dm_control import viewer
from dm_control import composer
from dm_control import manipulation
from dm_control.locomotion import arenas, tasks
from dm_control.locomotion.walkers import cmu_humanoid


walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(observable_options={'egocentric_camera': dict(enabled=True)})

arena = arenas.WallsCorridor(wall_gap=3., wall_width=2.5, #distributions.Uniform(2., 3.),
    wall_height=3, #distributions.Uniform(2.5, 3.5), 
    corridor_width=4., corridor_length=30.)

task = tasks.RunThroughCorridor(walker=walker, arena=arena, walker_spawn_position=(0.5, 0, 0),target_velocity=3.0, 
            physics_timestep=0.005, control_timestep=0.03, contact_termination=False, terminate_at_height=None)

environment = composer.Environment(time_limit=10, task=task, strip_singleton_obs_buffer_dim=True)

# ‘ALL‘ i'''s a tuple containing the names of all of the environments.
#print('\n'.join(manipulation.ALL))
#print('\n'.join(manipulation.get_environments_by_tag('vision')))
#env = manipulation.load('reassemble_3_bricks_fixed_order_vision', seed=1)
#env = suite.load(domain_name="lqr", task_name="lqr_6_2")

#environment = suite.load('dog','fetch')

action_spec = environment.action_spec()

def random_policy(time_step):
  print(time_step.reward)
  del time_step  
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

viewer.launch(environment, policy=random_policy)
