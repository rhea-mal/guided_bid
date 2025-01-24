import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import numpy as np

import collections

import copy

import ipdb

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

from scipy.spatial.transform import Rotation as R

from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env as robomimic_lowdim_create_env

import random

import h5py

## HIGH VARIANCE
# OFFSET_SCALE = 0.1
# GRASP_POS = 0.01
# PICK_ROT = 0.01
# PEG_LAT = 0.01
# PEG_HEIGHT=0.06
# NUT_HEIGHT=0.06

# ## MEDIUM VARIANCE
# OFFSET_SCALE = 0.08
# GRASP_POS = 0.008
# PICK_ROT = 0.008
# PEG_LAT = 0.008
# PEG_HEIGHT=0.04
# NUT_HEIGHT=0.04

##  NEW MEDIUM 2 VARIANCE
# OFFSET_SCALE = 0.065
# GRASP_POS = 0.0065
# PICK_ROT = 0.0065
# PEG_LAT = 0.0065
# PEG_HEIGHT=0.03
# NUT_HEIGHT=0.03

# LOW VARIANCE
# OFFSET_SCALE = 0.05
# GRASP_POS = 0.005
# PICK_ROT = 0.005
# PEG_LAT = 0.005
# PEG_HEIGHT=0.02
# NUT_HEIGHT=0.02

# NEW LOW VARIANCE 2
OFFSET_SCALE = 0.025
GRASP_POS = 0.0025
PICK_ROT = 0.0025
PEG_LAT = 0.0025
PEG_HEIGHT=0.01
NUT_HEIGHT=0.01

## NO VARIANCE
# OFFSET_SCALE = 0.0
# GRASP_POS = 0.0
# PICK_ROT = 0.0
# PEG_LAT = 0.0
# PEG_HEIGHT = 0.0
# NUT_HEIGHT = 0.0


class ScriptedPolicy:

    def __init__(self, env=None):
        self.env = env
        self.rotation_transformer = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        self.reset()

    def predict_action(self, obs_dict):
        if self.start is None:
            self.start = obs_dict[-1][14:14+6] 

        action = np.zeros((1,7))

        action[0][:3] = self.start[:3]
        action[0][2] -= self.step_num*.001

        action[0][4] = 3.14
        action[0][5] = 1.5

        action[0][3:6] = self.rotation_transformer.forward(action[:,3:6].reshape((1,3)))
        return action

    def reset(self):
        self.step_num = 0
        # self.start = None
        random_offset = np.random.uniform(low=-0.05, high=0.05, size=3)  # Adjust range as needed
        self.start = np.array([0.5, 0.0, 0.3]) + random_offset  # Default start position + random offset
        
        # Randomize the gripper open/close status
        self.gripper_state = np.random.choice([-1, 1])  # -1 for open, 1 for closed
        
        self.step_num = 0

class SmoothScriptedPolicy(ScriptedPolicy):

    def __init__(self, env=None):
        super().__init__(env)

        self.last_action = None

    def generate_trajectory(self):
        self.nut_pos = self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.obj_body_id['SquareNut']]
        self.nut_ori = self.env.env.env.env.env.sim.data.body_xmat[self.env.env.env.env.env.obj_body_id['SquareNut']].reshape(3,3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')

        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

        peg_pos = np.array(self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.peg1_body_id])

        above_peg_pos = peg_pos.copy()
        above_peg_pos[2] += .25
        above_peg_pos[1] -= 0.032

        if self.nut_rot > np.pi/2:
            pick_rot = self.nut_rot - np.pi
            drop_rot = 0.
        elif self.nut_rot < -np.pi/2:
            pick_rot = np.pi + self.nut_rot
        else:
            pick_rot = self.nut_rot

        grasp_pos = self.nut_pos.copy()
        grasp_pos[2] -= 0.062

        if np.abs(self.nut_rot) < 4.:
            grasp_pos[1] -= 0.029*np.cos(pick_rot)
            grasp_pos[0] += 0.029*np.sin(pick_rot)

        above_nut_pos = grasp_pos.copy()
        above_nut_pos[2] += .2

        self.trajectory = [{"t":30, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, -1.])])}, # changed
                           {"t":60, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, -1.])])}, #changed
                           {"t":90, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, 1.])])},
                           {"t":120, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, 1.])])},
                           {"t":140, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
                           {"t":180, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
                           {"t":185, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
                           {"t":190, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, -1.])])},
                           {"t":500, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2, -1.])])}]

    def reset(self):
        super().reset()
        self.mode = random.choice(['left', 'right'])  # Randomly select a curvature mode
        # self.mode = random.choice(['left', 'middle', 'right'])  # Randomly select a curvature mode
        # self.mode = 'middle'

        print('mode:', self.mode)
        self.generate_trajectory(self.mode)
        # self.trajectory_bkp = copy.deepcopy(self.trajectory)
        cur_xyz = self.env.env.env.env.get_observation()['robot0_eef_pos']

        self.last_t = 0
        self.last_action = np.concatenate([cur_xyz, np.array([0., 3.14159, 3.14159/2, -1.])])# changed 0

    def replan(self):
        super().reset()
        self.generate_trajectory(self.mode)
        cur_xyz = self.env.env.env.env.get_observation()['robot0_eef_pos']
        self.last_t = 0
        self.last_action = np.concatenate([cur_xyz, np.array([0., 3.14159, 3.14159/2, -1.])])# changed 0

    def predict_action(self, obs_dict):

        if self.step_num == self.trajectory[0]["t"]:
            self.last_action = self.trajectory[0]['action']
            self.last_t = self.trajectory[0]['t']
            self.trajectory.pop(0)
        
        action = self.last_action + (self.trajectory[0]['action'] - self.last_action)*(self.step_num - self.last_t)/(self.trajectory[0]["t"] - self.last_t)
        action = action.reshape((1,7))

        action[0][3:6] = self.rotation_transformer.forward(action[:,3:6].reshape((1,3)))

        # Control gripper explicitly based on task phase
        grasp_threshold = 0.01  # Distance threshold for grasp alignment
        # if np.linalg.norm(obs_dict['robot0_eef_pos'] - self.nut_pos) < grasp_threshold:
        #     # Grasp phase: Close the gripper
        #     action[0, 6] = 1  # Close gripper
        # # elif self.step_num in [self.trajectory[-1]["t"]]:  # Final release step
        # #     # Release phase: Open the gripper
        # #     action[0, 6] = -1  # Open gripper
        # else:
        #     # Default: Keep gripper state unchanged
        #     action[0, 6] = -1


        self.step_num += 1
        return action

class SquareSideScriptedPolicy(SmoothScriptedPolicy):

    def generate_curved_trajectory(self, start, end, mode='middle', vmax=0.0):
        """
        Generates a curved trajectory between two points.
        :param start: Starting position (x, y, z).
        :param end: Ending position (x, y, z).
        :param mode: 'left', 'right', or 'middle' for curvature direction.
        :return: List of trajectory waypoints.
        """
        control_point = (start + end) / 2
        offset = np.random.uniform(low=-vmax, high=vmax, size=3)
        offset[0] = np.random.uniform(low=-OFFSET_SCALE, high=OFFSET_SCALE)
        offset[1] = np.random.uniform(low=-OFFSET_SCALE, high=OFFSET_SCALE)  
        offset[2] = np.random.uniform(low=-OFFSET_SCALE, high=OFFSET_SCALE)  
    
        # offset = np.array([0.02, 0.02, 0.0])  # Example offset for curvature
        if mode == 'left':
            control_point += offset
        elif mode == 'right':
            control_point -= offset

        # Generate intermediate points (e.g., quadratic bezier curve)
        t_values = np.linspace(0, 1, num=30)  

        trajectory = [
            (1 - t) ** 2 * start + 2 * (1 - t) * t * control_point + t ** 2 * end
            for t in t_values
        ]
        return np.array(trajectory)

    def generate_trajectory(self, mode='middle'):
        self.nut_pos = self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.obj_body_id['SquareNut']]
        self.nut_ori = self.env.env.env.env.env.sim.data.body_xmat[self.env.env.env.env.env.obj_body_id['SquareNut']].reshape(3, 3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')

        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

        peg_pos = np.array(self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.peg1_body_id])

        above_peg_pos = peg_pos.copy()
        above_peg_pos[2] += 0.25 + np.random.uniform(-PEG_HEIGHT, PEG_HEIGHT)
        above_peg_pos[1] -= 0.032 + np.random.uniform(-PEG_LAT, PEG_LAT)

        if self.nut_rot > np.pi / 2:
            pick_rot = self.nut_rot - np.pi 
        elif self.nut_rot < -np.pi / 2:
            pick_rot = np.pi + self.nut_rot
        else:
            pick_rot = self.nut_rot

        # Compute the initial grasp position
        grasp_pos = self.nut_pos.copy()
        # grasp_pos[2] -= 0.062
        grasp_pos[2] -= 0.062 + np.random.uniform(-GRASP_POS, GRASP_POS)

        if np.abs(self.nut_rot) < 4.:
            grasp_pos[1] -= 0.029 * np.cos(pick_rot) + np.random.uniform(-PICK_ROT, PICK_ROT)
            grasp_pos[0] += 0.029 * np.sin(pick_rot) + np.random.uniform(-PICK_ROT, PICK_ROT)

        above_nut_pos = grasp_pos.copy()
        above_nut_pos[2] += 0.2 + np.random.uniform(-NUT_HEIGHT, NUT_HEIGHT)

        # Create trajectory with curvature applied to specific segments
        self.trajectory = [{"t": 30, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, -1.])])}] ### CHANGED FROM 0
        self.trajectory = [{"t": 40, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, -1.])])}] ### CHANGED FROM 0

        # Generate curved trajectories for selected segments
        curved_traj_1 = self.generate_curved_trajectory(above_nut_pos, grasp_pos, mode, 0.1)
        curved_traj_2 = self.generate_curved_trajectory(grasp_pos, above_nut_pos, mode, 0.2)
        curved_traj_3 = self.generate_curved_trajectory(above_nut_pos, above_peg_pos, mode, 0.3)

        # Segment 1: Above Nut -> Grasp Position
        for t, point in enumerate(curved_traj_1, start=50):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2 - self.nut_rot, -1.])])}) ## cHNAGED FROM 0

        # Grasp Position -> Grasp Position with Force
        self.trajectory.append({"t": 85, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159 / 2 - self.nut_rot, 1.])])})

        # Segment 2: Grasp Position -> Above Nut
        for t, point in enumerate(curved_traj_2, start=100):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2 - self.nut_rot, 1.])])})

        # Above Nut (Aligned) -> Above Peg
        for t, point in enumerate(curved_traj_3, start=150):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2, 1.])])})

        # End segment: Above Peg (Static Points)
        self.trajectory += [
            # {"t": 200, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159 / 2, 1.])])},
            {"t": 200, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159 / 2, -1.])])},
            {"t": 500, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159 / 2, -1.])])}
        ]

def create_robomimic_env():
    dev = True
    cam_h = 84 if not dev else 640
    cam_w = 84 if not dev else 640

    env_meta = {'env_name': 'NutAssemblySquare', 'type': 1, 'env_kwargs': {'has_renderer': False, 'has_offscreen_renderer': False, 'ignore_done': True, 'use_object_obs': True, 'use_camera_obs': False, 'control_freq': 20, 'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': False, 'interpolation': None, 'ramp_ratio': 0.2}, 'robots': ['Panda'], 'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'reward_shaping': False}}
    obs_keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

    env = robomimic_lowdim_create_env(env_meta, obs_keys)

    fps = 10
    crf = 22
    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)
    env_n_obs_steps = 1
    env_n_action_steps = 1
    max_steps = 500

    # hard reset doesn't influence lowdim env
    # robomimic_env.env.hard_reset = False
    env = MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicLowdimWrapper(
                    env=env,
                    obs_keys=obs_keys,
                    init_state=None,
                    render_hw=[cam_h, cam_w],
                    render_camera_name='agentview'
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=env_n_obs_steps,
            n_action_steps=env_n_action_steps,
            max_episode_steps=max_steps
        )
    
    return env, env_meta

def create_policy(env, mode='both'):
    
    if mode == 'both':
        mode = np.random.choice(['side', 'corner'])
    
    if mode == 'corner':
        return SquareCornerScriptedPolicy(env), "corner"
    elif mode == 'side':
        return SquareSideScriptedPolicy(env), "side"
    else:
        raise NotImplementedError("not valid scripted policy type")

@click.command()
# @click.option('-o', '--output_dir', default='data/uniform')
@click.option('-o', '--output_dir', default='data/diverse/test_randomstart_new_low2')
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--num_episodes', type=int, default=200)
@click.option('-s', '--num_side', type=int, default=5)
@click.option('--seed', type=int, default=0)
def main(output_dir, device, num_episodes, num_side, seed):
    # Set up seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    num_vids = 0

    print(f"Seed set to: {seed}")

    #if os.path.exists(output_dir):
        #click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    env, env_cfg = create_robomimic_env()

    dataset_file = h5py.File(pathlib.Path(output_dir).joinpath("demos.hdf5"), 'w')
    dataset_data_group = dataset_file.create_group('data')

    dataset_data_group.attrs['env_args'] = json.dumps(env_cfg.copy())

    total_rollouts = 0
    successful_rollouts = 0

    modes = ['side']*num_side + ['corner']*(num_episodes - num_side)
    random.shuffle(modes)

    for ep_num in range(num_episodes):

        success = False
        # policy, policy_type = create_policy(env, mode=modes[ep_num])
        policy, policy_type = create_policy(env, mode='side')

        print('policy_type:', policy_type)

        while not success:
            #NOTE Dataloader in this repo only needs obs and actions to train (see diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py)
            trajectory = {"obs": [],
                        "actions": [],}

            #start video
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()

            if ep_num < 10:
                filename = pathlib.Path(output_dir).joinpath(
                        f"vids/episode{ep_num}.mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env.env.file_path = filename
            else:
                env.env.file_path = None

            #reset env with seed
            assert isinstance(env.env.env, RobomimicLowdimWrapper)
            env.env.env.init_state = None
            env.seed(np.random.randint(0, 10000000))

            obs = env.reset()

            policy.reset()
            policy.replan()

            rews = []
            for step in range(300):

                obs_dict = {"object": obs[0,:14],
                        'robot0_eef_pos':obs[0,14:17],
                        'robot0_eef_quat': obs[0,17:21],
                        'robot0_gripper_qpos': obs[0, 21:]}

                action = policy.predict_action(obs_dict)



                trajectory['obs'].append(obs_dict)
                trajectory['actions'].append(action[0].copy())

                #TODO: why do two methods of getting raw_obs disagree slightly?
                #raw_obs = env.env.env.env.get_observation()
                obs, reward, done, info = env.step(action)
                rews.append(reward)

                if reward >= 1.0:
                    success = True
                    break

            if success:
                print(f"SUCCESS in episode {ep_num}.")
                successful_rollouts += 1

                # Save successful rollout
                ep_group = dataset_data_group.create_group(f'demo_{successful_rollouts}')
                obs_group = ep_group.create_group('obs')
                for obs_kwrd in trajectory['obs'][0].keys():
                    obs_kwrd = str(obs_kwrd)
                    obs_array = np.stack([od[obs_kwrd] for od in trajectory['obs']], axis=0)
                    obs_group.create_dataset(obs_kwrd, data=obs_array)
                action_array = np.stack([a for a in trajectory['actions']], axis=0)
                ep_group.create_dataset('actions', data=action_array)
                ep_group.attrs['scripted_policy_type'] = policy_type
                
            total_rollouts += 1

        print(f"Episode {ep_num} complete", max(rews), policy_type)

        # trajectory['actions'] list n x 7 
        # trajectory['actions'] list n x dict {'object', 'robot0_eef_pos''}

    dataset_data_group.attrs['data collection'] = f"{successful_rollouts} of {total_rollouts} total rollouts successful"
    dataset_file.close()
    print(f"{successful_rollouts} of {total_rollouts} total rollouts successful")

if __name__ == '__main__':
    main()