"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import random
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import numpy as np
import collections
import copy
import ipdb

from scipy.spatial.transform import Rotation as R
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.env_runner.robomimic_lowdim_runner import create_env as robomimic_lowdim_create_env

import h5py
import do_mpc
from casadi import vertcat, SX


class MPCPolicy:

    def __init__(self, env=None, horizon=20):
        self.env = env
        self.horizon = horizon
        self.rotation_transformer = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        self.reset()
        self.mpc = self.setup_mpc(horizon)
    def setup_mpc(self, horizon):
        # Define the MPC model
        model = do_mpc.model.Model('continuous')

        # Define state variables
        pos = model.set_variable('_x', 'pos', (3, 1))  # Position: (x, y, z)
        ori = model.set_variable('_x', 'ori', (3, 1))  # Orientation: (roll, pitch, yaw)

        # Define control variables
        vel = model.set_variable('_u', 'vel', (3, 1))  # Velocity: (vx, vy, vz)
        ang_vel = model.set_variable('_u', 'ang_vel', (3, 1))  # Angular velocity: (wx, wy, wz)

        # Define parameters for the goal
        goal = model.set_variable('_p', 'goal', (6, 1))  # Combined goal for position (3) and orientation (3)

        # Define system dynamics
        model.set_rhs('pos', vel)  # Position dynamics
        model.set_rhs('ori', ang_vel)  # Orientation dynamics

        # Finalize the model setup
        model.setup()

        # Initialize MPC controller
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(n_horizon=horizon, t_step=0.05)

        # Define the objective function
        mpc.set_objective(
            lterm=((pos - goal[:3, 0]).T @ (pos - goal[:3, 0])) + ((ori - goal[3:, 0]).T @ (ori - goal[3:, 0])),
            mterm=((pos - goal[:3, 0]).T @ (pos - goal[:3, 0])) + ((ori - goal[3:, 0]).T @ (ori - goal[3:, 0])),
        )

        # Define control effort penalty (regularization)
        mpc.set_rterm(vel=1e-2, ang_vel=1e-2)

        # Suppress verbose solver output
        mpc.set_param(
            nlpsol_opts={
                "ipopt.print_level": 0,  # Suppress solver output
                "ipopt.sb": "yes",      # Suppress banner
                "print_time": 0         # Suppress timing output
            }
        )

        # Create a parameter template for updating goals
        p_template = mpc.get_p_template(1)
        p_template['_p', 0, 'goal'] = np.zeros((6, 1))  # Initialize goal as a single vector for pos and ori
        mpc.set_p_fun(lambda t_now: p_template)

        # Finalize MPC setup
        mpc.setup()

        # Set initial guesses
        mpc.x0 = np.zeros((6, 1))  # Combined state for position and orientation
        mpc.u0 = np.zeros((6, 1))  # Combined control for velocity and angular velocity
        mpc.set_initial_guess()

        return mpc

    def reset(self):
        self.step_num = 0
        self.trajectory = None
        self.nut_pos = None
        self.nut_ori = None
        self.peg_pos = None
        self.grasp_pos = None
        self.above_nut_pos = None
        self.above_peg_pos = None
        self.pick_rot = None
        self.recompute_waypoints()

    def recompute_waypoints(self):
        """
        Compute key waypoints such as grasp position, above nut position, and above peg position.
        """
        # Extract nut position and orientation
        self.nut_pos = self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.obj_body_id['SquareNut']]
        self.nut_ori = self.env.env.env.env.env.sim.data.body_xmat[self.env.env.env.env.env.obj_body_id['SquareNut']].reshape(3, 3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')
        self.nut_rot = self.nut_ori[2]

        # Determine pick rotation
        if self.nut_rot > np.pi / 2:
            self.pick_rot = self.nut_rot - np.pi
        elif self.nut_rot < -np.pi / 2:
            self.pick_rot = np.pi + self.nut_rot
        else:
            self.pick_rot = self.nut_rot

        # Compute grasp position
        self.grasp_pos = self.nut_pos.copy()
        self.grasp_pos[2] -= 0.062
        if np.abs(self.nut_rot) < 4.:
            self.grasp_pos[1] -= 0.029 * np.cos(self.pick_rot)
            self.grasp_pos[0] += 0.029 * np.sin(self.pick_rot)

        # Compute above nut position
        self.above_nut_pos = self.grasp_pos.copy()
        self.above_nut_pos[2] += 0.2

        # Extract peg position
        self.peg_pos = np.array(self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.peg1_body_id])

        # Compute above peg position
        self.above_peg_pos = self.peg_pos.copy()
        self.above_peg_pos[2] += 0.25
        self.above_peg_pos[1] -= 0.032

        # Debugging info
        # print(f"Nut position: {self.nut_pos}, Nut orientation: {self.nut_ori}, Pick rotation: {self.pick_rot}")
        # print(f"Grasp position: {self.grasp_pos}, Above nut position: {self.above_nut_pos}")
        # print(f"Peg position: {self.peg_pos}, Above peg position: {self.above_peg_pos}")


    def predict_action(self, obs_dict):
        """
        Predicts the next action using MPC for the trajectory phases.
        """
        # Parse observation data from array format to meaningful components
        current_pos = obs_dict[0, 14:17]  # Extracting `robot0_eef_pos`
        current_ori = R.from_quat(obs_dict[0, 17:21]).as_euler('xyz')  # Extracting and converting `robot0_eef_quat`

        # Total steps and phases
        total_steps = 300
        phase_steps = total_steps // 4

        # Define goal position based on phase

        if self.step_num < phase_steps:
            # print("phase 1: open gripper")
            goal_pos = (1 - self.step_num / phase_steps) * self.above_nut_pos + \
                       (self.step_num / phase_steps) * self.grasp_pos
            gripper_action = -1.0  # Open gripper
        elif self.step_num < 2 * phase_steps:
            # print("phase 2: grasp position")
            goal_pos = self.grasp_pos
            gripper_action = 1.0  # Close gripper
        elif self.step_num < 3 * phase_steps:
            # print("phase 3: gripper close")
            goal_pos = (1 - (self.step_num - 2 * phase_steps) / phase_steps) * self.grasp_pos + \
                       ((self.step_num - 2 * phase_steps) / phase_steps) * self.above_nut_pos
            gripper_action = 1.0  # Hold gripper closed
        else:
            # print("phase 4: grasp release")
            goal_pos = (1 - (self.step_num - 3 * phase_steps) / phase_steps) * self.above_nut_pos + \
                       ((self.step_num - 3 * phase_steps) / phase_steps) * self.above_peg_pos
            gripper_action = -1.0  # Release gripper

        goal_ori = np.array([0., np.pi, np.pi / 2 - self.pick_rot])

        # Update MPC parameters
        p_template = self.mpc.get_p_template(1)
        p_template['_p', 0, 'goal'][:3, 0] = goal_pos.flatten()
        p_template['_p', 0, 'goal'][3:, 0] = goal_ori.flatten()
        self.mpc.set_p_fun(lambda t_now: p_template)

        # Predict action using MPC
        mpc_action = self.mpc.make_step(np.hstack([current_pos, current_ori]).reshape(-1, 1))

        # Format action
        formatted_action = np.zeros((1, 7))
        formatted_action[0, :3] = mpc_action[:3].flatten()
        formatted_action[0, 3:6] = mpc_action[3:6].flatten()
        formatted_action[0, 6] = gripper_action

        self.step_num += 1
        return formatted_action


def create_policy(env):
    return MPCPolicy(env)

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

@click.command()
# @click.option('-o', '--output_dir', default='data/uniform')
@click.option('-o', '--output_dir', default='data/diverse/test')
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
        print(f"Starting episode {ep_num}...")
        success = False
        policy = create_policy(env)

        while not success:
            print(f"  Attempting rollout for episode {ep_num}...")
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
            print("  Environment reset, starting rollout...")

            rews = []
            for step in range(300):

                obs_dict = {"object": obs[0,:14],
                        'robot0_eef_pos':obs[0,14:17],
                        'robot0_eef_quat': obs[0,17:21],
                        'robot0_gripper_qpos': obs[0, 21:]}
                action = policy.predict_action(obs)

                trajectory['obs'].append(obs_dict)
                trajectory['actions'].append(action[0].copy())

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

        print(f"Episode {ep_num} complete", max(rews))

    dataset_data_group.attrs['data collection'] = f"{successful_rollouts} of {total_rollouts} total rollouts successful"
    dataset_file.close()
    print(f"{successful_rollouts} of {total_rollouts} total rollouts successful")

if __name__ == '__main__':
    main()