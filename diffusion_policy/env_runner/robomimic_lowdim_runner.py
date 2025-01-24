import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import pdb
from diffusion_policy.sampler.single import coherence_sampler, ema_sampler, cma_sampler
from diffusion_policy.sampler.multi import contrastive_sampler, bidirectional_sampler
from diffusion_policy.sampler.condition import NoiseGenerator
import matplotlib.pyplot as plt

def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env

def compute_trajectory_smoothness(trajectory):
    """
    Compute smoothness for a trajectory based on the mean magnitude of 
    differences between consecutive actions.
    
    Args:
        trajectory (np.ndarray): Trajectory of shape (timesteps, action_dim).
    
    Returns:
        float: Smoothness score (lower values indicate smoother trajectories).
    """
    differences = np.diff(trajectory, axis=0)  # Differences between consecutive actions
    magnitudes = np.linalg.norm(differences, axis=1)  # Norm of differences
    smoothness = np.mean(magnitudes)  # Average magnitude
    return smoothness

def save_histogram(smoothness_values, output_path, bins=20):
    """
    Saves a histogram of smoothness values as a percentage of total.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Calculate the weights for percentage representation
    weights = (1 / len(smoothness_values)) * 100 * np.ones(len(smoothness_values))

    plt.figure(figsize=(8, 6))
    plt.hist(smoothness_values, bins=bins, color='blue', edgecolor='black', alpha=0.7, weights=weights)
    plt.title("Smoothness Distribution")
    plt.xlabel("Smoothness")
    plt.ylabel("Percentage (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the histogram
    print("Histogram saved to:", output_path)
    plt.savefig(output_path + '.png', format='png')
    plt.close()


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)
        self.output_dir =output_dir

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
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

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                # init_state = f[f'data/demo_{train_idx}/states'][0]
                if 'states' in f[f'data/demo_{train_idx}']:
                    init_state = f[f'data/demo_{train_idx}/states'][0]
                else:
                    print(f"Warning: 'states' key not found for demo_{train_idx}. Skipping initialization state.")
                    init_state = None


                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.sampler = None
        self.n_samples = 0
        self.topk = 0
        self.weak = None
        self.noise = 0.0
        self.decay = 1.0

    def set_sampler(self, sampler, nsample=1, topk=1, noise=0.0, decay=1.0, dist_dir=None):
        self.sampler = sampler
        self.n_samples = nsample
        self.topk = topk
        self.noise = noise
        self.decay = decay
        self.dist_dir = dist_dir
        if noise > 0:
            self.disruptor = NoiseGenerator(self.noise)
        print(f'Set sampler: {sampler} {topk}/{nsample}')

    def set_reference(self, weak):
        self.weak = weak

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_smoothness = []  

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    if self.sampler == 'random':
                        action_dict = policy.predict_action(obs_dict)
                    elif self.sampler == 'ema':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = ema_sampler(policy, action_prior, obs_dict, self.decay)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    elif self.sampler == 'contrast':
                        action_dict = contrastive_sampler(policy, self.weak, obs_dict, self.n_samples)
                    elif self.sampler == 'coherence':
                        if 'action_prior' not in locals():
                            action_prior = None
                        if 'pred_prior' not in locals():
                            pred_prior = None
                        obs_dict['prior'] = pred_prior
                        action_dict = coherence_sampler(policy, action_prior, obs_dict, self.n_samples, self.decay, self.dist_dir)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                        # # update prior
                        # pred_prior = torch.cat((policy.normalizer['action'].normalize(action_dict['action_pred']),
                        #                         policy.normalizer['obs'].normalize(action_dict['obs_pred'])), axis=-1)
                        # pred_prior = torch.cat((pred_prior[:, self.n_action_steps:], 
                        #                         pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)), axis=1)          
                        


                        pred_prior = policy.normalizer['action'].normalize(action_dict['action_pred'])
                        pred_prior = torch.cat((pred_prior[:, self.n_action_steps:], 
                                                pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)), axis=1)          


                        # Extrapolate pred_prior for the next step
                        pred_prior = torch.cat((
                            pred_prior[:, self.n_action_steps:], 
                            pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)
                        ), axis=1)
                        # Calculate smoothness for the current trajectory
                        actions = action_dict['action_pred'].detach().cpu().numpy()
                        smoothness = compute_trajectory_smoothness(actions)
                        all_smoothness.append(smoothness)  
                        
                    elif self.sampler == 'bid':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = bidirectional_sampler(policy, self.weak, obs_dict, action_prior, self.n_samples, self.decay)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]                        
                    elif self.sampler == 'cma':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = cma_sampler(policy, action_prior, obs_dict, self.n_samples, self.decay)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    elif self.sampler == 'lowvar':
                        if 'pred_prior' not in locals():
                            pred_prior = None
                        obs_dict['prior'] = pred_prior
                        action_dict = policy.predict_action(obs_dict)
                        # set prior throughout trajectory
                        if pred_prior == None:
                            pred_prior = policy.normalizer['action'].normalize(action_dict['action_pred'])
                            pred_prior = torch.cat((pred_prior[:, self.n_action_steps:], 
                                                    pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)), axis=1)
                    elif self.sampler == 'warmstart':
                        if 'pred_prior' not in locals():
                            pred_prior = None
                        obs_dict['prior'] = pred_prior
                        action_dict = policy.predict_action(obs_dict)
                        # update prior
                        pred_prior = policy.normalizer['action'].normalize(action_dict['action_pred'])
                        pred_prior = torch.cat((pred_prior[:, self.n_action_steps:], 
                                                pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)), axis=1)                        
                    elif self.sampler == 'wma':
                        if 'action_prior' not in locals():
                            action_prior = None
                            pred_prior = None
                        obs_dict['prior'] = pred_prior
                        action_dict = ema_sampler(policy, action_prior, obs_dict, self.decay)
                        # update prior
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                        pred_prior = torch.cat((policy.normalizer['action'].normalize(action_dict['action_pred']),
                                                policy.normalizer['obs'].normalize(action_dict['obs_pred'])), axis=-1)
                        pred_prior = torch.cat((pred_prior[:, self.n_action_steps:], 
                                                pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)), axis=1)
                    elif self.sampler == 'warmcoherence':
                        if 'action_prior' not in locals():
                            action_prior = None
                            pred_prior = None
                        obs_dict['prior'] = pred_prior
                        action_dict = coherence_sampler(policy, action_prior, obs_dict, self.n_samples, self.decay)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                        # update prior
                        pred_prior = torch.cat((policy.normalizer['action'].normalize(action_dict['action_pred']),
                                                policy.normalizer['obs'].normalize(action_dict['obs_pred'])), axis=-1)
                        pred_prior = torch.cat((pred_prior[:, self.n_action_steps:], 
                                                pred_prior[:, -1].unsqueeze(1).repeat(1, self.n_action_steps, 1)), axis=1)         
                    else:
                        action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        if all_smoothness:
            avg_smoothness = np.mean(all_smoothness)
            histogram_path = os.path.join(self.output_dir, 'smoothness_histogram')
            save_histogram(all_smoothness, histogram_path)
            print(f"Average Smoothness: {avg_smoothness}")
        else:
            print("No smoothness data available.")
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
