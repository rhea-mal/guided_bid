"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

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

from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import pdb
import ipdb

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--dist_dir', default=None)
@click.option('-e', '--noise', default=0.0)
@click.option('-p', '--perturb', default=0.0)
@click.option('-ah', '--ahorizon', default=8)
@click.option('-t', '--ntest', default=200)
@click.option('-s', '--sampler', required=True)
@click.option('-n', '--nsample', default=1)
@click.option('-m', '--nmode', default=1)
@click.option('-k', '--decay', default=0.9)
@click.option('-r', '--reference', default=None)
@click.option('-g', '--guidance', default=0.0)
@click.option('-z', '--seed', default=0)

def main(checkpoint, output_dir, dist_dir, noise, perturb, ahorizon, ntest, sampler, nsample, nmode, decay, reference, seed, guidance, device='cuda:0'):
    if os.path.exists(output_dir):
        print(f"Output path {output_dir} already exists and will be overwrited.")
    if dist_dir and os.path.exists(dist_dir):
        print(f"Distance path {dist_dir} already exists and will be overwrited.")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load reference
    if reference:
        try:
            payload = torch.load(open(reference, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg, output_dir=output_dir)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            weak = workspace.model
            if cfg.training.use_ema:
                weak = workspace.ema_model
            weak.n_action_steps = ahorizon
            device = torch.device(device)
            weak.to(device)
            weak.eval()
            print('Loaded weak model')
        except Exception as e:
            weak = None
            print('Skipped weak model')

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # turn off video
    cfg.task.env_runner['n_train_vis'] = 0
    cfg.task.env_runner['n_test_vis'] = 0
    cfg.task.env_runner['n_train'] = 1
    cfg.task.env_runner['n_test'] = ntest
    cfg.task.env_runner['n_action_steps'] = ahorizon
    cfg.task.env_runner['test_start_seed'] = 20000 + 10000 * seed

    # PERTURB DISABLED FOR ROBOMIC
    # if "perturb" not in cfg.task.env_runner:
    #     OmegaConf.set_struct(cfg.task.env_runner, False)  # Temporarily disable strict mode
    #     cfg.task.env_runner.perturb = perturb
    #     OmegaConf.set_struct(cfg.task.env_runner, True)  # Re-enable strict mode
    # else:
    #     cfg.task.env_runner.perturb = perturb


    policy.n_action_steps = ahorizon
    policy.guidance_scale=guidance

    print("\nEvaluation setting:")
    print("env:")
    try:
        print(f"  delay_horizon = {cfg.task.env_runner.n_latency_steps: <19} act_horizon = {cfg.task.env_runner.n_action_steps: <15} obsv_horizon = {cfg.task.env_runner.n_obs_steps}")
    except Exception as e:
        print(f"  delay_horizon = {float('nan'): <19} act_horizon = {cfg.task.env_runner.n_action_steps: <15} obsv_horizon = {cfg.task.env_runner.n_obs_steps}")
    print("policy:")
    print(f"  pred_horizon = {policy.horizon: <20} act_horizon = {policy.n_action_steps: <15} obsv_horizon = {policy.n_obs_steps}")

    print("Hydra config for env_runner:", cfg.task.env_runner)
    
    env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir,
            max_steps=400)

    # set sampler
    env_runner.set_sampler(sampler, nsample, nmode, noise, decay, dist_dir)
    if reference and weak:
        env_runner.set_reference(weak)

    runner_log = env_runner.run(policy)
    print(f"train: {runner_log['train/mean_score']}         test: {runner_log['test/mean_score']}")

    # dump log to json
    json_log = dict()
    json_log['checkpoint'] = checkpoint
    for key, value in runner_log.items():
        if 'video' not in key:
            json_log[key] = value
    if sampler == 'random':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}.json')
    elif sampler == 'ema':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{decay}.json')
    elif sampler == 'contrast':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}.json')
    elif sampler == 'positive':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}.json')
    elif sampler == 'negative':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}.json')
    elif sampler == 'coherence':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{decay}.json')
    elif sampler == 'bid':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}_{decay}.json')
    elif sampler == 'cma':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{decay}.json')
    elif sampler == 'warmstart':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}.json')
    elif sampler == 'wma':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{decay}.json')
    elif sampler == 'cwarm':
        out_path = os.path.join(output_dir, f'eval_{seed}_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{decay}.json')
    else:
        pass
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()