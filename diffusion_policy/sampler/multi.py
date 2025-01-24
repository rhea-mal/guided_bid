import torch
from diffusion_policy.sampler.metric import euclidean_distance, coverage_distance

import pdb
torch.set_printoptions(precision=2, sci_mode=False)

def contrastive_sampler(strong, weak, obs_dict, num_sample=10, name='contrast', factor=3):
    """
    Sample an action by contrasting outputs from strong and weak policies.

    Args:
        strong: a strong policy to predict near-optimal sequences of actions
        weak: a weak policy to predict sub-optimal sequences of actions
        obs_dict: dictionary containing observations at the current time step
        num_sample (int, optional): number of samples to generate
        name (str, optional): type of samples ('contrast', 'positive', 'negative')
        factor (int, optional): Factor to determine the number of top samples to consider

    Returns:
        dict: A dictionary of actions sampled using the contrastive approach.
    """
    # pre-process
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    obs_dict_batch = {key: val.unsqueeze(1).repeat(1, num_sample, 1, 1).view(B * num_sample, OH, OD) 
                      for key, val in obs_dict.items()}

    # predict
    action_strong_batch = strong.predict_action(obs_dict_batch)
    action_weak_batch = weak.predict_action(obs_dict_batch)

    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_strong_batch:
        action_strong_batch['action_obs_pred'] = action_strong_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_strong_batch:
        action_strong_batch['obs_pred'] = action_strong_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_weak_batch:
        action_weak_batch['action_obs_pred'] = action_weak_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_weak_batch:
        action_weak_batch['obs_pred'] = action_weak_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    # positive samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand =  action_strong_batch['action_pred'].unsqueeze(2)
    dist_pos = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor + 1
    values, _ = torch.topk(dist_pos, k=topk, largest=False, dim=-1)
    dist_avg_pos = values[:, :, 1:].mean(dim=-1)      # skip the self-distance first element 

    if name == "negative": dist_avg_pos.zero_()

    # negative samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand = action_weak_batch['action_pred'].unsqueeze(2)
    dist_neg = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_neg, k=topk, largest=False, dim=-1)
    dist_avg_neg = values[:, :, 0:].mean(dim=-1)

    if name == "positive": dist_avg_neg.zero_()

    # sample selection
    dist_avg = dist_avg_pos - dist_avg_neg
    index = dist_avg.argmin(dim=-1)

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_strong_batch.keys():
        action_dict[key] = action_strong_batch[key][range_tensor, index]

    return action_dict

def bidirectional_sampler(strong, weak, obs_dict, prior, num_sample=10, beta=0.5, factor=3):
    """
    Sample an action that preserves coherence with a prior and contrast outputs from strong and weak policies.
    Args:
        strong: a strong policy to predict near-optimal sequences of actions
        weak: a weak policy to predict sub-optimal sequences of actions
        prior: the prediction made in the previous time step
        obs_dict: dictionary containing observations at the current time step
        num_sample (int, optional): number of samples to generate
        name (str, optional): type of samples ('contrast', 'positive', 'negative')
        factor (int, optional): Factor to determine the number of top samples to consider

    Returns:
        dict: A dictionary of actions sampled using the contrastive approach.
    """    
    # pre-process
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    for key in obs_dict.keys():
        if key == 'prior':
            continue        
        obs_dict_batch[key] = obs_dict[key].unsqueeze(1).repeat(1, num_sample, 1, 1).view(B * num_sample, OH, OD)

    # predict
    action_strong_batch = strong.predict_action(obs_dict_batch)
    action_weak_batch = weak.predict_action(obs_dict_batch)

    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_strong_batch:
        action_strong_batch['action_obs_pred'] = action_strong_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_strong_batch:
        action_strong_batch['obs_pred'] = action_strong_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
    action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_weak_batch:
        action_weak_batch['action_obs_pred'] = action_weak_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_weak_batch:
        action_weak_batch['obs_pred'] = action_weak_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    # positive samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand =  action_strong_batch['action_pred'].unsqueeze(2)
    dist_pos = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_pos, k=topk, largest=False, dim=-1)
    dist_avg_pos = values[:, :, 1:].mean(dim=-1)      # skip the self-distance first element 

    # negative samples
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand = action_weak_batch['action_pred'].unsqueeze(2)
    dist_neg = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    topk = num_sample // factor
    values, _ = torch.topk(dist_neg, k=topk, largest=False, dim=-1)
    dist_avg_neg = values[:, :, 0:].mean(dim=-1)

    # backward
    if prior is not None:
        # distance measure
        start_overlap = strong.n_obs_steps - 1
        end_overlap = prior.shape[1]
        weights = torch.tensor([beta**i for i in range(end_overlap-start_overlap)]).to(src_expand.device)
        weights = weights / weights.sum()

        dist_strong = euclidean_distance(action_strong_batch['action_pred'][:, :, start_overlap:end_overlap], prior.unsqueeze(1)[:, :, start_overlap:], reduction='none')
        dist_weighted = dist_strong * weights.view(1, 1, end_overlap-start_overlap)
        dist_avg_prior = dist_weighted.sum(dim=2)

        # sample selection
        dist_avg = dist_avg_prior + dist_avg_pos - dist_avg_neg
        topk = factor
        _, topk_indices = torch.topk(dist_avg_prior, k=topk, largest=False, dim=1)
        topk_dist_avg = torch.gather(dist_avg, 1, topk_indices)
        _, min_index_in_topk = torch.min(topk_dist_avg, dim=1)
        index = topk_indices[torch.arange(dist_avg.shape[0]), min_index_in_topk]
    else:
        # sample selection
        dist_avg = dist_avg_pos - dist_avg_neg
        index = dist_avg.argmin(dim=-1)

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_strong_batch.keys():
        action_dict[key] = action_strong_batch[key][range_tensor, index]

    return action_dict
